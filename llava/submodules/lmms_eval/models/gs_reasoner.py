import copy
import math
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav


from llava.text_utils import preprocess_qwen
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
        ANSWER_START_TOKEN,
        ANSWER_END_TOKEN
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError:
    eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")


try:
    from llava.model.language_model.llava_qwen import LlavaQwenConfig

    AutoConfig.register("llava_qwen", LlavaQwenConfig)
except ImportError:
    eval_logger.debug("No Qwen for llava vid")

from llava.model.language_model.llava_llama import LlavaConfig

AutoConfig.register("llava_llama", LlavaConfig)

from llava.video_utils import VideoProcessor, merge_video_dict, unproject
import cv2



def build_video_dict(images, depths, poses, intrinsics, image_processor):
    # unproject depth to world coords
    axis_align_matrix = torch.eye(4).float()
    depth_intrinsics = torch.tensor(intrinsics).reshape(-1, 4, 4)
    if depth_intrinsics.shape[0] == 1:
        depth_intrinsics = depth_intrinsics.repeat(len(images), 1, 1)
    depths = torch.tensor(depths)   # (V, H, W)
    poses = torch.tensor(poses).reshape(-1, 4, 4)
    poses = torch.stack([axis_align_matrix @ pose for pose in poses])     # (V, 4, 4)
    world_coords = unproject(depth_intrinsics.float(), poses.float(), depths.float())    # (V, H, W, 3)
    
    images = [img.convert("RGB") for img in images]

    V, H, W, _ = world_coords.shape
    crop_size = 384
    # Determine which dimension to use as the reference for scaling
    if H <= W:  # height <= width: scale based on height
        new_height = crop_size
        new_width = int(W * (crop_size / H))
    else:  # height > width: scale based on width
        new_width = crop_size
        new_height = int(H * (crop_size / W))
    
    # Note: Do not require the orignal image size to be same as world_coords size, only require the same aspect ratio
    images = [frame.resize((new_width, new_height)) for frame in images]
    resized_coords = [cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST) for coords in world_coords]

    # Calculate the position and perform the center crop
    left = (new_width - crop_size) // 2
    right = left + crop_size
    top = (new_height - crop_size) // 2
    bottom = top + crop_size
    images = [frame.crop((left, top, right, bottom)) for frame in images]
    resized_coords = [coords[top:bottom, left:right, :] for coords in resized_coords]

    world_coords = torch.from_numpy(np.stack(resized_coords))
    images = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]

    return {
        "images": images,
        "world_coords": world_coords
    }


@register_model("gs_reasoner")
class GS_Reasoner(lmms):
    """
    GS_Reasoner Model
    """

    def __init__(
        self,
        pretrained: str = "data/models/LLaVA-Video-7B-Qwen2",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        conv_template="qwen_1_5",
        use_cache=True,
        max_frames_num: int = 3,
        overwrite: bool = True,
        video_decode_backend: str = "pyav",
        data_folder="data/processed_data",
        frame_sampling_strategy="uniform",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            
        self.pretrained = pretrained
        self.model_name = "llavanext-qwen"
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.max_frames_num = int(max_frames_num)

        if self.overwrite == True:
            overwrite_config = {
                'tie_word_embeddings': False, 
                'use_cache': True, 
                "vocab_size": 151653     
            }
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map, overwrite_config=overwrite_config, torch_dtype="bfloat16")
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                self.model_name,
                device_map=self.device_map,
                torch_dtype="bfloat16",
            )

        self._config = self._model.config
        self.model.eval()
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.video_processor = VideoProcessor(
            data_folder=data_folder,
            frame_sampling_strategy=frame_sampling_strategy,
            mode="val"
        )
            
        
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        # fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    
    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
                
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            contexts = f"<image>\n{contexts}"
            
            if "vsibench" in task:
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])

                video_dict = self.video_processor.process_3d_video(
                        visuals,
                        self._image_processor,
                        force_sample=True,
                        frames_upbound=self.max_frames_num,
                )

            video_dict = merge_video_dict([video_dict])
            image_tensors = video_dict.pop('images').to(device=self.model.device, dtype=self.model.dtype)
            for k in video_dict:
                if video_dict[k] is not None:
                    video_dict[k] = video_dict[k].to(device=self.model.device, dtype=self.model.dtype)
           
            conv = conv_templates[self.conv_template].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2c
            keywords = [stop_str]

            input_ids = preprocess_qwen([[{'from': 'human', 'value': contexts}, {'from': 'gpt','value': None}]], self.tokenizer, has_image=True)["input_ids"].to(device=self.model.device)
            
            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    modalities="video",
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    video_dict=video_dict,
                )
            
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            
            res.append(outputs)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            # if visuals.split("/")[0] not in ["scannet", "scannetpp"]: 
            #     res.append("")
            #     continue
            
            video_dict = self.video_processor.process_3d_video(
                    visuals,
                    self._image_processor,
                    force_sample=True,
                    frames_upbound=self.max_frames_num,
            )
            video_dict = merge_video_dict([video_dict])
            image_tensors = video_dict.pop('images').to(device=self.model.device, dtype=self.model.dtype)
            for k in video_dict:
                if video_dict[k] is not None:
                    video_dict[k] = video_dict[k].to(device=self.model.device, dtype=self.model.dtype)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            
            data_dict = preprocess_qwen([[{'from': 'human', 'value': contexts}, {'from': 'gpt','value': continuation}]], self.tokenizer, has_image=True)
            
            input_ids = data_dict["input_ids"].to(device=self.model.device)
            labels = data_dict["labels"].to(device=self.model.device)
            
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image_tensors, modalities="video", video_dict=video_dict)

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[labels!=IGNORE_INDEX]
            greedy_tokens = greedy_tokens[labels!=IGNORE_INDEX]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res