# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig, TrainerCallback
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, BBOX_START_TOKEN, BBOX_END_TOKEN, THINK_START_TOKEN, THINK_END_TOKEN, ANSWER_START_TOKEN, ANSWER_END_TOKEN
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token, resize_and_center_crop
from llava.video_utils import VideoProcessor, merge_video_dict
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord
from transformers.integrations import is_deepspeed_zero3_enabled
from llava.arguments import ModelArguments, DataArguments, TrainingArguments
from llava.text_utils import preprocess, preprocess_multimodal, get_positive_mask
from llava.utils_3d import transform_point, transform_bbox, convert_bbox_to_str, augment_point_cloud_torch
import copy
from llava.train.train_callback import ConvertToFP32Callback
import open3d as o3d


torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) and "ground_head" not in name:
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    
    rank0_print("lora_module_names", lora_module_names)
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def add_tokens_and_resize_embeddings(
    new_tokens: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, img_data_augmentation=False, pcd_data_augmentation=False, mode="train"):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []
        self.video_processor = VideoProcessor(
            data_args.data_folder,
            frame_sampling_strategy=getattr(data_args, "frame_sampling_strategy", "uniform"),
            img_data_augmentation=img_data_augmentation,
            mode=mode
        )
        self.mode = mode
        self.pcd_data_augmentation = (pcd_data_augmentation and mode=="train")
        
        self.data_args = copy.deepcopy(data_args)
        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            self.data_args.dataset_paths = []
            for file_name in file_names:
                self.data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                self.data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            self.data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        # self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # SEGA
            if sample.get("metadata", {}).get("dataset", None) in ["scanrefer", "multi3drefer", "sr3d", "nr3d", "locate3d_scannet", "vsibench_cot"]:
                length_list.append(1)
            elif sample.get("metadata", {}).get("dataset", None) in ["scanQA", "sqa3d", "vlm_3r_scannet"]:
                length_list.append(2)
            elif sample.get("metadata", {}).get("dataset", None) in ["scan2cap"]:
                length_list.append(3)
            else:
                raise ValueError("bad data")
        return length_list


    @property
    def task_lengths(self):
        task_mapping = {
            "scanqa": 0,
            "sqa3d": 0,
            "vlm_3r_scannet": 0,
            "scan2cap": 1,
            "scanrefer": 2,
            "multi3drefer": 2,
            "nr3d": 2,
            "sr3d": 2,
            "locate3d_scannet": 2,
            "vsibench_cot": 2,
        }
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            length_list.append(
                (task_mapping[sample['metadata']['dataset'].lower()], cur_len)
            )
        return length_list


    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        dataset_type = self.list_data_dict[i].get("metadata", {}).get("dataset", None)
        
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            if dataset_type not in ["vsibench_cot"]:
                video_file = self.list_data_dict[i]["video"]
            else:
                video_file = os.path.join(self.list_data_dict[i]["data_source"], self.list_data_dict[i]["scene_name"])
            
            try:
                video_dict = self.video_processor.process_3d_video(
                    video_file,
                    image_processor=self.data_args.image_processor,
                    force_sample=self.data_args.force_sample,
                    frames_upbound=self.data_args.frames_upbound,
                )
                image = video_dict.pop("images")
                video_size = video_dict.pop("video_size")


                object_metadata = None
                if dataset_type in ["scanrefer", "multi3drefer", "sr3d", "nr3d", "locate3d_scannet", "scan2cap"]:
                    object_metadata = copy.deepcopy(self.list_data_dict[i]["metadata"]["object"])
                
                # pcd data augmentation
                if self.pcd_data_augmentation and dataset_type in ["scanrefer", "multi3drefer", "sr3d", "nr3d", "locate3d_scannet", "scan2cap", "vsibench_cot"]:                    
                    world_coords = video_dict["world_coords"]
                    pcd = world_coords.flatten(0, 2)
                    
                    bbox = None
                    if dataset_type in ["scanrefer", "multi3drefer", "sr3d", "nr3d", "locate3d_scannet", "scan2cap"]:
                        bbox_lst = []
                        for obj in object_metadata:
                            for bbox in obj["object_bbox"]:
                                bbox_lst.append(bbox)
                        bbox = torch.tensor(bbox_lst) if len(bbox_lst) != 0 else None
                    
                    # use data augmentation config for dataset ["vsibench_cot"]
                    data_aug_conf = None
                    if dataset_type in ["vsibench_cot"]:
                        data_aug_conf = self.list_data_dict[i]["metadata"]["data_augmentation"]
                    
                    aug_pcd, aug_bbox = augment_point_cloud_torch(pcd, bboxes=bbox, data_aug_conf=data_aug_conf)
                    
                    video_dict["world_coords"] = aug_pcd.reshape(*world_coords.shape)
                    
                    # write augment bbox back to object_metadata
                    if dataset_type in ["scanrefer", "multi3drefer", "sr3d", "nr3d", "locate3d_scannet", "scan2cap"]:
                        if aug_bbox is not None: # for multi3drefer there may have no bbox
                            i = 0
                            aug_bbox_lst = aug_bbox.tolist()
                            for obj in object_metadata:
                                for j in range(len(obj["object_bbox"])):
                                    obj["object_bbox"][j] = aug_bbox_lst[i]
                                    i+=1

                if dataset_type not in ["vsibench_cot"]:
                    if self.data_args.add_time_instruction:
                        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                        sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                    if self.data_args.add_spatial_instruction:
                        spatial_instruction = (
                            "The video captures 3D spatial information of a scene. Your task is to focus on the spatial relationships in the video.\n"
                            "Skip the reasoning process and directly answer the question following exactly this struture:\n"
                            "<think></think><answer>[Final answer here]</answer>\n"
                            "\n"
                            "Question: "
                        )
                        sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{spatial_instruction}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'

                image = [(image, video_size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                raise e
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # substitute <ground> token with real bbox
        if dataset_type in ["scanrefer", "multi3drefer", "sr3d", "nr3d", "locate3d_scannet", "scan2cap"]:
            boxes_aabb_str = ""
            assert object_metadata is not None, "These datasets should contain object_metadata"
            for i, obj in enumerate(object_metadata):
                obj_bbox_lst = torch.tensor(obj["object_bbox"])

                if len(obj_bbox_lst) != 0:
                    min_xyz = obj_bbox_lst[:, :3] - obj_bbox_lst[:, 3:] / 2
                    max_xyz = obj_bbox_lst[:, :3] + obj_bbox_lst[:, 3:] / 2
                    
                    obj_bbox_aabb = torch.cat([min_xyz, max_xyz], dim=-1).reshape(-1, 6)
                    obj_bbox_str_lst = convert_bbox_to_str(obj_bbox_aabb)

                    # shuffle box here to learn more robust grounding
                    random.shuffle(obj_bbox_str_lst)
                    
                    if dataset_type != "scan2cap":
                        boxes_aabb_str += f"{obj['object_name']} {len(obj_bbox_str_lst)} "
                    for box_str in obj_bbox_str_lst:
                        boxes_aabb_str += f"{BBOX_START_TOKEN}{box_str}{BBOX_END_TOKEN}"
                else:
                    boxes_aabb_str = f"{BBOX_START_TOKEN}{BBOX_END_TOKEN}"
                if i != len(object_metadata)-1:
                    boxes_aabb_str += ", "    
            
            for conv in sources[0]:
                if "<ground>" in conv["value"]:
                    if dataset_type != "scan2cap":
                        conv["value"] = conv["value"].replace("<ground>", f"{THINK_START_TOKEN}{THINK_END_TOKEN}{ANSWER_START_TOKEN}{boxes_aabb_str}{ANSWER_END_TOKEN}")
                    else:
                        conv["value"] = conv["value"].replace("<ground>", boxes_aabb_str)

        if dataset_type in ["scanQA", "sqa3d", "scan2cap"]:
            sources[0][1]["value"] = f"{THINK_START_TOKEN}{THINK_END_TOKEN}{ANSWER_START_TOKEN}{sources[0][1]['value']}{ANSWER_END_TOKEN}"


        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        # build text token input for LLM from plain text conversation
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
            data_dict["video_dict"] = video_dict
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)
        
        positive_token_masks = instances[0].get("positive_token_masks", None)
        if positive_token_masks is not None:
            batch["positive_token_masks"] = positive_token_masks.unsqueeze(0)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images
            if "video_dict" in instances[0]:
                batch["video_dict"] = merge_video_dict([instance["video_dict"] for instance in instances])
                

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        
        if "box_label" in instances[0]:
            batch["box_labels"] = [instance["box_label"]  for instance in instances]
            batch["use_object_proposals"] = True

        if "gt_box" in instances[0]:
            batch["gt_box"] = [instance["gt_box"] for instance in instances]


        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path, data_args=data_args, img_data_augmentation=data_args.img_data_augmentation, pcd_data_augmentation=data_args.pcd_data_augmentation, mode="train")
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path, data_args=data_args, img_data_augmentation=False, pcd_data_augmentation=False, mode="val")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode
    
    # configs for 3d tasks
    if model_args.object_feature_type is not None:
        overwrite_config["object_feature_type"] = model_args.object_feature_type

    if model_args.world_position_embedding_type is not None:
        overwrite_config["world_position_embedding_type"] = model_args.world_position_embedding_type
        overwrite_config["voxel_size"] = model_args.voxel_size
        """
            ============== x_min ================
            min: -11.505682945251465
            min@95: -7.609058666229248
            min@90: -6.1256390571594235
            ============== y_min ================
            min: -13.966400146484375
            min@95: -9.432173728942871
            min@90: -7.719909286499023
            ============== z_min ================
            min: -7.088255405426025
            min@95: -0.9921181917190549
            min@90: -0.4241578817367552
            ============== x_max ================
            max: 10.725899696350098
            max@95: 7.726488876342769
            max@90: 6.042660427093505
            ============== y_max ================
            max: 14.70956039428711
            max@95: 8.782552528381343
            max@90: 7.462656307220459
            ============== z_max ================
            max: 7.892434597015381
            max@95: 3.281849384307861
            max@90: 3.0370256423950197
        """
        overwrite_config["min_xyz_range"] = [-15, -15, -5]
        overwrite_config["max_xyz_range"] = [15, 15, 5]

    
    if model_args.ground_head_type is not None:
        overwrite_config["ground_head_type"] = model_args.ground_head_type
        overwrite_config["ground_head_hidden_size"] = model_args.ground_head_hidden_size
        overwrite_config["ground_loss_scale"] = model_args.ground_loss_scale
        overwrite_config["ground_head_temperature"] = model_args.ground_head_temperature

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
    
    model_args.voxel_size = data_args.voxel_size
    model_args.min_xyz_range = data_args.min_xyz_range
    model_args.max_xyz_range = data_args.max_xyz_range


    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)

    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    # add grounding tokens
    from llava.constants import THINK_START_TOKEN, THINK_END_TOKEN, ANSWER_START_TOKEN, ANSWER_END_TOKEN, BBOX_START_TOKEN, BBOX_END_TOKEN
    bbox_token = [BBOX_START_TOKEN, BBOX_END_TOKEN, THINK_START_TOKEN, THINK_END_TOKEN, ANSWER_START_TOKEN, ANSWER_END_TOKEN]
    add_tokens_and_resize_embeddings(
        bbox_token,
        tokenizer=tokenizer,
        model=model
    )

    if model_args.world_position_embedding_type is not None:
        if "sonata" in model_args.world_position_embedding_type:
            model.get_model().initialize_spatial_encoder(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
         
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        ### Deciding train which part of the model
        if training_args.lora_enable:
            # We update embed_tokens as the size of embedding has changed.
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lora_" in name:
                    param.requires_grad_(True)
        else:
            if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
                model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
                model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
                if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                    model.requires_grad_(False)
                if model_args.tune_mm_mlp_adapter:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = True
                if model_args.tune_mm_vision_resampler:
                    for p in model.get_model().vision_resampler.parameters():
                        p.requires_grad = True

                model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
                if training_args.freeze_mm_mlp_adapter:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = False

                model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
                if training_args.freeze_mm_vision_resampler:
                    for p in model.get_model().vision_resampler.parameters():
                        p.requires_grad = False

                model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
                if model_args.unfreeze_mm_vision_tower:
                    vision_tower.requires_grad_(True)
                else:
                    vision_tower.requires_grad_(False)

            else:
                rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
                model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
                # Set the entire model to not require gradients by default
                model.requires_grad_(False)
                vision_tower.requires_grad_(False)
                model.get_model().mm_projector.requires_grad_(False)
                model.get_model().vision_resampler.requires_grad_(False)
                if model_args.world_position_embedding_type is not None:
                    if "sonata" in model_args.world_position_embedding_type:
                        spatial_encoder = model.get_model().spatial_encoder
                        spatial_encoder.requires_grad_(False)
                        for param in spatial_encoder.parameters():
                            param.requires_grad = False
                
                # Parse the mm_tunable_parts to decide which parts to unfreeze
                tunable_parts = model_args.mm_tunable_parts.split(",")
                if "mm_mlp_adapter" in tunable_parts:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = True
                if "mm_vision_resampler" in tunable_parts:
                    for p in model.get_model().vision_resampler.parameters():
                        p.requires_grad = True
                if "mm_vision_tower" in tunable_parts:
                    for name, param in model.named_parameters():
                        if "vision_tower" in name:
                            param.requires_grad_(True)
                if "spatial_encoder" in tunable_parts:
                    for p in model.get_model().spatial_encoder.parameters():
                        p.requires_grad = True
                if "mm_language_model" in tunable_parts:
                    for name, param in model.named_parameters():
                        if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name and "spatial_encoder" not in name:
                            param.requires_grad_(True)

        other_opt_param_name = []

        if model_args.world_position_embedding_type is not None:
            if "sonata" in model_args.world_position_embedding_type and "spatial_encoder" in tunable_parts:
                other_opt_param_name += ["mm_spatial_projector", "world_position_embedding", "pointnet_pooling", "ada_crossattn"]
            else:
                other_opt_param_name.append("world_position_embedding")
        
        if len(other_opt_param_name) != 0:
            for name, param in model.named_parameters():
                    for opn in other_opt_param_name:
                        if opn in name:
                            param.requires_grad_(True)

        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        # rank0_print('trainable params', [n for n, p in model.named_parameters() if p.requires_grad])
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    training_args.label_names = ["labels"] # used for not include box label for evaluation
    callbacks = []

    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module, callbacks=callbacks)
    
    if training_args.resume_from_checkpoint is not None:
        trainer._load_from_checkpoint(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and training_args.resume_from_checkpoint is None:
        trainer.train(resume_from_checkpoint=True)
    else:
        try:
            trainer.train()
        except Exception as e:
            import traceback; traceback.print_exc()
            exit()
        trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
