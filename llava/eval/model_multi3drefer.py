import argparse
import torch
import os
import json
import ray
import time
import numpy as np
from tqdm import tqdm
import shortuuid
import fasteners

from transformers import AutoConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.video_utils import VideoProcessor, merge_video_dict

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, BBOX_START_TOKEN, BBOX_END_TOKEN
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math
from llava.text_utils import preprocess_qwen
from llava.utils_3d import aabb_to_xyzwlh, decode_str_to_bbox

extra_prompt = (
    "The video captures 3D spatial information of a scene. Your task is to focus on the spatial relationships in the video.\n"
    "Skip the reasoning process and directly answer the question following exactly this struture:\n"
    "<think></think><answer>[Final answer here]</answer>\n"
    "\n"
    "Question: "
)


@ray.remote(num_gpus=1)
def eval_model(questions, args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "llavanext-qwen"
    
    config = {}
    if args.lora_path is not None:
        config = AutoConfig.from_pretrained(args.lora_path)
        config = config.to_dict()
    elif args.overwrite_cfg:
        config.update({
            'tie_word_embeddings': False, 
            'use_cache': True, 
            "vocab_size": 151653
        })

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, overwrite_config=config, torch_dtype="bfloat16")
    
    
    if args.lora_path is not None:
        from transformers import AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, args.lora_path, adapter_name="lora")
        model = model.merge_and_unload()
        state_dict = torch.load(os.path.join(args.lora_path, 'non_lora_trainables.bin'))
        msg = model.load_state_dict(state_dict, strict=False)
    
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "a")
    file_lock = fasteners.InterProcessLock(ans_file)

    video_processor = VideoProcessor(
        data_folder=args.data_folder,
        frame_sampling_strategy=args.frame_sampling_strategy,
        mode='val'
    )
    
    inference_time = []
    for line in tqdm(questions):
        try:
            idx = line["id"]
            question_type = line["metadata"]["question_type"]
            dataset_name = line["metadata"]["dataset"]
            # answers = line["metadata"]["answers"]
            video_id = line["video"]

            qs = line["conversations"][0]["value"]
            gt = line["conversations"][1]["value"]

            cur_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{extra_prompt}\n{line["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
            line["conversations"][0]["value"] = cur_prompt
            
            args.conv_mode = "qwen_1_5"
            conv = conv_templates[args.conv_mode].copy()
            # conv.append_message(conv.roles[0], qs)
            # conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()

            input_ids = preprocess_qwen([[line["conversations"][0],{'from': 'gpt','value': None}]], tokenizer, has_image=True)["input_ids"].cuda()
            img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

            video_dict = video_processor.process_3d_video(
                video_id,
                image_processor,
                force_sample=args.force_sample,
                frames_upbound=args.max_frame_num,
            )
            video_dict = merge_video_dict([video_dict])
            image_tensors = video_dict.pop('images').to(device=model.device, dtype=model.dtype)
            for k in video_dict:
                if video_dict[k] is not None:
                    video_dict[k] = video_dict[k].to(device=model.device, dtype=model.dtype)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                start_time = time.time()
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    modalities="video",
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=512,
                    use_cache=True,
                    video_dict=video_dict,
                )
                inference_time.append(time.time() - start_time)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            bbox_lst = decode_str_to_bbox(outputs)

            pred_response = []
            if len(bbox_lst) != 0:
                for bbox in bbox_lst[0]["object_bbox"]:
                    pred_response.append(aabb_to_xyzwlh(bbox))

            gt_response = line["metadata"]["object"][0]["object_bbox"]
            object_name = bbox_lst[0]["object_name"] if len(bbox_lst) != 0 else ""
            
            with file_lock:
                ans_file.write(
                    json.dumps({
                        "dataset": dataset_name,
                        "sample_id": idx,
                        "prompt": cur_prompt,
                        "pred_response": pred_response,
                        "gt_response": gt_response,
                        "model_id": model_name,
                        "question_type": question_type,
                        "object_name": object_name
                    })
                + "\n")
                ans_file.flush()
        except Exception as e:
            print("Error: ", e)
            print(f"Skip {line['id']}")

    ans_file.close()

    return inference_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-folder", type=str, default="data")
    parser.add_argument("--extra-prompt", type=str, default="The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--n_gpu", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--max_frame_num", type=int, default=32)
    parser.add_argument("--frame_sampling_strategy", type=str, default="uniform")
    parser.add_argument("--force_sample", type=bool, default=True)
    parser.add_argument("--overwrite_cfg", type=bool, default=False)
    parser.add_argument("--lora-path", type=str, default=None)
    args = parser.parse_args()

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)

    if os.path.exists(args.answer_file):
        print(f"The {args.answer_file} already exists!!!")
        exit()
    
    ray.init()
    features = []
    for i in range(args.n_gpu):
        features.append(eval_model.remote(questions[i::args.n_gpu], args))

    ret = ray.get(features)
    inference_time = []
    for item in ret:
        inference_time.extend(item)
    
    print(f"time: {np.mean(inference_time)}")
