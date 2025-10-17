#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="data/models/$1"
ANWSER_FILE="results/$4/$1.jsonl"

python3 llava/eval/model_multi3drefer.py \
    --model-path $CKPT \
    --data-folder ./data/processed_data \
    --n_gpu 8 \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --question-file data/GS-Reasoner-Data/3d_visual_grounding/multi3drefer_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true

python llava/eval/eval_multi3drefer.py --input-file $ANWSER_FILE