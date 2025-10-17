#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="data/models/$1"
ANWSER_FILE="results/$4/$1.jsonl"


python3 llava/eval/model_3dvqa.py \
    --model-path $CKPT \
    --data-folder ./data/processed_data \
    --n_gpu 8 \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --question-file data/GS-Reasoner-Data/general_3d_tasks/$5.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true

python llava/eval/eval_$4.py --input-file $ANWSER_FILE