#!/bin/bash

# Set up the data folder
DATA_FOLDER="data/processed_data"
TRAIN_DATA_YAML="scripts/3d/train/train_multi_finetune.yaml"
EVAL_DATA_YAML="scripts/3d/train/eval_multi_finetune.yaml"

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

################ Arnold Jobs ################

LLM_VERSIO="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="data/models/siglip-so400m-patch14-384"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-qwen-gsreasoner-frame32-finetune-bs256"


PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"
# PREV_STAGE_CHECKPOINT
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

NUM_GPUS=8
BATCH_SIZE=256

N_NODES=$((NUM_GPUS/8))
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/NUM_GPUS))

torchrun --nnodes ${N_NODES} --nproc_per_node=8 --master_port=43000 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --train_data_path $TRAIN_DATA_YAML \
    --eval_data_path $EVAL_DATA_YAML \
    --data_folder $DATA_FOLDER \
    --mm_tunable_parts="mm_mlp_adapter,spatial_encoder,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --load_best_model_at_end False \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --output_dir ./ckpt/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type sonata_sample_64 \
    --object_feature_type patch14-pe \
    --ground_head_type autoreg \
    --group_by_task_length True \
    --frame_sampling_strategy uniform \
    --frames_upbound 32 \
    --report_to tensorboard \
    --attn_implementation flash_attention_2 \
    --pcd_data_augmentation True \
    --resume_from_checkpoint ckpt/llavanext-qwen-gsreasoner-randomframe48-pretrain-bs16 \
    > "./ckpt/${MID_RUN_NAME}.log" 2>&1
exit 0;