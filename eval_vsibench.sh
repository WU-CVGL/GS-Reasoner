export LMMS_EVAL_LAUNCHER="accelerate" 

# evaluate using pred depth
accelerate launch \
  --num_processes=8 \
  -m llava.submodules.lmms_eval \
  --model gs_reasoner \
  --model_args "pretrained=data/models/GS-Reasoner,conv_template=qwen_1_5,max_frames_num=32,data_folder=data/vggt_slam_processed_data/" \
  --tasks vsibench_cot \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix gs-reasoner \
  --output_path results/vsibench

# evaluate using gt depth
accelerate launch \
  --num_processes=1 \
  -m llava.submodules.lmms_eval \
  --model gs_reasoner \
  --model_args "pretrained=data/models/GS-Reasoner,conv_template=qwen_1_5,max_frames_num=32,data_folder=data/processed_data/" \
  --tasks vsibench_cot \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix gs-reasoner \
  --output_path results/vsibench
