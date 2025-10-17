## ------------------------------------------------- eval scanqa -------------------------------------------------#
bash scripts/3d/eval/eval_3dvqa.sh GS-Reasoner uniform 32 scanqa scanqa_val_llava_style

## ------------------------------------------------- eval sqa -------------------------------------------------#
bash scripts/3d/eval/eval_3dvqa.sh GS-Reasoner uniform 32 sqa3d sqa3d_test_llava_style

## ------------------------------------------------- eval scan2cap -------------------------------------------------#
bash scripts/3d/eval/eval_3dvqa.sh GS-Reasoner uniform 32 scan2cap scan2cap_val_llava_style




