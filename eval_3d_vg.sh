
# ## ------------------------------------------------- eval scanrefer autoreg -------------------------------------------------#
bash scripts/3d/eval/eval_singlerefer.sh GS-Reasoner uniform 32 scanrefer scanrefer_val_llava_style

# ------------------------------------------------- eval multi3drefer autoreg -------------------------------------------------#
bash scripts/3d/eval/eval_multi3drefer.sh GS-Reasoner uniform 32 multi3drefer

# ## ------------------------------------------------- eval sr3d autoreg -------------------------------------------------#
bash scripts/3d/eval/eval_singlerefer.sh GS-Reasoner uniform 32 sr3d sr3d_test_llava_style

# ## ------------------------------------------------- eval nr3d autoreg -------------------------------------------------#
bash scripts/3d/eval/eval_singlerefer.sh GS-Reasoner uniform 32 nr3d nr3d_val_llava_style
