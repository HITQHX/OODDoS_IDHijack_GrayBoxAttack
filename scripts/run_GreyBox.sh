#!/bin/bash
current_dir=$(pwd)
echo "Current directory: $current_dir"

# 注意：这里和 run_UADA.sh 一样，假设您是在代码仓库的根目录下运行该脚本
# 即执行: bash scripts/run_GreyBox.sh
python VLAAttacker/GreyBox_wrapper.py \
    --maskidx 0 \
    --lr 1e-2 \
    --server $current_dir \
    --device 0 \
    --iter 2000 \
    --accumulate 1 \
    --bs 10 \
    --warmup 20 \
    --tags "greybox semantic_jailbreak" \
    --filterGripTrainTo1 false \
    --geometry true \
    --patch_size "3,50,50" \
    --wandb_project "false" \
    --wandb_entity "xxx" \
    --innerLoop 10 \
    --dataset "libero_spatial" \
    --malicious_text "pick up the black bowl between the plate and the ramekin and place it on the plate" \
    --beta 1.0 \
    --gamma 0.5 \
    --tau 5.0
