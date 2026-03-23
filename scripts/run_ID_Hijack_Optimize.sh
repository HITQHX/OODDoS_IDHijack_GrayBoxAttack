#!/bin/bash
# 联合优化的域内动作流形扰乱攻击 (算法 4)

python VLAAttacker/ID_Hijack_Optimize_wrapper.py \
    --server /root/autodl-tmp/roboticAttack \
    --dataset "libero_spatial" \
    --bs 16 \
    --iter 3000 \
    --lr 0.03 \
    --device 0 \
    --patch_h 56 \
    --patch_w 56 \
    --gamma1 1 \
    --gamma2 1 \
    --exec_pool_dir "/root/autodl-tmp/roboticAttack/malicious_exec_pool_libero_spatial_1" \
    --xmin 0 \
    --ymin 196 \
    --xmax 224 \
    --ymax 224