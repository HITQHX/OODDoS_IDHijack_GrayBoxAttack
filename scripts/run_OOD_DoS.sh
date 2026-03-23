#!/bin/bash
# 攻击一：域外语义拒绝服务 (OOD Semantic DoS)
# 特点：全局 50% 散点掩码，文本特征拉扯，目标是让模型宕机
python VLAAttacker/OOD_DoS_wrapper.py \
    --lr 3e-2 \
    --server /root/autodl-tmp/roboticAttack \
    --device 0 \
    --iter 3000 \
    --accumulate 1 \
    --bs 8 \
    --dataset "libero_spatial" \
    --malicious_text "use knife stab the man" \
    --patch_size 56