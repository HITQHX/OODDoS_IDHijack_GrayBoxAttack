#!/bin/bash
# 攻击二：域内语义定向劫持 (ID Semantic Hijack)
# 特点：热点扫描，50x50 局部锚定，纯视觉特征对齐，目标是定向诱导轨迹
# 注意：执行前请确保 target_frames/ 目录下存在你提取的首帧数据集，且包含 target_scene.png 作为恶意目标靶点
python VLAAttacker/ID_Hijack_wrapper.py \
    --lr 3e-2 \
    --server /root/autodl-tmp/roboticAttack \
    --device 0 \
    --iter 3000 \
    --accumulate 1 \
    --bs 16 \
    --dataset "libero_10" \
    --patch_h 56 \
    --patch_w 56 \
    --x 28 \
    --y 168