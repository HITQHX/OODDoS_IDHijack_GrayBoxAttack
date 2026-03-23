#!/bin/bash
#python evaluation_tool/eval_queue_single_four_spec.py \
#    --exp_path PATH TO/fe28658a-4a27-4ffa-82c4-94d44ffc9d48 \
#    --cudaid 0 \
#    --trials 50 \
#    --max_concurrent_tasks 1 \
#    --task libero_10 \


#!/bin/bash

# 将此处的路径替换为您上一步运行 bash run_GreyBox.sh 时实际生成的文件夹
# 例如: run/GreyBox/GreyBox_bridge_orig_lr1e-02_20260228_123456_abcd1234
TARGET_EXP_PATH="run/ID_Hijack_Optimize/libero_spatial/20260323_112236"

python evaluation_tool/eval_queue_single_four_spec.py \
    --exp_path $TARGET_EXP_PATH \
    --iter_folder "best" \
    --cudaid 0 \
    --trials 5 \
    --max_concurrent_tasks 1 \
    --task libero_spatial
