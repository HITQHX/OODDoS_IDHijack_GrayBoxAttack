"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys

# =========================================================================
# 🔴 终极路径黑客与底层静音
# =========================================================================
# 屏蔽 TensorFlow 底层 C++ 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

LIBERO_DIR = "/root/autodl-tmp/LIBERO" 
if LIBERO_DIR not in sys.path:
    sys.path.insert(0, LIBERO_DIR)

WHITE_PATCH_DIR = "/root/autodl-tmp/roboticAttack/VLAAttacker/white_patch"
if WHITE_PATCH_DIR not in sys.path:
    sys.path.insert(0, WHITE_PATCH_DIR)

ROBOTIC_ATTACK_DIR = "/root/autodl-tmp/roboticAttack"
if ROBOTIC_ATTACK_DIR not in sys.path:
    sys.path.insert(0, ROBOTIC_ATTACK_DIR)
# =========================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import torch
import random
import argparse
import torchvision
# 忽略因为动态导入可能产生的 Pylance 警告
from libero.libero import benchmark
import wandb
from appply_random_transform import RandomPatchTransform

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from PIL import Image
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def eval_libero(cfg) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    randomPatchTransform = RandomPatchTransform('cpu', False)
    patch = torch.load(cfg.patchroot)
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    
    # 修正保存路径：如果上层传了 eval_dir 就用它，否则用默认的
    actual_log_dir = cfg.eval_dir if cfg.eval_dir else cfg.local_log_dir
    os.makedirs(actual_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(actual_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    
    print(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        # initial_states = task_suite.get_task_init_states(task_id)
        
        # 修改后：拦截加载错误，启用动态生成模式
        try:
            initial_states = task_suite.get_task_init_states(task_id)
        except FileNotFoundError:
            print(f"⚠️ [系统提示] 找不到任务 {task_id} 的 .init 缓存文件。已切换至完全依赖 BDDL 的动态随机生成模式！")
            initial_states = None
        
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=224)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            # obs = env.set_init_state(initial_states[episode_idx]) 

            if initial_states is not None:
                try:
                    init_state = initial_states[episode_idx]
                    env.set_init_state(init_state)
                except ValueError as e:
                    print(f"⚠️ [系统提示] 状态维度不匹配（已注入自定义实体 Patch），跳过强制状态覆盖！错误信息: {e}")

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 193  
            elif cfg.task_suite_name == "libero_object":
                max_steps = 254  
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 270  
            elif cfg.task_suite_name == "libero_10":
                max_steps = 505  
            elif cfg.task_suite_name == "libero_90":
                max_steps = 373  
                
            
            SAFE_ZONES = [
                        [10, 170, 80, 240],   # 左下角安全区
                        [170, 170, 240, 240], # 右下角安全区
                        [90, 190, 160, 250]   # 正下方近端安全区
                    ]
            current_safe_zone = random.choice(SAFE_ZONES)
            current_fixed_xy = None

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            # flash_target_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png" # 换成你提取的那张首帧
            # flash_img_pil = Image.open(flash_target_path).convert("RGB").resize((224, 224))
            # flash_img_np = np.array(flash_img_pil)[..., ::-1].copy() # 转换为 BGR (LIBERO 默认格式) 或者 RGB，取决于你框架的默认读取

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size) 
                    
                    img, current_fixed_xy = randomPatchTransform.simulation_table_patch_single(
                        img, patch, current_safe_zone, fixed_xy=current_fixed_xy
                    )
                    
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    
                    # dummy_task_description = "a"*len(task_description)
                    # dummy_task_description = "put the cream cheese in the bowl"

                    # print(f"replace the task description to: {dummy_task_description}")
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor
                    )

                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            print(f"Saving replay video...")
            save_rollout_video(
                replay_images, total_episodes, success=done, 
                task_description=task_description, log_file=log_file, exp_name=cfg.exp_name,
                eval_dir=actual_log_dir  # 🔴 将视频保存在我们指定的目录下
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            })

    log_file.close()

    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)
        
    # 保存总体统计结果
    summary_path = os.path.join(actual_log_dir, "success_rate_summary.txt")
    with open(summary_path, "a") as file:
        file.write(f"success_rate/total:{float(total_successes) / float(total_episodes)}, num_episodes/total:{total_episodes} position_info:{cfg.angle}_{cfg.shx}_{cfg.shy}_{cfg.x}_{cfg.y} \n")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate configuration for model training/evaluation")
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family")
    parser.add_argument("--exp_name", type=str, default=f"libero_object", help="Model family")
    parser.add_argument("--pretrained_checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-object", help="Pretrained checkpoint path")
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--center_crop", type=str2bool, default=True, help="Center crop?")
    parser.add_argument("--task_suite_name", type=str, default="libero_object", help="Task suite")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Number of steps to wait")
    parser.add_argument("--num_trials_per_task", type=int, default=5, help="Number of rollouts per task")
    parser.add_argument("--run_id_note", type=str, default=f"test_libero_object", help="Extra note")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs", help="Local directory")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Whether to log to W&B")
    parser.add_argument("--wandb_project", type=str, default="LIBERO", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default="entity", help="W&B entity")
    parser.add_argument("--seed", type=int, default=7, help="Random Seed")
    parser.add_argument("--patchroot", type=str, required=True, help="Path to patch.pt")
    parser.add_argument("--x", type=int, default=120)
    parser.add_argument("--y", type=int, default=160)
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--shx", type=float, default=0)
    parser.add_argument("--shy", type=float, default=0)
    parser.add_argument("--cudaid", type=int, default=0)
    
    # 🔴 关键修复：添加上层脚本传递的缺失参数
    parser.add_argument("--eval_dir", type=str, default="", help="Directory to save evaluation results and videos")
    parser.add_argument("--geometry", type=str2bool, default=True, help="Apply geometry transformations to patch")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    eval_libero(args)