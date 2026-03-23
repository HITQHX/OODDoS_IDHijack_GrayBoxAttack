import cv2
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Extract mid-execution frames from a robotic task video")
    parser.add_argument('--video_path', type=str,  help="目标任务的 mp4 视频路径")
    parser.add_argument('--save_dir', type=str, default="./malicious_exec_pool_libero_10", help="保存提取帧的目录")
    parser.add_argument('--num_frames', type=int, default=40, help="想要提取的帧数")
    # 截掉开头 30%（机械臂还在寻路）和结尾 10%（已经抓完收回了），只留中间最核心的操作段
    parser.add_argument('--skip_start_ratio', type=float, default=0.05, help="跳过视频开头的比例")
    parser.add_argument('--skip_end_ratio', type=float, default=0.40, help="跳过视频结尾的比例")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.video_path = "/root/autodl-tmp/roboticAttack/run/ID_Hijack/libero_spatial/20260317_184157/eval_results/best/libero_10/2026_03_18-10_22_35--episode=1--success=True--task=put_both_the_alphabet_soup_and_the_tomato_sauce_in.mp4"
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {args.video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 视频总帧数: {total_frames}")

    # 计算我们要截取的核心片段范围
    start_frame = int(total_frames * args.skip_start_ratio)
    end_frame = int(total_frames * (1.0 - args.skip_end_ratio))
    print(f"✂️ 截取核心动作区间: 第 {start_frame} 帧 -> 第 {end_frame} 帧")

    # 在核心片段中均匀采样 num_frames 个索引
    target_indices = np.linspace(start_frame, end_frame, args.num_frames, dtype=int)
    
    saved_count = 0
    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV 默认读入是 BGR，保存为 PNG
            save_path = os.path.join(args.save_dir, f"exec_frame_{idx:04d}.png")
            cv2.imwrite(save_path, frame)
            saved_count += 1
            print(f"✅ 已保存特征帧: {save_path}")
        else:
            print(f"⚠️ 无法读取第 {idx} 帧")

    cap.release()
    print(f"🎉 抽帧完成！共提取 {saved_count} 张高能状态帧至 {args.save_dir}")

if __name__ == '__main__':
    main()