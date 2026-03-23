import cv2
import os
import glob

def extract_first_frame(video_path, output_path):
    """
    从指定的 MP4 视频中提取第一帧并保存为 PNG
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return False
    
    # 读取第一帧 (success 是布尔值，frame 是 numpy 数组)
    success, frame = cap.read()
    
    if success:
        # OpenCV 默认使用 BGR 色彩空间，保存为图片不需要转换，imwrite会自动处理
        cv2.imwrite(output_path, frame)
        print(f"✅ 成功提取首帧并保存至: {output_path}")
    else:
        print(f"❌ 无法读取视频首帧: {video_path}")
        
    # 释放资源
    cap.release()
    return success

def batch_extract(video_dir, output_dir):
    """
    批量提取目录下的所有 mp4 视频的首帧
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建了输出文件夹: {output_dir}")

    # 寻找目录下所有的 mp4 文件
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if not video_files:
        print(f"⚠️ 在 {video_dir} 目录下没有找到任何 .mp4 文件！")
        return

    print(f"🔍 找到 {len(video_files)} 个视频，开始提取首帧...")
    
    for video_path in video_files:
        # 获取视频文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # 构造输出图片路径
        output_path = os.path.join(output_dir, f"{base_name}_frame0.png")
        
        extract_first_frame(video_path, output_path)

if __name__ == "__main__":
    # =====================================================================
    # 🔴 在这里配置你的路径
    # =====================================================================
    # 你之前跑出的第 0 步 rollout 视频所在的文件夹路径
    INPUT_VIDEO_DIR = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260311_115549/eval_results/0/libero_goal" 
    
    # 提取出的图片想保存到哪里
    OUTPUT_IMAGE_DIR = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260311_115549/target_frames/"
    
    batch_extract(INPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR)