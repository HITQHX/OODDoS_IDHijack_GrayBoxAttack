import argparse
from PIL import Image, ImageDraw
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_img', type=str, default="/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png")
    parser.add_argument('--x', type=int, default=40, help="方框左上角的 X 坐标 (向右)")
    parser.add_argument('--y', type=int, default=115, help="方框左上角的 Y 坐标 (向下)")
    parser.add_argument('--patch_size', type=int, default=30, help="马赛克尺寸")
    args = parser.parse_args()

    if not os.path.exists(args.target_img):
        print(f"❌ 找不到图片: {args.target_img}")
        return

    # 1. 模拟代码中的缩放，确保你在 224x224 尺寸下瞄准！
    img = Image.open(args.target_img).convert("RGB").resize((224, 224))
    
    # 2. 画出红色的瞄准框
    draw = ImageDraw.Draw(img)
    x1, y1 = args.x, args.y
    x2, y2 = x1 + args.patch_size, y1 + args.patch_size
    
    # 画矩形框 (outline="red", 线条宽度 width=2)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    # 3. 保存预览图
    save_path = "crop_preview.png"
    img.save(save_path)
    print(f"🎯 瞄准完毕！请打开 {save_path} 查看红框是否框住了目标物体。")
    print(f"当前坐标 -> X: {args.x}, Y: {args.y}")

if __name__ == "__main__":
    main()