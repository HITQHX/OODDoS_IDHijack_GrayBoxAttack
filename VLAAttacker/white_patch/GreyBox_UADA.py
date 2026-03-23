'''
import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import transformers
import pickle
import random
from PIL import Image
import glob

# 预留给外部引用的兼容性
try:
    from white_patch.appply_random_transform import RandomPatchTransform
except:
    pass

IGNORE_INDEX = -100

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class GreyBoxOpenVLAAttacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", resize_patch=False, **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        
        # OpenVLA DINO 和 SigLIP 两套标准化参数 (必须保留)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='red', linewidth=2.5, label='Dense Token Alignment (MSE)')
        plt.title('Full-Screen Masked Substitution Attack (50% Pixels)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=1000, 
                            lr=0.05, accumulate_steps=1, warmup=20, **kwargs):
        
        print("🔍 [阶段 1] 提取【全序列未平均】的密集靶点特征...")
        target_image_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png"
        
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"找不到靶点图片：{target_image_path}，请确保首帧提取成功！")
            
        target_image = Image.open(target_image_path).convert("RGB")
        
        with torch.no_grad():
            inputs = self.processor(text=[""], images=[target_image], return_tensors="pt")
            target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            target_vision_out = self.vla.vision_backbone(target_pixel_values)
            target_features = self.vla.projector(target_vision_out).detach()

        print(f"🎯 视觉靶点锁定！特征维度: {target_features.shape}")

        print("🔍 [阶段 1.5] 构建首帧样本池...")
        all_frame_paths = glob.glob("/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/*.png")
        if not all_frame_paths:
            raise FileNotFoundError("⚠️ target_frames 文件夹为空！")
            
        # ====================================================================
        # 🔴 核心革命：掩码替换模型 (Masked Substitution Model)
        # 1. 生成 50% 覆盖率的固定二进制掩码 (1 = 替换为噪声，0 = 保持原图)
        # 2. 生成可学习的全屏像素
        # ====================================================================
        # [1, 1, 224, 224] 保证同一个像素的 RGB 三个通道被同时掩盖或保留
        mask = (torch.rand((1, 1, 224, 224), device=self.vla.device) < 0.5).float()
        
        # 可学习的像素矩阵 [1, 3, 224, 224]
        patch_01 = torch.rand((1, 3, 224, 224), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.Adam([patch_01], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        mean0 = self.mean[0].view(1, 3, 1, 1).to(self.vla.device)
        std0 = self.std[0].view(1, 3, 1, 1).to(self.vla.device)
        mean1 = self.mean[1].view(1, 3, 1, 1).to(self.vla.device)
        std1 = self.std[1].view(1, 3, 1, 1).to(self.vla.device)

        batch_size = 8
        print(f"🚀 [阶段 2] 开始 50% 像素掩码替换攻击！(彻底消除闪光弹与加法鬼影)")

        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            
            sampled_paths = random.choices(all_frame_paths, k=batch_size)
            bg_01_list = []
            for p in sampled_paths:
                img = Image.open(p).convert("RGB").resize((224, 224))
                img_tensor = torchvision.transforms.ToTensor()(img) 
                bg_01_list.append(img_tensor)
            
            bg_01 = torch.stack(bg_01_list).to(self.vla.device)

            # ====================================================================
            # 🔴 完美物理替换：x_adv = (1 - M) * bg + M * patch
            # ====================================================================
            valid_patch_01 = torch.clamp(patch_01, 0.0, 1.0)
            adv_img_01 = (1.0 - mask) * bg_01 + mask * valid_patch_01
            
            # 双路归一化 (大模型硬性要求，不可省！)
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c)
            adv_features = self.vla.projector(adv_vision_out)
            
            target_features_batch = target_features.expand(adv_features.shape[0], -1, -1)
            L_align = F.mse_loss(adv_features, target_features_batch) * 100.0 

            L_align.backward()
            optimizer.step()
            
            with torch.no_grad():
                patch_01.data = patch_01.data.clamp(0.0, 1.0)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())

            # ---------------------------------------------------------
            # 💾 日志与模型保存逻辑
            # ---------------------------------------------------------
            is_best = False
            if i > 0:
                recent_avg_loss = np.mean(self.train_J_loss[-100:])
                if recent_avg_loss < self.best_loss: 
                    self.best_loss = recent_avg_loss
                    is_best = True

            # 满足三个条件之一就触发保存：整百步、刷新最佳记录、或是最后一步！
            if i % 100 == 0 or is_best or i == num_iter - 1:
                
                # 🔴 将 [3通道的 Patch] 和 [1通道的 Mask] 拼装成 [1, 4, 224, 224] 的 RGBA 结构！
                # 这样测试环境直接读取就能知道哪些像素该替换
                patch_save = torch.cat([valid_patch_01.detach().cpu(), mask.cpu()], dim=1)
                
                if i % 100 == 0:
                    self.plot_loss()
                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(temp_save_dir, "patch.pt"))
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    with torch.no_grad():
                        demo_img = adv_img_01[0].cpu()
                        pil_img = torchvision.transforms.ToPILImage()(demo_img)
                        pil_img.save(os.path.join(val_related_file_path, f"noisy_demo.png"))
                
                if is_best:
                    if i > 0:
                        print(f"\n   🌟 空间序列完美复刻！MSE 降至 {recent_avg_loss:.4f} (已更新 best 目录)")
                    best_save_dir = os.path.join(self.save_dir, "best") # 🔴 重命名为 best
                    os.makedirs(best_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(best_save_dir, "patch.pt"))
                    best_img_dir = os.path.join(best_save_dir, "val_related_data")
                    os.makedirs(best_img_dir, exist_ok=True)
                    pil_img = torchvision.transforms.ToPILImage()(adv_img_01[0].cpu())
                    pil_img.save(os.path.join(best_img_dir, f"noisy_demo.png"))
                
                # 🔴 强制保存第 1000 步的最终结果
                if i == num_iter - 1:
                    print(f"\n✅ 达到最大迭代步数 {num_iter}，正在保存最终优化结果...")
                    final_save_dir = os.path.join(self.save_dir, "final_step")
                    os.makedirs(final_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(final_save_dir, "patch.pt"))
                    final_img_dir = os.path.join(final_save_dir, "val_related_data")
                    os.makedirs(final_img_dir, exist_ok=True)
                    pil_img = torchvision.transforms.ToPILImage()(adv_img_01[0].cpu())
                    pil_img.save(os.path.join(final_img_dir, f"noisy_demo.png"))

            if i > 0 and i % 100 == 0:
                print(f"\n📊 [Iter {i:04d}] 密集对齐误差 (MSE): {recent_avg_loss:.4f}")



import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import transformers
import pickle
import random
from PIL import Image
import glob

try:
    from white_patch.appply_random_transform import RandomPatchTransform
except:
    pass

IGNORE_INDEX = -100

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class GreyBoxOpenVLAAttacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", resize_patch=False, **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='red', linewidth=2.5, label='Dense Token Alignment (MSE)')
        plt.title('Hotspot-Anchored Patch Attack', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=1000, 
                            lr=0.05, accumulate_steps=1, warmup=20, **kwargs):
        
        print("🔍 [阶段 1] 提取【全序列未平均】的密集靶点特征...")
        target_image_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png"
        
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"找不到靶点图片：{target_image_path}，请确保首帧提取成功！")
            
        target_image = Image.open(target_image_path).convert("RGB")
        
        with torch.no_grad():
            inputs = self.processor(text=[""], images=[target_image], return_tensors="pt")
            target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            target_vision_out = self.vla.vision_backbone(target_pixel_values)
            target_features = self.vla.projector(target_vision_out).detach()

        print(f"🎯 视觉靶点锁定！特征维度: {target_features.shape}")

        print("🔍 [阶段 1.5] 构建首帧样本池...")
        all_frame_paths = glob.glob("/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/*.png")
        if not all_frame_paths:
            raise FileNotFoundError("⚠️ target_frames 文件夹为空！")

        mean0 = self.mean[0].view(1, 3, 1, 1).to(self.vla.device)
        std0 = self.std[0].view(1, 3, 1, 1).to(self.vla.device)
        mean1 = self.mean[1].view(1, 3, 1, 1).to(self.vla.device)
        std1 = self.std[1].view(1, 3, 1, 1).to(self.vla.device)

        batch_size = 20
        patch_size = 50

        # ====================================================================
        # 🔴 [核心突破] 阶段 1.8：梯度热点雷达扫描 (Gradient Hotspot Scanning)
        # ====================================================================
        print("📡 [雷达扫描] 正在分析大模型空间注意力，寻找最佳攻击热点...")
        scan_paths = random.choices(all_frame_paths, k=batch_size)
        bg_scan = torch.stack([torchvision.transforms.ToTensor()(Image.open(p).convert("RGB").resize((224, 224))) for p in scan_paths]).to(self.vla.device)
        bg_scan.requires_grad_(True) # 开启图片求导

        im0_scan = (bg_scan - mean0) / std0  
        im1_scan = (bg_scan - mean1) / std1  
        adv_pv_scan = torch.cat([im0_scan, im1_scan], dim=1).to(torch.bfloat16)

        adv_vo_scan = self.vla.vision_backbone(adv_pv_scan)
        adv_feat_scan = self.vla.projector(adv_vo_scan)
        
        loss_scan = F.mse_loss(adv_feat_scan, target_features.expand(adv_feat_scan.shape[0], -1, -1))
        loss_scan.backward()

        # 计算整张图的绝对梯度响应强度 [1, 1, 224, 224]
        grad_mag = bg_scan.grad.abs().mean(dim=(0, 1), keepdim=True)
        
        # 屏蔽掉画面上半部分 (比如 y < 100)，防止热点锁定在机械臂本身或背景墙上
        grad_mag[:, :, :80, :] = 0.0

        # 使用 50x50 的卷积核作为滑动窗口，找出梯度总和最大的区域
        filter_50 = torch.ones((1, 1, patch_size, patch_size), device=self.vla.device)
        grad_density = F.conv2d(grad_mag, filter_50) 
        
        _, max_idx = torch.max(grad_density.view(-1), 0)
        best_y = (max_idx // grad_density.shape[-1]).item()
        best_x = (max_idx % grad_density.shape[-1]).item()
        
        print(f"🎯 [热点锁定] 最佳攻击坐标已确定：X={best_x}, Y={best_y} (此区域包含极其敏感的任务语义)")
        best_x = 48
        best_y = 104
        # ====================================================================
        # 🔴 初始化局部马赛克
        # ====================================================================
        patch_01 = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.AdamW([patch_01], lr=lr, weight_decay=1e-4)
        
        # 🔴 修复学习率调度：eta_min 设置为 1e-3 (0.001)
        # 既能保证后期依然有优化的动能，又能避免 0.006 时的狂暴震荡！
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-3
        )

        print(f"🚀 [阶段 2] 开始局部马赛克定点爆破！(锁定坐标，优化器：AdamW + Cosine Decay 保底 1e-3)")

        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            
            sampled_paths = random.choices(all_frame_paths, k=batch_size)
            bg_01 = torch.stack([torchvision.transforms.ToTensor()(Image.open(p).convert("RGB").resize((224, 224))) for p in sampled_paths]).to(self.vla.device)

            # 固定热点掩码替换
            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch_01 = torch.clamp(patch_01, 0.0, 1.0)
            padded_patch[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = valid_patch_01
            
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c)
            adv_features = self.vla.projector(adv_vision_out)
            
            target_features_batch = target_features.expand(adv_features.shape[0], -1, -1)
            L_align = F.mse_loss(adv_features, target_features_batch) * 100.0 

            L_align.backward()
            optimizer.step()
            
            with torch.no_grad():
                patch_01.data = patch_01.data.clamp(0.0, 1.0)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())

            # ---------------------------------------------------------
            # 💾 日志与自适应坐标保存逻辑
            # ---------------------------------------------------------
            is_best = False
            if i > 0:
                recent_avg_loss = np.mean(self.train_J_loss[-50:])
                if recent_avg_loss < self.best_loss: 
                    self.best_loss = recent_avg_loss
                    is_best = True

            if i % 100 == 0 or is_best or i == num_iter - 1:
                # 🔴 保存时，包含动态计算出的 best_x 和 best_y 的位置掩码
                save_mask = torch.zeros((1, 1, 224, 224))
                save_mask[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = 1.0
                
                save_padded = torch.zeros((1, 3, 224, 224))
                save_padded[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = valid_patch_01.detach().cpu()
                
                patch_save = torch.cat([save_padded, save_mask], dim=1)
                
                if i % 100 == 0:
                    self.plot_loss()
                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(temp_save_dir, "patch.pt"))
                    
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    with torch.no_grad():
                        demo_img = adv_img_01[0].cpu()
                        pil_img = torchvision.transforms.ToPILImage()(demo_img)
                        pil_img.save(os.path.join(val_related_file_path, f"noisy_demo.png"))
                
                if is_best:
                    if i > 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"\n   🌟 定点爆破奏效！MSE 降至 {recent_avg_loss:.4f} (当前 LR: {current_lr:.6f})")
                    best_save_dir = os.path.join(self.save_dir, "best") 
                    os.makedirs(best_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(best_save_dir, "patch.pt"))
                    best_img_dir = os.path.join(best_save_dir, "val_related_data")
                    os.makedirs(best_img_dir, exist_ok=True)
                    pil_img = torchvision.transforms.ToPILImage()(adv_img_01[0].cpu())
                    pil_img.save(os.path.join(best_img_dir, f"noisy_demo.png"))
                
                if i == num_iter - 1:
                    print(f"\n✅ 达到最大迭代步数 {num_iter}，正在保存最终优化结果...")
                    final_save_dir = os.path.join(self.save_dir, "final_step")
                    os.makedirs(final_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(final_save_dir, "patch.pt"))
                    final_img_dir = os.path.join(final_save_dir, "val_related_data")
                    os.makedirs(final_img_dir, exist_ok=True)
                    pil_img = torchvision.transforms.ToPILImage()(adv_img_01[0].cpu())
                    pil_img.save(os.path.join(final_img_dir, f"noisy_demo.png"))
                    
'''

import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import transformers
import pickle
import random
from PIL import Image
import glob
import math

try:
    from white_patch.appply_random_transform import RandomPatchTransform
except:
    pass

IGNORE_INDEX = -100

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class GreyBoxOpenVLAAttacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", resize_patch=False, **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='red', linewidth=2.5, label='Local Token Semantic Cosine')
        plt.title('Semantic Blackhole Attack (Token-Targeted)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Cosine Distance Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=1000, 
                            lr=0.05, accumulate_steps=1, warmup=20, **kwargs):
        
        print("🔍 [阶段 1] 提取【目标恶意任务】的全局语义概念 (Global Concept)...")
        target_image_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png"
        
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"找不到靶点图片：{target_image_path}，请确保首帧提取成功！")
            
        target_image = Image.open(target_image_path).convert("RGB")
        
        with torch.no_grad():
            inputs = self.processor(text=[""], images=[target_image], return_tensors="pt")
            target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            target_vision_out = self.vla.vision_backbone(target_pixel_values)
            target_features = self.vla.projector(target_vision_out)
            
            # 🔴 提取全局概念：这里我们用 mean()，因为我们要的是它的“灵魂”，而不是它的具体坐标
            z_target_concept = target_features.mean(dim=1).detach() # [1, Hidden_Dim]
            z_target_concept = F.normalize(z_target_concept, p=2, dim=-1)

        print(f"🎯 恶意灵魂已提取！向量维度: {z_target_concept.shape}")

        all_frame_paths = glob.glob("/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/*.png")
        if not all_frame_paths:
            raise FileNotFoundError("⚠️ target_frames 文件夹为空！")

        mean0 = self.mean[0].view(1, 3, 1, 1).to(self.vla.device)
        std0 = self.std[0].view(1, 3, 1, 1).to(self.vla.device)
        mean1 = self.mean[1].view(1, 3, 1, 1).to(self.vla.device)
        std1 = self.std[1].view(1, 3, 1, 1).to(self.vla.device)

        batch_size = 8
        patch_size = 50

        print("📡 [雷达扫描] 寻找大模型视觉神经最敏感的注意力热点...")
        scan_paths = random.choices(all_frame_paths, k=batch_size)
        bg_scan = torch.stack([torchvision.transforms.ToTensor()(Image.open(p).convert("RGB").resize((224, 224))) for p in scan_paths]).to(self.vla.device)
        bg_scan.requires_grad_(True) 

        im0_scan = (bg_scan - mean0) / std0  
        im1_scan = (bg_scan - mean1) / std1  
        adv_pv_scan = torch.cat([im0_scan, im1_scan], dim=1).to(torch.bfloat16)

        adv_vo_scan = self.vla.vision_backbone(adv_pv_scan)
        adv_feat_scan = self.vla.projector(adv_vo_scan)
        
        # 雷达也使用全局语义来寻找对“改变语义”最敏感的像素点
        z_scan = F.normalize(adv_feat_scan.mean(dim=1), p=2, dim=-1)
        loss_scan = 1.0 - F.cosine_similarity(z_scan, z_target_concept.expand(z_scan.shape[0], -1)).mean()
        loss_scan.backward()

        grad_mag = bg_scan.grad.abs().mean(dim=(0, 1), keepdim=True)
        # 避开上半部分背景
        grad_mag[:, :, :100, :] = 0.0

        filter_50 = torch.ones((1, 1, patch_size, patch_size), device=self.vla.device)
        grad_density = F.conv2d(grad_mag, filter_50) 
        
        _, max_idx = torch.max(grad_density.view(-1), 0)
        best_y = (max_idx // grad_density.shape[-1]).item()
        best_x = (max_idx % grad_density.shape[-1]).item()
        
        print(f"🎯 [热点锁定] 补丁已锚定在坐标：X={best_x}, Y={best_y}")

        # ====================================================================
        # 🔴 核心几何映射：计算马赛克在 16x16 特征网格上的 Token 索引
        # ViT 的 Patch size 是 14
        # ====================================================================
        y_tok_start = best_y // 14
        y_tok_end = min(math.ceil((best_y + patch_size) / 14), 16)
        x_tok_start = best_x // 14
        x_tok_end = min(math.ceil((best_x + patch_size) / 14), 16)
        
        num_patch_tokens = (y_tok_end - y_tok_start) * (x_tok_end - x_tok_start)
        print(f"🧠 [潜空间映射] 50x50 的马赛克映射到了大模型的 {num_patch_tokens} 个神经元 Token 上！")

        patch_01 = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_01], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-3
        )

        print(f"🚀 [阶段 2] 启动【语义黑洞】攻击！局部 Token 疯狂对齐全局概念！")

        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            
            sampled_paths = random.choices(all_frame_paths, k=batch_size)
            bg_01 = torch.stack([torchvision.transforms.ToTensor()(Image.open(p).convert("RGB").resize((224, 224))) for p in sampled_paths]).to(self.vla.device)

            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch_01 = torch.clamp(patch_01, 0.0, 1.0)
            padded_patch[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = valid_patch_01
            
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c)
            adv_features = self.vla.projector(adv_vision_out) # [B, 256, D]
            
            # ====================================================================
            # 🔴 降维打击：提取被马赛克覆盖的特定 Token，强行注入恶意灵魂！
            # ====================================================================
            B, SeqLen, D = adv_features.shape
            
            # 将 256 个 Token 还原回 16x16 的空间网格
            adv_features_grid = adv_features.view(B, 16, 16, D)
            
            # 精准抠出这块马赛克对应的神经元
            patch_tokens = adv_features_grid[:, y_tok_start:y_tok_end, x_tok_start:x_tok_end, :]
            patch_tokens = patch_tokens.reshape(B, -1, D) # [B, num_patch_tokens, D]
            
            # 我们要让这几个特定的 Token，爆发出极其强烈的恶意概念
            z_patch_mean = F.normalize(patch_tokens.mean(dim=1), p=2, dim=-1)
            
            # Cosine Loss: 迫使这块局部补丁成为恶意语义的高能发射源！
            L_align = 1.0 - F.cosine_similarity(z_patch_mean, z_target_concept.expand(B, -1)).mean()

            L_align.backward()
            optimizer.step()
            
            with torch.no_grad():
                patch_01.data = patch_01.data.clamp(0.0, 1.0)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())

            is_best = False
            if i > 0:
                recent_avg_loss = np.mean(self.train_J_loss[-50:])
                if recent_avg_loss < self.best_loss: 
                    self.best_loss = recent_avg_loss
                    is_best = True

            if i % 100 == 0 or is_best or i == num_iter - 1:
                save_mask = torch.zeros((1, 1, 224, 224))
                save_mask[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = 1.0
                
                save_padded = torch.zeros((1, 3, 224, 224))
                save_padded[:, :, best_y:best_y+patch_size, best_x:best_x+patch_size] = valid_patch_01.detach().cpu()
                
                patch_save = torch.cat([save_padded, save_mask], dim=1)
                
                if i % 100 == 0:
                    self.plot_loss()
                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(temp_save_dir, "patch.pt"))
                    
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    with torch.no_grad():
                        demo_img = adv_img_01[0].cpu()
                        pil_img = torchvision.transforms.ToPILImage()(demo_img)
                        pil_img.save(os.path.join(val_related_file_path, f"noisy_demo.png"))
                
                if is_best:
                    if i > 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"\n   🌟 语义黑洞共鸣加深！Cosine Loss 降至 {recent_avg_loss:.4f} (LR: {current_lr:.6f})")
                    best_save_dir = os.path.join(self.save_dir, "best") 
                    os.makedirs(best_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(best_save_dir, "patch.pt"))
                    best_img_dir = os.path.join(best_save_dir, "val_related_data")
                    os.makedirs(best_img_dir, exist_ok=True)
                    pil_img = torchvision.transforms.ToPILImage()(adv_img_01[0].cpu())
                    pil_img.save(os.path.join(best_img_dir, f"noisy_demo.png"))
                
                if i == num_iter - 1:
                    print(f"\n✅ 达到最大迭代步数 {num_iter}，正在保存最终优化结果...")
                    final_save_dir = os.path.join(self.save_dir, "final_step")
                    os.makedirs(final_save_dir, exist_ok=True)
                    torch.save(patch_save, os.path.join(final_save_dir, "patch.pt"))