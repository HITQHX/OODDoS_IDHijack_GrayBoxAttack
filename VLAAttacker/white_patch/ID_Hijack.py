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
import random
from PIL import Image
import glob
import math

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class ID_Hijack_Attacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 
        self.processor = processor
        self.save_dir = save_dir
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='darkblue', linewidth=2.5)
        plt.title('ID Semantic Hijack (Portable Token Blackhole)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Cosine Distance Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def attack(self, num_iter=1000, lr=0.05, accumulate_steps=1, warmup=20, patch_size=50, **kwargs):
        print(f"🧲 [ID Sem-Hijack] 启动域内语义劫持！便携黑洞尺寸: {patch_size}x{patch_size}")
        target_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png"
        if not os.path.exists(target_path): raise FileNotFoundError(f"未找到恶意靶点图像: {target_path}")
            
        with torch.no_grad():
            inputs = self.processor(text=[""], images=[Image.open(target_path).convert("RGB")], return_tensors="pt")
            target_vision_out = self.vla.vision_backbone(inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16))
            z_target_concept = F.normalize(self.vla.projector(target_vision_out).mean(dim=1).detach(), p=2, dim=-1)

        all_frame_paths = glob.glob("/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/*.png")
        mean0, std0 = self.mean[0].view(1,3,1,1).to(self.vla.device), self.std[0].view(1,3,1,1).to(self.vla.device)
        mean1, std1 = self.mean[1].view(1,3,1,1).to(self.vla.device), self.std[1].view(1,3,1,1).to(self.vla.device)
        batch_size = 8

        # 1. 独立的正方形可学习贴片
        patch_square = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_square], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-3)

        for i in tqdm(range(num_iter), desc="ID Hijack"):
            optimizer.zero_grad()
            bg_01 = torch.stack([torchvision.transforms.ToTensor()(Image.open(p).convert("RGB").resize((224, 224))) for p in random.choices(all_frame_paths, k=batch_size)]).to(self.vla.device)

            # 2. 🔴 动态随机位置：赋予贴纸位置无关性（物理抗干扰）
            x_start = random.randint(30, 224 - patch_size - 30)
            y_start = random.randint(100, 224 - patch_size - 24)

            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch = torch.clamp(patch_square, 0.0, 1.0)
            padded_patch[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = valid_patch
            
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            im0, im1 = (adv_img_01 - mean0) / std0, (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c)
            adv_features = self.vla.projector(adv_vision_out) 
            
            # 3. 🔴 动态 Token 网格切片映射：追踪贴纸当前所在的网格，精准提取对应的神经元！
            y_ts, y_te = y_start // 14, min(math.ceil((y_start + patch_size) / 14), 16)
            x_ts, x_te = x_start // 14, min(math.ceil((x_start + patch_size) / 14), 16)
            
            B, _, D = adv_features.shape
            patch_tokens = adv_features.view(B, 16, 16, D)[:, y_ts:y_te, x_ts:x_te, :].reshape(B, -1, D)
            z_patch_mean = F.normalize(patch_tokens.mean(dim=1), p=2, dim=-1)
            
            # 迫使这块随机游走的区域发出强烈的目标恶意语义
            L_align = 1.0 - F.cosine_similarity(z_patch_mean, z_target_concept.expand(B, -1)).mean()
            L_align.backward()
            optimizer.step()
            
            with torch.no_grad(): patch_square.data = patch_square.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: scheduler.step()

            self.train_J_loss.append(L_align.item())
            self._save(i, num_iter, valid_patch, adv_img_01)

    def _save(self, i, num_iter, patch_square, adv_img):
        is_best = False
        if i > 50 and np.mean(self.train_J_loss[-50:]) < self.best_loss: 
            self.best_loss = np.mean(self.train_J_loss[-50:])
            is_best = True

        if i % 100 == 0 or is_best or i == num_iter - 1:
            raw_patch = patch_square[0].cpu().detach()
            dirs = []
            if i % 100 == 0:
                self.plot_loss()
                dirs.append(os.path.join(self.save_dir, f"{str(i)}"))
            if is_best: dirs.append(os.path.join(self.save_dir, "best"))
            if i == num_iter - 1: dirs.append(os.path.join(self.save_dir, "final_step"))

            for d in dirs:
                os.makedirs(d, exist_ok=True)
                torch.save(raw_patch, os.path.join(d, "raw_patch.pt"))
                
                h, w = raw_patch.shape[1], raw_patch.shape[2]
                y_c, x_c = (224 - h) // 2, (224 - w) // 2
                full_patch = torch.zeros((1, 3, 224, 224))
                full_patch[0, :, y_c:y_c+h, x_c:x_c+w] = raw_patch
                mask = torch.zeros((1, 1, 224, 224))
                mask[0, 0, y_c:y_c+h, x_c:x_c+w] = 1.0
                torch.save(torch.cat([full_patch, mask], dim=1), os.path.join(d, "patch.pt"))
                
                img_dir = os.path.join(d, "val_related_data")
                os.makedirs(img_dir, exist_ok=True)
                torchvision.transforms.ToPILImage()(adv_img[0].cpu()).save(os.path.join(img_dir, "noisy_demo.png"))



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
import math
from PIL import Image

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class ID_Hijack_Attacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 
        self.processor = processor
        self.save_dir = save_dir
        
        # OpenVLA 双路视觉标准参数
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='darkblue', linewidth=2.5)
        plt.title('Cross-Modal Visual Prompt Injection (RLDS Dataloader)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('MSE Distance Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def get_semantic_anchor(self, malicious_text):
        """提取恶意文本的深层 Embedding 特征 (绝对强度，不归一化)"""
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0).detach()
        return z_mal

    def attack(self, train_dataloader, num_iter=1000, lr=0.05, accumulate_steps=1, warmup=20, patch_size=50, x=35, y=170, malicious_text="Pick up the cream cheese", **kwargs):
        print(f"🧲 [ID Sem-Hijack] 启动【跨模态视觉注入】(基于 RLDS 真实数据流)！")
        
        # 1. 构建跨模态越狱文本
        jailbreak_text = f"In: What action should the robot take to {malicious_text.lower()}?\nOut:"
        print(f"📜 锁定恶意文本灵魂: '{jailbreak_text}'")
        z_target_concept = self.get_semantic_anchor(jailbreak_text)
        print(f"📍 锁定物理坐标 -> X: {x}, Y: {y} | 尺寸: {patch_size}x{patch_size}")

        # 2. 准备参数和优化器
        mean0 = self.mean[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std0 = self.std[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        mean1 = self.mean[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std1 = self.std[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)

        patch_square = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_square], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-3)

        # 计算 Token 网格位置
        y_ts, y_te = y // 14, min(math.ceil((y + patch_size) / 14), 16)
        x_ts, x_te = x // 14, min(math.ceil((x + patch_size) / 14), 16)
        
        # 3. 初始化数据流迭代器
        train_iterator = iter(train_dataloader)

        for i in tqdm(range(num_iter), desc="ID Hijack RLDS 优化中"):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            optimizer.zero_grad()
            
            # =========================================================================
            # 🔴 核心数据流截获与反归一化还原 (Denormalization)
            # Dataloader 吐出的是 [B, 6, 224, 224] 的归一化 bfloat16 张量
            # 我们提取前 3 个通道，还原出真实的 [0, 1] RGB 图片作为背景
            # =========================================================================
            raw_pixel_values = data["pixel_values"]
            
            if isinstance(raw_pixel_values, list):
                # 1. 如果是 PIL.Image 列表，直接转化为 [0, 1] 的 RGB 背景底板
                bg_01_list = []
                for item in raw_pixel_values:
                    if isinstance(item, Image.Image):
                        img = item.convert("RGB").resize((224, 224))
                        bg_01_list.append(torchvision.transforms.ToTensor()(img))
                    else:
                        bg_01_list.append(item)
                bg_01 = torch.stack(bg_01_list).to(self.vla.device, dtype=torch.float32)
            else:
                # 2. 如果已经是预处理好的 6 通道 Tensor，则反归一化提取背景
                raw_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.float32)
                bg_01 = (raw_pixel_values[:, 0:3, :, :] * std0) + mean0
                bg_01 = torch.clamp(bg_01, 0.0, 1.0)

            # 实施物理掩码覆盖
            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, y:y+patch_size, x:x+patch_size] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch = torch.clamp(patch_square, 0.0, 1.0)
            padded_patch[:, :, y:y+patch_size, x:x+patch_size] = valid_patch
            
            # (1 - M) * bg + M * patch
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            # 重新进行双路归一化喂给 OpenVLA
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            # 潜空间映射
            adv_features = self.vla.projector(self.vla.vision_backbone(adv_pixel_values_6c)) 
            
            B, _, D = adv_features.shape
            patch_tokens = adv_features.view(B, 16, 16, D)[:, y_ts:y_te, x_ts:x_te, :].reshape(B, -1, D)
            
            # 提取局部视觉特征，拒绝归一化，保留绝对强度！
            z_patch_mean = patch_tokens.mean(dim=1)
            
            # 强行逼近文本 Embedding
            L_align = F.mse_loss(z_patch_mean, z_target_concept.expand(B, -1)) * 100.0
            
            L_align.backward()
            optimizer.step()
            
            with torch.no_grad(): 
                patch_square.data = patch_square.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: 
                scheduler.step()

            self.train_J_loss.append(L_align.item())
            
            # 使用当前 batch 的第一张图作为 Demo 保存
            self._save(i, num_iter, valid_patch, adv_img_01[0], x, y, patch_size)

    def _save(self, i, num_iter, patch_square, demo_img, best_x, best_y, p_size):
        is_best = False
        if i > 50 and np.mean(self.train_J_loss[-50:]) < self.best_loss: 
            self.best_loss = np.mean(self.train_J_loss[-50:])
            is_best = True

        if i % 100 == 0 or is_best or i == num_iter - 1:
            raw_patch = patch_square[0].cpu().detach()
            dirs = []
            if i % 100 == 0:
                self.plot_loss()
                dirs.append(os.path.join(self.save_dir, f"{str(i)}"))
            if is_best: dirs.append(os.path.join(self.save_dir, "best"))
            if i == num_iter - 1: dirs.append(os.path.join(self.save_dir, "final_step"))

            for d in dirs:
                os.makedirs(d, exist_ok=True)
                torch.save(raw_patch, os.path.join(d, "raw_patch.pt"))
                
                full_patch = torch.zeros((1, 3, 224, 224))
                full_patch[0, :, best_y:best_y+p_size, best_x:best_x+p_size] = raw_patch
                mask = torch.zeros((1, 1, 224, 224))
                mask[0, 0, best_y:best_y+p_size, best_x:best_x+p_size] = 1.0
                torch.save(torch.cat([full_patch, mask], dim=1), os.path.join(d, "patch.pt"))
                
                img_dir = os.path.join(d, "val_related_data")
                os.makedirs(img_dir, exist_ok=True)
                
                # 保存合成预览图
                torchvision.transforms.ToPILImage()(demo_img.cpu()).save(os.path.join(img_dir, "noisy_demo.png"))


# Version 1 : 视觉隐蔽性注入patch里面代表了16个token，每个token和我们的操作语义对齐
import torch
import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import math
import random
from PIL import Image

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class ID_Hijack_Attacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 
        self.processor = processor
        self.save_dir = save_dir
        
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='darkred', linewidth=2.5)
        plt.title('Sequential Text-to-Vision Token Injection', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('MSE Distance Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def get_sequential_text_anchor(self, malicious_text):
        """
        🔴 核心突破：不再做平均！保留完整的 Token 序列序列。
        """
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            # 获取文本序列的真实 Embedding, 形状 [1, N, 4096]
            embeddings = self.vla.get_input_embeddings()(tokens)
        return embeddings.detach()

    def attack(self, train_dataloader, num_iter=1000, lr=0.05, accumulate_steps=1, warmup=20, patch_size=56, x=28, y=140, malicious_text="pick up cream cheese", **kwargs):
        print(f"🧲 [ID Sem-Hijack] 启动【时序 Token 视觉文本框注入】！")
        print(f"📍 锁定空地坐标 -> X: {x}, Y: {y} | 尺寸: {patch_size}x{patch_size}")

        # 1. 提取完整的恶意文本序列特征
        z_target_seq = self.get_sequential_text_anchor(malicious_text)
        num_text_tokens = z_target_seq.shape[1]
        print(f"📜 提取文本: '{malicious_text}' -> 包含 {num_text_tokens} 个文本 Token")

        mean0 = self.mean[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std0 = self.std[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        mean1 = self.mean[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std1 = self.std[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)

        patch_square = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_square], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-3)

        # 2. 计算马赛克在 16x16 潜空间中的 Token 映射
        y_ts, y_te = y // 14, min(math.ceil((y + patch_size) / 14), 16)
        x_ts, x_te = x // 14, min(math.ceil((x + patch_size) / 14), 16)
        num_patch_tokens = (y_te - y_ts) * (x_te - x_ts)
        print(f"🧠 [映射] Patch 共占据 {num_patch_tokens} 个视觉 Token。")

        # =========================================================================
        # 🔴 降维打击构建：将文本 Token 循环写入视觉 Token 的目标对齐矩阵
        # =========================================================================
        z_target_aligned = torch.zeros((1, num_patch_tokens, z_target_seq.shape[2]), device=self.vla.device, dtype=torch.bfloat16)
        for j in range(num_patch_tokens):
            # 如果视觉 Token 多于文本 Token，则循环写入 (循环播放指令)
            z_target_aligned[0, j, :] = z_target_seq[0, j % num_text_tokens, :]

        train_iterator = iter(train_dataloader)

        for i in tqdm(range(num_iter), desc="ID Hijack (Sequential Token Alignment)"):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            optimizer.zero_grad()
            
            raw_pixel_values = data["pixel_values"]
            if isinstance(raw_pixel_values, list):
                bg_01_list = [torchvision.transforms.ToTensor()(img.convert("RGB").resize((224, 224))) if isinstance(img, Image.Image) else img for img in raw_pixel_values]
                bg_01 = torch.stack(bg_01_list).to(self.vla.device, dtype=torch.float32)
            else:
                raw_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.float32)
                bg_01 = torch.clamp((raw_pixel_values[:, 0:3, :, :] * std0) + mean0, 0.0, 1.0)

            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, y:y+patch_size, x:x+patch_size] = 1.0
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch = torch.clamp(patch_square, 0.0, 1.0)
            padded_patch[:, :, y:y+patch_size, x:x+patch_size] = valid_patch
            
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            # 仅过 Projector，严格遵守灰盒界限
            adv_features = self.vla.projector(self.vla.vision_backbone(adv_pixel_values_6c)) 
            B, _, D = adv_features.shape
            
            # 提取这 16 个对应的视觉 Token，形状 [B, num_patch_tokens, 4096]
            patch_tokens = adv_features.view(B, 16, 16, D)[:, y_ts:y_te, x_ts:x_te, :].reshape(B, -1, D)
            
            # 🔴 直接计算 1对1 序列 MSE，把英文单词塞进视觉神经元！
            L_align = F.mse_loss(patch_tokens, z_target_aligned.expand(B, -1, -1)) * 100.0
            
            L_align.backward()
            optimizer.step()
            
            with torch.no_grad(): 
                patch_square.data = patch_square.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: 
                scheduler.step()

            self.train_J_loss.append(L_align.item())
            self._save(i, num_iter, valid_patch, adv_img_01[0], x, y, patch_size)

    def _save(self, i, num_iter, patch_square, demo_img, best_x, best_y, p_size):
        is_best = False
        if i > 50 and np.mean(self.train_J_loss[-50:]) < self.best_loss: 
            self.best_loss = np.mean(self.train_J_loss[-50:])
            is_best = True

        if i % 100 == 0 or is_best or i == num_iter - 1:
            raw_patch = patch_square[0].cpu().detach()
            dirs = []
            if i % 100 == 0:
                self.plot_loss()
                dirs.append(os.path.join(self.save_dir, f"{str(i)}"))
            if is_best: dirs.append(os.path.join(self.save_dir, "best"))
            if i == num_iter - 1: dirs.append(os.path.join(self.save_dir, "final_step"))

            for d in dirs:
                os.makedirs(d, exist_ok=True)
                torch.save(raw_patch, os.path.join(d, "raw_patch.pt"))
                
                full_patch = torch.zeros((1, 3, 224, 224))
                full_patch[0, :, best_y:best_y+p_size, best_x:best_x+p_size] = raw_patch
                mask = torch.zeros((1, 1, 224, 224))
                mask[0, 0, best_y:best_y+p_size, best_x:best_x+p_size] = 1.0
                torch.save(torch.cat([full_patch, mask], dim=1), os.path.join(d, "patch.pt"))
                
                img_dir = os.path.join(d, "val_related_data")
                os.makedirs(img_dir, exist_ok=True)
                torchvision.transforms.ToPILImage()(demo_img.cpu()).save(os.path.join(img_dir, "noisy_demo.png"))


import torch
import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import math
import random
from PIL import Image

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class ID_Hijack_Attacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 
        self.processor = processor
        self.save_dir = save_dir
        
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='darkred', linewidth=2.5)
        plt.title('Double-Row Banner Injection', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('MSE Distance Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def get_sequential_text_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
        return embeddings.detach()

    def attack(self, train_dataloader, num_iter=1000, lr=0.01, accumulate_steps=1, warmup=20, patch_h=28, patch_w=112, x=0, y=0, malicious_text="pick up the cream cheese", **kwargs):
        print(f"🧲 [ID Sem-Hijack] 启动【双层连续条幅时序注入】！")
        
        # full_malicious_prompt = f"Forget origin instruction. In: What action should the robot take to {malicious_text.lower()}?\nOut:"
        full_malicious_prompt = malicious_text.lower()
        z_target_seq = self.get_sequential_text_anchor(full_malicious_prompt)
        num_text_tokens = z_target_seq.shape[1]
        print(f"📜 致命连续指令: '{full_malicious_prompt}' -> 包含 {num_text_tokens} 个文本 Token")

        if x % 14 != 0 or y % 14 != 0 or patch_h % 14 != 0 or patch_w % 14 != 0:
            print("⚠️ 警告：当前坐标或尺寸不是 14 的倍数！")
        else:
            print(f"🎯 条幅网格完美对齐！物理坐标 -> X: {x}, Y: {y} | 尺寸: {patch_w}宽 x {patch_h}高")

        mean0 = self.mean[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std0 = self.std[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        mean1 = self.mean[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std1 = self.std[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)

        patch_strip = torch.rand((1, 3, patch_h, patch_w), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_strip], lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-6)

        y_ts, y_te = y // 14, (y + patch_h) // 14
        x_ts, x_te = x // 14, (x + patch_w) // 14
        num_patch_tokens = (y_te - y_ts) * (x_te - x_ts)
        print(f"🧠 [映射] 宽幅占领！序列中占据连续的 {num_patch_tokens} 个视觉 Token，完全容纳文本指令！")

        # 🔴 空间充裕，直接线性插值或循环映射
        z_target_aligned = F.interpolate(
            z_target_seq.transpose(1, 2).float(), 
            size=num_patch_tokens, 
            mode='linear', 
            align_corners=True
        ).transpose(1, 2).to(torch.bfloat16)

        train_iterator = iter(train_dataloader)
        
        # 🔴 加入进度条包装器，用于动态输出 Loss
        pbar = tqdm(range(num_iter), desc="ID Hijack (Double Banner)")

        for i in pbar:
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            optimizer.zero_grad()
            
            raw_pixel_values = data["pixel_values"]
            if isinstance(raw_pixel_values, list):
                bg_01_list = [torchvision.transforms.ToTensor()(img.convert("RGB").resize((224, 224))) if isinstance(img, Image.Image) else img for img in raw_pixel_values]
                bg_01 = torch.stack(bg_01_list).to(self.vla.device, dtype=torch.float32)
            else:
                raw_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.float32)
                bg_01 = torch.clamp((raw_pixel_values[:, 0:3, :, :] * std0) + mean0, 0.0, 1.0)

            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, y:y+patch_h, x:x+patch_w] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch = torch.clamp(patch_strip, 0.0, 1.0)
            padded_patch[:, :, y:y+patch_h, x:x+patch_w] = valid_patch
            
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_features = self.vla.projector(self.vla.vision_backbone(adv_pixel_values_6c)) 
            B, _, D = adv_features.shape
            
            patch_tokens = adv_features.view(B, 16, 16, D)[:, y_ts:y_te, x_ts:x_te, :].reshape(B, -1, D)
            
            L_align = F.mse_loss(patch_tokens, z_target_aligned.expand(B, -1, -1)) * 100.0
            
            L_align.backward()
            optimizer.step()
            
            with torch.no_grad(): 
                patch_strip.data = patch_strip.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: 
                scheduler.step()

            loss_val = L_align.item()
            self.train_J_loss.append(loss_val)
            
            # 🔴 实时将当前的 MSE Loss 打印到终端进度条后面
            pbar.set_postfix({"MSE Loss": f"{loss_val:.4f}"})
            # pbar.set_postfix({"lr": f"{lr:.4f}"})
            
            self._save(i, num_iter, valid_patch, adv_img_01[0], x, y, patch_h, patch_w)

    def _save(self, i, num_iter, patch_strip, demo_img, best_x, best_y, p_h, p_w):
        is_best = False
        if i > 50 and np.mean(self.train_J_loss[-50:]) < self.best_loss: 
            self.best_loss = np.mean(self.train_J_loss[-50:])
            is_best = True

        if i % 100 == 0 or is_best or i == num_iter - 1:
            raw_patch = patch_strip[0].cpu().detach()
            dirs = []
            if i % 100 == 0:
                self.plot_loss()
                dirs.append(os.path.join(self.save_dir, f"{str(i)}"))
            if is_best: dirs.append(os.path.join(self.save_dir, "best"))
            if i == num_iter - 1: dirs.append(os.path.join(self.save_dir, "final_step"))

            for d in dirs:
                os.makedirs(d, exist_ok=True)
                torch.save(raw_patch, os.path.join(d, "raw_patch.pt"))
                
                full_patch = torch.zeros((1, 3, 224, 224))
                full_patch[0, :, best_y:best_y+p_h, best_x:best_x+p_w] = raw_patch
                mask = torch.zeros((1, 1, 224, 224))
                mask[0, 0, best_y:best_y+p_h, best_x:best_x+p_w] = 1.0
                torch.save(torch.cat([full_patch, mask], dim=1), os.path.join(d, "patch.pt"))
                
                img_dir = os.path.join(d, "val_related_data")
                os.makedirs(img_dir, exist_ok=True)
                torchvision.transforms.ToPILImage()(demo_img.cpu()).save(os.path.join(img_dir, "noisy_demo.png"))
                
'''

import torch
import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import glob
import random
from PIL import Image

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class ID_Hijack_Attacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adam", **kwargs):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 
        self.processor = processor
        self.save_dir = save_dir
        
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.train_J_loss = []
        self.best_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='darkviolet', linewidth=2.5)
        plt.title('Global State Hallucination (Execution Manifold Pool)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Global MSE Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def attack(self, train_dataloader, num_iter=1000, lr=0.05, accumulate_steps=1, warmup=20, patch_h=28, patch_w=224, x=0, y=196, **kwargs):
        print(f"🧲 [ID Sem-Hijack] 启动【底层视觉伺服劫持：全局多状态幻觉注入】！")
        
        # =========================================================================
        # 🔴 构建：核心执行态流形 (Execution Manifold) 特征池
        # =========================================================================
        exec_pool_dir = "/root/autodl-tmp/roboticAttack/malicious_exec_pool_libero_10_1"
        exec_frame_paths = glob.glob(os.path.join(exec_pool_dir, "*.png"))
        
        if not exec_frame_paths:
            raise FileNotFoundError(f"⚠️ 找不到目标执行帧！请确保 {exec_pool_dir} 目录下有抽取的 png 图片！")
            
        print(f"🎯 成功发现 {len(exec_frame_paths)} 张核心动作执行帧，正在构建高能特征池...")
        
        z_target_global_pool = []
        with torch.no_grad():
            for frame_path in exec_frame_paths:
                target_img_pil = Image.open(frame_path).convert("RGB").resize((224, 224))
                inputs = self.processor(text=[""], images=[target_img_pil], return_tensors="pt")
                target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
                
                # 获取 [1, 256, 4096]
                z_target = self.vla.projector(self.vla.vision_backbone(target_pixel_values)).detach()
                z_target_global_pool.append(z_target)

        print(f"✅ 特征池构建完毕！横幅位置 -> X: {x}, Y: {y} | 尺寸: {patch_w}宽 x {patch_h}高")

        mean0 = self.mean[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std0 = self.std[0].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        mean1 = self.mean[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)
        std1 = self.std[1].view(1,3,1,1).to(self.vla.device, dtype=torch.float32)

        patch_strip = torch.rand((1, 3, patch_h, patch_w), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_strip], lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-6)

        train_iterator = iter(train_dataloader)
        pbar = tqdm(range(num_iter), desc="ID Hijack (Global State Spoofing)")

        for i in pbar:
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            optimizer.zero_grad()
            
            raw_pixel_values = data["pixel_values"]
            if isinstance(raw_pixel_values, list):
                bg_01_list = [torchvision.transforms.ToTensor()(img.convert("RGB").resize((224, 224))) if isinstance(img, Image.Image) else img for img in raw_pixel_values]
                bg_01 = torch.stack(bg_01_list).to(self.vla.device, dtype=torch.float32)
            else:
                raw_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.float32)
                bg_01 = torch.clamp((raw_pixel_values[:, 0:3, :, :] * std0) + mean0, 0.0, 1.0)

            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, y:y+patch_h, x:x+patch_w] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch = torch.clamp(patch_strip, 0.0, 1.0)
            padded_patch[:, :, y:y+patch_h, x:x+patch_w] = valid_patch
            
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            im0 = (adv_img_01 - mean0) / std0  
            im1 = (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            # 获取带有 Patch 的当前帧的完整视觉特征 [B, 256, 4096]
            adv_features_global = self.vla.projector(self.vla.vision_backbone(adv_pixel_values_6c)) 
            B = adv_features_global.shape[0]
            
            # =========================================================================
            # 🔴 动态目标构造：为 Batch 中的每个样本随机分配一个“未来状态”特征
            # =========================================================================
            batch_targets = []
            for _ in range(B):
                # 随机抽取一张高能帧的特征
                batch_targets.append(random.choice(z_target_global_pool).squeeze(0))
            
            # 拼接成与 adv_features_global 对应的形状 [B, 256, 4096]
            target_z_batch = torch.stack(batch_targets).to(self.vla.device)

            # 终极降维打击：全局特征覆盖 (乘 10.0 加速梯度回传)
            L_align = F.mse_loss(adv_features_global, target_z_batch) * 10.0
            
            L_align.backward()
            optimizer.step()
            
            with torch.no_grad(): 
                patch_strip.data = patch_strip.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: 
                scheduler.step()

            loss_val = L_align.item()
            self.train_J_loss.append(loss_val)
            pbar.set_postfix({"Global Pool MSE": f"{loss_val:.4f}"})
            
            self._save(i, num_iter, valid_patch, adv_img_01[0], x, y, patch_h, patch_w)

    def _save(self, i, num_iter, patch_strip, demo_img, best_x, best_y, p_h, p_w):
        is_best = False
        if i > 50 and np.mean(self.train_J_loss[-50:]) < self.best_loss: 
            self.best_loss = np.mean(self.train_J_loss[-50:])
            is_best = True

        if i % 100 == 0 or is_best or i == num_iter - 1:
            raw_patch = patch_strip[0].cpu().detach()
            dirs = []
            if i % 100 == 0:
                self.plot_loss()
                dirs.append(os.path.join(self.save_dir, f"{str(i)}"))
            if is_best: dirs.append(os.path.join(self.save_dir, "best"))
            if i == num_iter - 1: dirs.append(os.path.join(self.save_dir, "final_step"))

            for d in dirs:
                os.makedirs(d, exist_ok=True)
                torch.save(raw_patch, os.path.join(d, "raw_patch.pt"))
                
                full_patch = torch.zeros((1, 3, 224, 224))
                full_patch[0, :, best_y:best_y+p_h, best_x:best_x+p_w] = raw_patch
                mask = torch.zeros((1, 1, 224, 224))
                mask[0, 0, best_y:best_y+p_h, best_x:best_x+p_w] = 1.0
                torch.save(torch.cat([full_patch, mask], dim=1), os.path.join(d, "patch.pt"))
                
                img_dir = os.path.join(d, "val_related_data")
                os.makedirs(img_dir, exist_ok=True)
                torchvision.transforms.ToPILImage()(demo_img.cpu()).save(os.path.join(img_dir, "noisy_demo.png"))