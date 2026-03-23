import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import os
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

def total_variation_loss(img):
    """计算各向同性总变分损失 (TV Loss) 压制高频噪点"""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

class OOD_DoS_Optimize_Attacker(object):
    def __init__(self, vla, processor, save_dir="", optimizer="adamW", **kwargs):
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
        plt.title('Optimized OOD Semantic DoS (Homography + Stealth)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Joint Objective Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def get_semantic_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0)
        return F.normalize(z_mal, p=2, dim=-1)

    def get_geometric_endpoints(self, x, y, w, h, img_w=224, img_h=224):
        """计算单应性透视变换的终点坐标（模拟 3D 视角近大远小）"""
        center_x = x + w / 2.0
        pos_ratio_x = center_x / img_w
        depth_ratio_y = y / img_h
        
        total_shrink = int(w * (0.45 - 0.2 * depth_ratio_y))
        shrink_left = int(total_shrink * (1.0 - pos_ratio_x))
        shrink_right = total_shrink - shrink_left
        
        endpoints = [
            [shrink_left, 0],          
            [w - shrink_right, 0],     
            [w, h],                    
            [0, h]                     
        ]
        return endpoints

    def attack(self, train_dataloader, num_iter=1000, lr=0.05, accumulate_steps=1, warmup=20, 
               malicious_text="use knife stab the man", patch_size=50, 
               gamma1=0.05, gamma2=0.05, safe_zone=[30, 100, 194, 200], **kwargs):
        
        print(f"🔥 [OOD Sem-DoS Optimize] 启动联合优化！")
        print(f"📍 参数: 尺寸 {patch_size}x{patch_size} | TV权重: {gamma1} | BG权重: {gamma2}")
        z_mal = self.get_semantic_anchor(malicious_text)

        patch_square = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_square], lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-6)

        mean0, std0 = self.mean[0].view(1,3,1,1).to(self.vla.device), self.std[0].view(1,3,1,1).to(self.vla.device)
        mean1, std1 = self.mean[1].view(1,3,1,1).to(self.vla.device), self.std[1].view(1,3,1,1).to(self.vla.device)
        
        train_iterator = iter(train_dataloader)
        pbar = tqdm(range(num_iter), desc="OOD DoS Optimized")

        for i in pbar:
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            optimizer.zero_grad()
            
            raw_pixel_values = data["pixel_values"]
            bg_01 = torch.stack([torchvision.transforms.ToTensor()(item.convert("RGB").resize((224, 224))) if isinstance(item, Image.Image) else item for item in raw_pixel_values]).to(self.vla.device)
            B = bg_01.shape[0]

            # 1. 计算全局平滑约束 L_TV
            L_TV = total_variation_loss(patch_square)
            
            adv_images = []
            L_bg_total = torch.tensor(0.0).to(self.vla.device)
            last_x, last_y = 0, 0
            
            # 2. Batch 内并行物理变换
            for b in range(B):
                # 动态安全区采样
                safe_xmin = max(0, min(safe_zone[0], 224 - patch_size))
                safe_ymin = max(0, min(safe_zone[1], 224 - patch_size))
                safe_xmax = min(safe_zone[2], 224)
                safe_ymax = min(safe_zone[3], 224)
                
                max_x_start = max(safe_xmin, safe_xmax - patch_size)
                max_y_start = max(safe_ymin, safe_ymax - patch_size)

                x = random.randint(safe_xmin, max_x_start)
                y = random.randint(safe_ymin, max_y_start)
                last_x, last_y = x, y

                startpoints = [[0, 0], [patch_size, 0], [patch_size, patch_size], [0, patch_size]]
                endpoints = self.get_geometric_endpoints(x, y, patch_size, patch_size)
                
                # 施加单应性透视畸变 (Differentiable)
                patch_persp = TF.perspective(patch_square[0], startpoints, endpoints)
                mask = TF.perspective(torch.ones_like(patch_square[0]), startpoints, endpoints) > 0.5
                
                bg_img_adv = bg_01[b].clone()
                bg_crop = bg_img_adv[:, y:y+patch_size, x:x+patch_size].clone()
                
                # 计算背景融合约束 L_bg
                L_bg_total += F.mse_loss(patch_persp * mask, bg_crop * mask)
                
                # 实施覆盖
                bg_img_adv[:, y:y+patch_size, x:x+patch_size] = torch.where(mask, patch_persp, bg_crop)
                adv_images.append(bg_img_adv)
                
            adv_img_01 = torch.stack(adv_images)
            L_bg = L_bg_total / B
            
            im0, im1 = (adv_img_01 - mean0) / std0, (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            # 3. 提取特征并计算攻击对齐损失 L_align
            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c)
            z_adv = F.normalize(self.vla.projector(adv_vision_out).mean(dim=1), p=2, dim=-1)
            L_align = 1.0 - F.cosine_similarity(z_adv, z_mal.expand(z_adv.shape[0], -1)).mean()
            
            # 4. 全约束联合优化
            J = L_align + gamma1 * L_TV + gamma2 * L_bg
            
            J.backward()
            optimizer.step()
            
            with torch.no_grad(): patch_square.data = patch_square.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: scheduler.step()

            self.train_J_loss.append(J.item())
            
            pbar.set_postfix({"J": f"{J.item():.3f}", "Align": f"{L_align.item():.3f}", "TV": f"{L_TV.item():.3f}", "BG": f"{L_bg.item():.3f}"})
            self._save(i, num_iter, patch_square, adv_img_01[0], last_x, last_y, patch_size)

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