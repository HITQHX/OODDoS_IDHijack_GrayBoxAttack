import torch
import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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

def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

class ID_Hijack_Optimize_Attacker(object):
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
        plt.plot(smooth_curve(self.train_J_loss), color='darkviolet', linewidth=2.5)
        plt.title('Optimized State Hallucination (Homography + Stealth)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Joint Objective Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def get_geometric_endpoints(self, x, y, w, h, img_w=224, img_h=224):
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
               patch_h=28, patch_w=224, gamma1=2, gamma2=2, 
               safe_zone=[0, 196, 224, 224], 
               exec_pool_dir="/root/autodl-tmp/roboticAttack/malicious_exec_pool_libero_10_1", **kwargs):
               
        print(f"🧲 [ID Sem-Hijack Optimize] 启动域内流形扰乱联合优化！")
        
        exec_frame_paths = glob.glob(os.path.join(exec_pool_dir, "*.png"))
        if not exec_frame_paths:
            raise FileNotFoundError(f"⚠️ 找不到目标执行帧！请确保 {exec_pool_dir} 有抽取的截帧。")
            
        print(f"🎯 成功加载 {len(exec_frame_paths)} 张核心动作执行帧构建高能特征池...")
        
        z_target_global_pool = []
        with torch.no_grad():
            for frame_path in exec_frame_paths:
                target_img_pil = Image.open(frame_path).convert("RGB").resize((224, 224))
                inputs = self.processor(text=[""], images=[target_img_pil], return_tensors="pt")
                target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
                z_target = self.vla.projector(self.vla.vision_backbone(target_pixel_values)).detach()
                z_target_global_pool.append(z_target)

        patch_strip = torch.rand((1, 3, patch_h, patch_w), device=self.vla.device, requires_grad=True)
        optimizer = torch.optim.AdamW([patch_strip], lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-6)

        mean0, std0 = self.mean[0].view(1,3,1,1).to(self.vla.device), self.std[0].view(1,3,1,1).to(self.vla.device)
        mean1, std1 = self.mean[1].view(1,3,1,1).to(self.vla.device), self.std[1].view(1,3,1,1).to(self.vla.device)

        train_iterator = iter(train_dataloader)
        pbar = tqdm(range(num_iter), desc="ID Hijack Optimized")

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

            L_TV = total_variation_loss(patch_strip)
            
            adv_images = []
            L_bg_total = torch.tensor(0.0).to(self.vla.device)
            last_x, last_y = 0, 0
            
            for b in range(B):
                safe_xmin = max(0, min(safe_zone[0], 224 - patch_w))
                safe_ymin = max(0, min(safe_zone[1], 224 - patch_h))
                safe_xmax = min(safe_zone[2], 224)
                safe_ymax = min(safe_zone[3], 224)
                
                max_x_start = max(safe_xmin, safe_xmax - patch_w)
                max_y_start = max(safe_ymin, safe_ymax - patch_h)

                x = random.randint(safe_xmin, max_x_start)
                y = random.randint(safe_ymin, max_y_start)
                last_x, last_y = x, y

                startpoints = [[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]]
                endpoints = self.get_geometric_endpoints(x, y, patch_w, patch_h)
                
                patch_persp = TF.perspective(patch_strip[0], startpoints, endpoints)
                mask = TF.perspective(torch.ones_like(patch_strip[0]), startpoints, endpoints) > 0.5
                
                bg_img_adv = bg_01[b].clone()
                bg_crop = bg_img_adv[:, y:y+patch_h, x:x+patch_w].clone()
                
                L_bg_total += F.mse_loss(patch_persp * mask, bg_crop * mask)
                
                bg_img_adv[:, y:y+patch_h, x:x+patch_w] = torch.where(mask, patch_persp, bg_crop)
                adv_images.append(bg_img_adv)
                
            adv_img_01 = torch.stack(adv_images)
            L_bg = L_bg_total / B
            
            im0, im1 = (adv_img_01 - mean0) / std0, (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_features_global = self.vla.projector(self.vla.vision_backbone(adv_pixel_values_6c)) 
            
            batch_targets = []
            for _ in range(B):
                batch_targets.append(random.choice(z_target_global_pool).squeeze(0))
            target_z_batch = torch.stack(batch_targets).to(self.vla.device)

            L_align = F.mse_loss(adv_features_global, target_z_batch) * 10.0
            
            J = L_align + gamma1 * L_TV + gamma2 * L_bg
            
            J.backward()
            optimizer.step()
            
            with torch.no_grad(): patch_strip.data = patch_strip.data.clamp(0.0, 1.0)
            if (i + 1) % accumulate_steps == 0: scheduler.step()

            self.train_J_loss.append(J.item())
            pbar.set_postfix({"J": f"{J.item():.3f}", "MSE": f"{L_align.item():.3f}", "TV": f"{L_TV.item():.3f}", "BG": f"{L_bg.item():.3f}"})
            
            self._save(i, num_iter, patch_strip, adv_img_01[0], last_x, last_y, patch_h, patch_w)

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