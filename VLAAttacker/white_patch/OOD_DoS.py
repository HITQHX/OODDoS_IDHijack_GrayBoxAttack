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

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

class OOD_DoS_Attacker(object):
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
        plt.title('OOD Semantic DoS Attack (Random Square Patch)', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Cosine Distance Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def get_semantic_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0)
        return F.normalize(z_mal, p=2, dim=-1)

    def attack(self, train_dataloader, num_iter=1000, lr=0.05, accumulate_steps=1, warmup=20, malicious_text="use knife stab the man", patch_size=50, **kwargs):
        print(f"🔥 [OOD Sem-DoS] 启动域外拒绝服务！方形尺寸: {patch_size}x{patch_size}，目标文本: '{malicious_text}'")
        z_mal = self.get_semantic_anchor(malicious_text)

        # 1. 初始化指定尺寸的独立方形贴纸
        patch_square = torch.rand((1, 3, patch_size, patch_size), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.AdamW([patch_square], lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter // accumulate_steps, eta_min=1e-6)

        mean0, std0 = self.mean[0].view(1,3,1,1).to(self.vla.device), self.std[0].view(1,3,1,1).to(self.vla.device)
        mean1, std1 = self.mean[1].view(1,3,1,1).to(self.vla.device), self.std[1].view(1,3,1,1).to(self.vla.device)
        
        train_iterator = iter(train_dataloader)

        for i in tqdm(range(num_iter), desc="OOD DoS 优化中"):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            optimizer.zero_grad()
            
            raw_pixel_values = data["pixel_values"]
            bg_01 = torch.stack([torchvision.transforms.ToTensor()(item.convert("RGB").resize((224, 224))) if isinstance(item, Image.Image) else item for item in raw_pixel_values]).to(self.vla.device)

            # 2. 🔴 随机游走机制：在安全区域（桌面）内随机生成坐标
            x_start = random.randint(30, 224 - patch_size - 30)
            y_start = random.randint(100, 224 - patch_size - 24)
            
            mask = torch.zeros((1, 1, 224, 224), device=self.vla.device)
            mask[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = 1.0
            
            padded_patch = torch.zeros((1, 3, 224, 224), device=self.vla.device)
            valid_patch = torch.clamp(patch_square, 0.0, 1.0)
            padded_patch[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = valid_patch

            # 纯粹替换
            adv_img_01 = (1.0 - mask) * bg_01 + mask * padded_patch
            
            im0, im1 = (adv_img_01 - mean0) / std0, (adv_img_01 - mean1) / std1  
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1).to(torch.bfloat16)

            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c)
            # 全局特征崩溃拉扯
            z_adv = F.normalize(self.vla.projector(adv_vision_out).mean(dim=1), p=2, dim=-1)
            
            L_align = 1.0 - F.cosine_similarity(z_adv, z_mal.expand(z_adv.shape[0], -1)).mean()
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
            # 🔴 同时保存 [3, size, size] 原始贴纸，方便物理测试时随意缩放和移动！
            raw_patch = patch_square[0].cpu().detach()
            
            dirs = []
            if i % 100 == 0:
                self.plot_loss()
                dirs.append(os.path.join(self.save_dir, f"{str(i)}"))
            if is_best: dirs.append(os.path.join(self.save_dir, "best"))
            if i == num_iter - 1: dirs.append(os.path.join(self.save_dir, "final_step"))

            for d in dirs:
                os.makedirs(d, exist_ok=True)
                # 保存纯粹的正方形张量
                torch.save(raw_patch, os.path.join(d, "raw_patch.pt"))
                
                # 为了兼容目前的测试脚本，默认生成一张中心带掩码的 4 通道图
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