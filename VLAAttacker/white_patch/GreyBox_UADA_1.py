'''
import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from white_patch.appply_random_transform import RandomPatchTransform
from tqdm import tqdm
import os
import transformers
import pickle
import random
from PIL import Image

IGNORE_INDEX = -100

def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images, mean, std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

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
    def __init__(self, vla, processor, save_dir="", optimizer="adam", resize_patch=False):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.randomPatchTransform = RandomPatchTransform(self.vla.device, resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.train_L_feat = []
        self.train_L_align = []
        self.train_L_camou = []
        self.train_L_bg = []
        
        self.best_moving_avg_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        x_ticks = list(range(0, len(self.train_J_loss)))

        # 🔴 解决曲线平直的视觉错觉：采用 4 行独立子图，各自拥有独立的 Y 轴自适应缩放！
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 1. 宏观总目标
        axs[0].plot(x_ticks, smooth_curve(self.train_J_loss), color='black', linewidth=2.5)
        axs[0].set_title('Total Weighted Objective (J_Loss)', fontsize=12, fontweight='bold')
        axs[0].set_ylabel('Value')
        
        # 2. 语义劫持 (你的核心目标，现在它的微小下降将被清晰放大)
        axs[1].plot(x_ticks, smooth_curve(self.train_L_align), color='red', linewidth=2.5)
        axs[1].set_title('Cognitive Hijack (L_align)', fontsize=12, fontweight='bold')
        axs[1].set_ylabel('Cosine Distance')
        
        # 3. 视觉致盲
        axs[2].plot(x_ticks, smooth_curve(self.train_L_feat), color='blue', linewidth=2.5)
        axs[2].set_title('Sensory Blindness (L_feat)', fontsize=12, fontweight='bold')
        axs[2].set_ylabel('Feature Similarity')
        
        # 4. 物理伪装与平滑度
        axs[3].plot(x_ticks, smooth_curve(self.train_L_camou), color='green', label='Smoothness (L_camou)', linewidth=2)
        axs[3].plot(x_ticks, smooth_curve(self.train_L_bg), color='purple', label='Background Stealth (L_bg)', linestyle='--', linewidth=2)
        axs[3].set_title('Physical Stealth Metrics', fontsize=12, fontweight='bold')
        axs[3].set_xlabel('Optimization Iterations', fontsize=12)
        axs[3].set_ylabel('Loss Value')
        axs[3].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close(fig)

    def save_info(self, path):
        with open(os.path.join(path, 'train_J_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_J_loss, file)
        with open(os.path.join(path, 'train_L_feat.pkl'), 'wb') as file:
            pickle.dump(self.train_L_feat, file)
        with open(os.path.join(path, 'train_L_align.pkl'), 'wb') as file:
            pickle.dump(self.train_L_align, file)
        with open(os.path.join(path, 'train_L_camou.pkl'), 'wb') as file:
            pickle.dump(self.train_L_camou, file)

    def get_semantic_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0)
            z_mal = F.normalize(z_mal, p=2, dim=-1)
        return z_mal

    def compute_tv_loss(self, patch):
        tv_h = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).mean()
        tv_w = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).mean()
        return tv_h + tv_w

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=5000, 
                            patch_size=[3, 50, 50], lr=0.03, accumulate_steps=1, 
                            malicious_text="swing the knife and destroy the object",
                            alpha=0.5, beta=20.0, gamma=0.1, tau=0.5,
                            geometry=True, innerLoop=1, maskidx=None, warmup=20):
        
        SAFE_ZONES = [
            [10, 200, 80, 270],   
            [150, 200, 220, 270], 
            [90, 200, 160, 270]   
        ]

        print("🔍 正在提取真实桌面，初始化物理本征向量...")
        init_data = next(iter(train_dataloader))
        init_image = init_data["pixel_values"][0] 
        
        if isinstance(init_image, Image.Image) or isinstance(init_image, np.ndarray):
            init_img_tensor = torchvision.transforms.ToTensor()(init_image).to(self.vla.device)
        else:
            init_img_tensor = init_image.clone().to(self.vla.device)
            
        c, h, w = patch_size
        crop_x, crop_y = SAFE_ZONES[0][0], SAFE_ZONES[0][1]
        original_bg = init_img_tensor[:, crop_y:crop_y+h, crop_x:crop_x+w].clone().detach()
        
        if original_bg.shape[1] != h or original_bg.shape[2] != w:
            original_bg = F.interpolate(original_bg.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        # ====================================================================
        # 🔴 你的神级机制：物理光学分解 (Intrinsic Image Decomposition)
        # 抛弃自由 RGB，仅优化单通道的 "阴影(alpha)" 和 "反光(beta)"
        # ====================================================================
        delta_alpha = torch.zeros((1, h, w), device=self.vla.device, requires_grad=True)
        delta_beta  = torch.zeros((1, h, w), device=self.vla.device, requires_grad=True)

        # 优化器现在完全无法触碰色相(Hue)的维度！
        optimizer = torch.optim.Adam([delta_alpha, delta_beta], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        z_mal = self.get_semantic_anchor(malicious_text)
        train_iterator = iter(train_dataloader)
        previous_adv_features = None

        print(f"🚀 [Grey-Box] 开始本征光学约束优化 | 目标: '{malicious_text}' | LR: {lr}")

        for i in tqdm(range(num_iter)):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            raw_pixel_values = data["pixel_values"]
            
            with torch.no_grad():
                if isinstance(raw_pixel_values, list):
                    dummy_text = [""] * len(raw_pixel_values)
                    inputs = self.processor(text=dummy_text, images=raw_pixel_values, return_tensors="pt")
                    clean_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
                else:
                    clean_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.bfloat16)

                clean_vision_out = self.vla.vision_backbone(clean_pixel_values)
                z_clean = F.normalize(self.vla.projector(clean_vision_out).mean(dim=1), p=2, dim=-1)

            optimizer.zero_grad()
            J_loss_sum, L_feat_sum, L_align_sum, L_camou_sum, L_bg_sum = 0, 0, 0, 0, 0
            
            for inner_loop in range(innerLoop):
                current_safe_zone = random.choice(SAFE_ZONES)
                
                # 🔴 核心合成公式：原始颜色 * (1 + 阴影) + 反光
                # 这保证了无论如何优化，颜色永远沿着 [黄->棕] 或 [黄->浅黄] 的直线运动！
                patch = torch.clamp(original_bg * (1.0 + delta_alpha) + delta_beta, 0, 1)

                modified_images, L_bg = self.randomPatchTransform.apply_stealth_perspective_batch(
                    raw_pixel_values, patch, self.mean, self.std, safe_zone=current_safe_zone)

                adv_vision_out = self.vla.vision_backbone(modified_images.to(torch.bfloat16))
                adv_features = self.vla.projector(adv_vision_out).mean(dim=1)
                z_adv = F.normalize(adv_features, p=2, dim=-1)
                
                L_feat = F.cosine_similarity(z_adv, z_clean).mean()
                L_align = 1.0 - F.cosine_similarity(z_adv, z_mal.unsqueeze(0).expand(z_adv.shape[0], -1)).mean()
                
                # 🔴 TV 平滑度只约束我们添加的阴影和反光层，完美保留了桌子原本的木纹！
                L_TV = self.compute_tv_loss(delta_alpha) + self.compute_tv_loss(delta_beta)
                
                if previous_adv_features is not None:
                    L_temporal = F.mse_loss(adv_features, previous_adv_features.detach())
                else:
                    L_temporal = torch.tensor(0.0).to(self.vla.device)
                L_camou = L_TV + tau * L_temporal

                eta = 2.0 
                J_loss = alpha * L_feat + beta * L_align + gamma * L_camou + eta * L_bg

                (J_loss / innerLoop).backward()
                previous_adv_features = adv_features.detach()
                
                J_loss_sum += J_loss.item()
                L_feat_sum += L_feat.item()
                L_align_sum += L_align.item()
                L_camou_sum += L_camou.item()
                L_bg_sum += L_bg.item()

            optimizer.step()
            
            # ====================================================================
            # 🔴 极其严格的物理极限约束
            # alpha 限制在 ±0.3 (允许加深30%变成棕色)
            # beta 限制在 ±0.1 (允许增加10%的白色高光)
            # ====================================================================
            with torch.no_grad():
                delta_alpha.data = delta_alpha.data.clamp(-0.1, 0.1)
                delta_beta.data  = delta_beta.data.clamp(-0.1, 0.1)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(J_loss_sum / innerLoop)
            self.train_L_feat.append(L_feat_sum / innerLoop)
            self.train_L_align.append(L_align_sum / innerLoop)
            self.train_L_camou.append(L_camou_sum / innerLoop)
            self.train_L_bg.append(L_bg_sum / innerLoop)

            if i % 100 == 0:
                self.plot_loss()
                
                patch_save = torch.clamp(original_bg * (1.0 + delta_alpha) + delta_beta, 0, 1)

                if i == 0:
                    print(f"\n📸 [初始化] 记录天然完美伪装状态...")
                    temp_save_dir = os.path.join(self.save_dir, "0")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch_save.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    with torch.no_grad():
                        denorm_images = self.randomPatchTransform.denormalize(
                            modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                        for o in range(min(4, denorm_images.shape[0])):
                            pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                            pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                
                elif i > 0:
                    recent_avg_J = np.mean(self.train_J_loss[-100:])
                    recent_avg_align = np.mean(self.train_L_align[-100:])
                    
                    # 🔴 你要求的终端完美日志输出
                    print(f"\n📊 [Iter {i:04d}] 综合性能监控:")
                    print(f"   ➤ [主目标] J_Loss: {recent_avg_J:.4f} | L_align (语义对齐): {recent_avg_align:.4f}")
                    print(f"   ➤ [副目标] L_feat (致盲度): {np.mean(self.train_L_feat[-100:]):.4f} | L_camou (平滑度): {np.mean(self.train_L_camou[-100:]):.4f} | L_bg (背景融合): {np.mean(self.train_L_bg[-100:]):.4f}")

                    if recent_avg_J < self.best_moving_avg_loss:
                        print(f"   🌟 新突破! 平均 J_Loss 降至新低 {recent_avg_J:.4f}，正在保存最优补丁...")
                        self.best_moving_avg_loss = recent_avg_J
                        
                        temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                        os.makedirs(temp_save_dir, exist_ok=True)
                        torch.save(patch_save.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                        
                        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                        os.makedirs(val_related_file_path, exist_ok=True)
                        with torch.no_grad():
                            denorm_images = self.randomPatchTransform.denormalize(
                                modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                            for o in range(min(4, denorm_images.shape[0])):
                                pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                                pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                            
                        last_save_dir = os.path.join(self.save_dir, "last")
                        os.makedirs(last_save_dir, exist_ok=True)
                        torch.save(patch_save.detach().cpu(), os.path.join(last_save_dir, "patch.pt"))
                        
                        last_img_dir = os.path.join(last_save_dir, "val_related_data")
                        os.makedirs(last_img_dir, exist_ok=True)
                        for o in range(min(4, denorm_images.shape[0])):
                            pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                            pil_img.save(os.path.join(last_img_dir, f"{str(o)}.png"))
                            
                self.save_info(path=self.save_dir)


import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from white_patch.appply_random_transform import RandomPatchTransform
from tqdm import tqdm
import os
import transformers
import pickle
import random
from PIL import Image

IGNORE_INDEX = -100

def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images, mean, std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

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
    def __init__(self, vla, processor, save_dir="", optimizer="adam", resize_patch=False):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.randomPatchTransform = RandomPatchTransform(self.vla.device, resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.train_L_align = []
        self.train_L_camou = []
        self.train_L_bg = []
        
        self.best_moving_avg_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        x_ticks = list(range(0, len(self.train_J_loss)))

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        axs[0].plot(x_ticks, smooth_curve(self.train_J_loss), color='black', linewidth=2.5)
        axs[0].set_title('Total Uncertainty-Weighted Objective (J_Loss)', fontsize=12, fontweight='bold')
        
        # 移除了碍事的 L_feat，只关注 L_align
        axs[1].plot(x_ticks, smooth_curve(self.train_L_align), color='red', linewidth=2.5)
        axs[1].set_title('Cognitive Hijack (L_align)', fontsize=12, fontweight='bold')
        
        axs[2].plot(x_ticks, smooth_curve(self.train_L_camou), color='green', label='Anisotropic Smoothness', linewidth=2)
        axs[2].plot(x_ticks, smooth_curve(self.train_L_bg), color='purple', label='Background Stealth', linestyle='--', linewidth=2)
        axs[2].set_title('Physical Stealth Metrics', fontsize=12, fontweight='bold')
        axs[2].set_xlabel('Optimization Iterations', fontsize=12)
        axs[2].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close(fig)

    def save_info(self, path):
        with open(os.path.join(path, 'train_J_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_J_loss, file)
        with open(os.path.join(path, 'train_L_align.pkl'), 'wb') as file:
            pickle.dump(self.train_L_align, file)
        with open(os.path.join(path, 'train_L_camou.pkl'), 'wb') as file:
            pickle.dump(self.train_L_camou, file)

    def get_semantic_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0)
            z_mal = F.normalize(z_mal, p=2, dim=-1)
        return z_mal

    # 创新 2：各向异性平滑 (Anisotropic TV)
    # 让生成的纹理顺着原木纹的方向生长！
    def compute_anisotropic_tv_loss(self, noise, original_bg):
        # 1. 计算原背景的自然纹理方向
        bg_dx = torch.abs(original_bg[:, :, 1:] - original_bg[:, :, :-1]).mean().item() + 1e-6
        bg_dy = torch.abs(original_bg[:, 1:, :] - original_bg[:, :-1, :]).mean().item() + 1e-6
        
        # 2. 计算权重：如果背景在 X 变化剧烈(横纹)，我们就允许噪声 X 变化剧烈，严惩 Y 方向的跳变
        weight_dx = bg_dy / (bg_dx + bg_dy)
        weight_dy = bg_dx / (bg_dx + bg_dy)
        
        # 3. 对抗噪声的 TV 计算
        noise_dx = torch.pow(noise[:, :, 1:] - noise[:, :, :-1], 2).mean()
        noise_dy = torch.pow(noise[:, 1:, :] - noise[:, :-1, :], 2).mean()
        
        return weight_dx * noise_dx + weight_dy * noise_dy

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=5000, 
                            patch_size=[3, 50, 50], lr=0.03, accumulate_steps=1, 
                            malicious_text="swing the knife and destroy the object",
                            # 恢复最纯粹的力量：主攻对齐，仅保留极微弱的平滑度约束，彻底抛弃 L_bg
                            beta=1.0, gamma=0.05, 
                            geometry=True, innerLoop=1, maskidx=None, warmup=20):
        
        SAFE_ZONES = [
            [10, 200, 80, 270],   
            [170, 200, 240, 270], 
            [90, 200, 160, 270]   
        ]

        print("🔍 提取桌面背景，初始化各向异性本征向量...")
        init_data = next(iter(train_dataloader))
        init_image = init_data["pixel_values"][0] 
        
        if isinstance(init_image, Image.Image) or isinstance(init_image, np.ndarray):
            init_img_tensor = torchvision.transforms.ToTensor()(init_image).to(self.vla.device)
        else:
            init_img_tensor = init_image.clone().to(self.vla.device)
            
        c, h, w = patch_size
        crop_x, crop_y = SAFE_ZONES[0][0], SAFE_ZONES[0][1]
        original_bg = init_img_tensor[:, crop_y:crop_y+h, crop_x:crop_x+w].clone().detach()
        if original_bg.shape[1] != h or original_bg.shape[2] != w:
            original_bg = F.interpolate(original_bg.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        delta_alpha = torch.zeros((1, h, w), device=self.vla.device, requires_grad=True)
        delta_beta  = torch.zeros((1, h, w), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.Adam([delta_alpha, delta_beta], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        z_mal = self.get_semantic_anchor(malicious_text)
        train_iterator = iter(train_dataloader)

        print(f"🚀 [Grey-Box] 开始纯粹本征攻击 (彻底解除橡皮筋效应) | 目标: '{malicious_text}'")

        for i in tqdm(range(num_iter)):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            raw_pixel_values = data["pixel_values"]
            optimizer.zero_grad()
            J_loss_sum, L_align_sum, L_camou_sum = 0, 0, 0
            
            for inner_loop in range(innerLoop):
                current_safe_zone = random.choice(SAFE_ZONES)
                patch = torch.clamp(original_bg * (1.0 + delta_alpha) + delta_beta, 0, 1)

                modified_images, _ = self.randomPatchTransform.apply_stealth_perspective_batch(
                    raw_pixel_values, patch, self.mean, self.std, safe_zone=current_safe_zone)

                adv_vision_out = self.vla.vision_backbone(modified_images.to(torch.bfloat16))
                adv_features = self.vla.projector(adv_vision_out).mean(dim=1)
                z_adv = F.normalize(adv_features, p=2, dim=-1)
                
                z_mal_batch = z_mal.unsqueeze(0).expand(z_adv.shape[0], -1)
                L_align = 1.0 - F.cosine_similarity(z_adv, z_mal_batch).mean()
                L_camou = self.compute_anisotropic_tv_loss(delta_alpha, original_bg) + self.compute_anisotropic_tv_loss(delta_beta, original_bg)

                # ====================================================================
                # 前 10% 步数零约束猛攻，之后维持一个极小的常数约束 (gamma=0.05) 进行平滑抛光
                # ====================================================================
                progress = i / num_iter
                current_gamma = 0.1 if progress < 0.4 else gamma

                J_loss = beta * L_align + current_gamma * L_camou

                (J_loss / innerLoop).backward()
                
                J_loss_sum += J_loss.item()
                L_align_sum += L_align.item()
                L_camou_sum += L_camou.item()

            optimizer.step()
            
            with torch.no_grad():
                delta_alpha.data = delta_alpha.data.clamp(-0.25, 0.25)
                delta_beta.data  = delta_beta.data.clamp(-0.25, 0.25)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(J_loss_sum / innerLoop)
            self.train_L_align.append(L_align_sum / innerLoop)
            self.train_L_camou.append(L_camou_sum / innerLoop)
            
            # 为了保持画图函数不报错，填入占位符，其实 L_bg 已经被废弃了
            self.train_L_bg.append(0.0) 

            if i % 100 == 0:
                self.plot_loss()
                patch_save = torch.clamp(original_bg * (1.0 + delta_alpha) + delta_beta, 0, 1)

                # ====================================================================
                #  1. 无条件快照：无论好坏，必定保存当前 step 的独立文件夹
                # ====================================================================
                temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(patch_save.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                
                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                os.makedirs(val_related_file_path, exist_ok=True)
                
                with torch.no_grad():
                    denorm_images = self.randomPatchTransform.denormalize(
                        modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                    # 解除限制：完整遍历整个 Batch (8张图全部保存)
                    for o in range(denorm_images.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                
                # ====================================================================
                # 2. 终端日志与精英保留策略
                # ====================================================================
                if i == 0:
                    print(f"\n📸 [Iter 0000] 记录天然完美伪装初始状态...")
                elif i > 0:
                    recent_avg_align = np.mean(self.train_L_align[-100:])
                    
                    print(f"\n📊 [Iter {i:04d}] 优化监控:")
                    print(f"   ➤ [约束状态] W_camou: {current_gamma:.3f} | L_bg 已被物理级投影替代！")
                    print(f"   ➤ [实际误差] L_align (越低越好): {recent_avg_align:.4f} | L_camou: {np.mean(self.train_L_camou[-100:]):.4f}")

                    # 只有突破历史最佳时，才去覆盖 'last' 目录（供后续测试脚本调用）
                    if recent_avg_align < self.best_moving_avg_loss: 
                        print(f"   🌟 新突破! 语义对齐降至 {recent_avg_align:.4f}，已更新全局最优补丁 (last 目录)！")
                        self.best_moving_avg_loss = recent_avg_align
                        
                        last_save_dir = os.path.join(self.save_dir, "last")
                        os.makedirs(last_save_dir, exist_ok=True)
                        torch.save(patch_save.detach().cpu(), os.path.join(last_save_dir, "patch.pt"))
                        
                        last_img_dir = os.path.join(last_save_dir, "val_related_data")
                        os.makedirs(last_img_dir, exist_ok=True)
                        # 解除限制：完整遍历整个 Batch
                        for o in range(denorm_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                            pil_img.save(os.path.join(last_img_dir, f"{str(o)}.png"))
                            
                self.save_info(path=self.save_dir)


import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from white_patch.appply_random_transform import RandomPatchTransform
from tqdm import tqdm
import os
import transformers
import pickle
import random
from PIL import Image

IGNORE_INDEX = -100

def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images, mean, std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

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
    def __init__(self, vla, processor, save_dir="", optimizer="adam", resize_patch=False):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.randomPatchTransform = RandomPatchTransform(self.vla.device, resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.train_L_align = []
        self.train_L_camou = []
        self.train_L_bg = [] # 保留此空数组防止画图函数崩溃
        
        self.best_moving_avg_loss = float('inf')

    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        x_ticks = list(range(0, len(self.train_J_loss)))

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        axs[0].plot(x_ticks, smooth_curve(self.train_J_loss), color='black', linewidth=2.5)
        axs[0].set_title('Total Objective (J_Loss)', fontsize=12, fontweight='bold')
        
        axs[1].plot(x_ticks, smooth_curve(self.train_L_align), color='red', linewidth=2.5)
        axs[1].set_title('Cognitive Hijack (L_align)', fontsize=12, fontweight='bold')
        
        axs[2].plot(x_ticks, smooth_curve(self.train_L_camou), color='green', label='Smoothness & Temporal', linewidth=2)
        axs[2].set_title('Physical Stealth Metrics', fontsize=12, fontweight='bold')
        axs[2].set_xlabel('Optimization Iterations', fontsize=12)
        axs[2].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close(fig)

    def save_info(self, path):
        with open(os.path.join(path, 'train_J_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_J_loss, file)
        with open(os.path.join(path, 'train_L_align.pkl'), 'wb') as file:
            pickle.dump(self.train_L_align, file)
        with open(os.path.join(path, 'train_L_camou.pkl'), 'wb') as file:
            pickle.dump(self.train_L_camou, file)

    def get_semantic_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0)
            z_mal = F.normalize(z_mal, p=2, dim=-1)
        return z_mal

    # 恢复基础的各向同性 TV (因为现在背景是动态的，无需参考原背景)
    def compute_tv_loss(self, patch):
        tv_h = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).mean()
        tv_w = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).mean()
        return tv_h + tv_w

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=5000, 
                            patch_size=[3, 50, 50], lr=0.03, accumulate_steps=1, 
                            malicious_text="swing the knife and destroy the object",
                            beta=1.0, gamma=0.05, tau=0.5, # 🔴 添加了你的 tau 参数
                            geometry=True, innerLoop=1, maskidx=None, warmup=20):
        
        SAFE_ZONES = [
            [10, 200, 80, 270],   
            [170, 200, 240, 270], 
            [90, 200, 160, 270]   
        ]

        print("🔍 启动无底色变色龙光影层初始化...")
        c, h, w = patch_size
        
        # 🔴 彻底舍弃截取静态桌面！初始化两个纯净的透明光影层
        delta_alpha = torch.zeros((1, h, w), device=self.vla.device, requires_grad=True)
        delta_beta  = torch.zeros((1, h, w), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.Adam([delta_alpha, delta_beta], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        z_mal = self.get_semantic_anchor(malicious_text)
        train_iterator = iter(train_dataloader)
        
        # 用于 L_temporal
        previous_adv_features = None 

        print(f"🚀 [Grey-Box] 开始跨任务通用光影攻击 | 目标: '{malicious_text}'")

        for i in tqdm(range(num_iter)):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            raw_pixel_values = data["pixel_values"]
            optimizer.zero_grad()
            J_loss_sum, L_align_sum, L_camou_sum = 0, 0, 0
            
            for inner_loop in range(innerLoop):
                current_safe_zone = random.choice(SAFE_ZONES)
                
                # 🔴 传入透明的光影矩阵，底层变换算子会自动吸取不同的桌子背景进行融合
                modified_images, _ = self.randomPatchTransform.apply_stealth_perspective_batch(
                    raw_pixel_values, delta_alpha, delta_beta, self.mean, self.std, safe_zone=current_safe_zone)

                adv_vision_out = self.vla.vision_backbone(modified_images.to(torch.bfloat16))
                adv_features = self.vla.projector(adv_vision_out).mean(dim=1)
                z_adv = F.normalize(adv_features, p=2, dim=-1)
                
                z_mal_batch = z_mal.unsqueeze(0).expand(z_adv.shape[0], -1)
                L_align = 1.0 - F.cosine_similarity(z_adv, z_mal_batch).mean()
                
                # 🔴 恢复完整的 L_camou = L_TV + tau * L_temporal
                L_TV = self.compute_tv_loss(delta_alpha) + self.compute_tv_loss(delta_beta)
                if previous_adv_features is not None:
                    L_temporal = F.mse_loss(adv_features, previous_adv_features.detach())
                else:
                    L_temporal = torch.tensor(0.0).to(self.vla.device)
                
                L_camou = L_TV + tau * L_temporal

                progress = i / num_iter
                current_gamma = 0.1 if progress < 0.2 else gamma

                J_loss = beta * L_align + current_gamma * L_camou

                (J_loss / innerLoop).backward()
                
                # 记录以备下一轮 temporal 计算
                previous_adv_features = adv_features.detach()
                
                J_loss_sum += J_loss.item()
                L_align_sum += L_align.item()
                L_camou_sum += L_camou.item()

            optimizer.step()
            
            with torch.no_grad():
                delta_alpha.data = delta_alpha.data.clamp(-0.15, 0.15)
                delta_beta.data  = delta_beta.data.clamp(-0.15, 0.15)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(J_loss_sum / innerLoop)
            self.train_L_align.append(L_align_sum / innerLoop)
            self.train_L_camou.append(L_camou_sum / innerLoop)
            self.train_L_bg.append(0.0) 

            if i % 100 == 0:
                self.plot_loss()
                
                # 🔴 极其重要的改动：我们将提取好的两张透明网格拼成 [2, 50, 50] 的张量保存！
                # 这样测试阶段加载 patch.pt 时，就能完美识别它是一个“光影滤镜”而不是RGB图片
                patch_save = torch.cat([delta_alpha, delta_beta], dim=0)

                temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(patch_save.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                
                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                os.makedirs(val_related_file_path, exist_ok=True)
                
                with torch.no_grad():
                    denorm_images = self.randomPatchTransform.denormalize(
                        modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                    for o in range(denorm_images.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                
                if i == 0:
                    print(f"\n📸 [Iter 0000] 记录天然完美伪装初始状态...")
                elif i > 0:
                    recent_avg_align = np.mean(self.train_L_align[-100:])
                    
                    print(f"\n📊 [Iter {i:04d}] 优化监控:")
                    print(f"   ➤ [实际误差] L_align (越低越好): {recent_avg_align:.4f} | L_camou (包含TV+Temporal): {np.mean(self.train_L_camou[-100:]):.4f}")

                    if recent_avg_align < self.best_moving_avg_loss: 
                        print(f"   🌟 新突破! 语义对齐降至 {recent_avg_align:.4f}，已更新全局最优物理滤镜 (last 目录)！")
                        self.best_moving_avg_loss = recent_avg_align
                        
                        last_save_dir = os.path.join(self.save_dir, "last")
                        os.makedirs(last_save_dir, exist_ok=True)
                        torch.save(patch_save.detach().cpu(), os.path.join(last_save_dir, "patch.pt"))
                        
                        last_img_dir = os.path.join(last_save_dir, "val_related_data")
                        os.makedirs(last_img_dir, exist_ok=True)
                        for o in range(denorm_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                            pil_img.save(os.path.join(last_img_dir, f"{str(o)}.png"))
                            
                self.save_info(path=self.save_dir)
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
from PIL import Image

IGNORE_INDEX = -100

def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images, mean, std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

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
    def __init__(self, vla, processor, save_dir="", optimizer="adam"):
        self.vla = vla.eval()
        self.vla.requires_grad_(False) 

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        
        # OpenVLA 默认的图像均值和方差
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        self.train_J_loss = []
        self.best_moving_avg_loss = float('inf')
        
    def plot_loss(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(smooth_curve(self.train_J_loss), color='red', linewidth=2.5, label='Visual Feature Alignment Loss')
        plt.title('Full-Image Unconstrained Attack Upper Bound', fontsize=14, fontweight='bold')
        plt.xlabel('Optimization Iterations')
        plt.ylabel('Cosine Distance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'), dpi=300)
        plt.close()

    def save_info(self, path):
        with open(os.path.join(path, 'train_J_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_J_loss, file)
        with open(os.path.join(path, 'train_L_align.pkl'), 'wb') as file:
            pickle.dump(self.train_L_align, file)
        with open(os.path.join(path, 'train_L_camou.pkl'), 'wb') as file:
            pickle.dump(self.train_L_camou, file)

    '''
    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=1000, 
                            lr=0.05, accumulate_steps=1, warmup=20, **kwargs):
        
        print("🔍 [阶段 1] 正在提取【目标恶意任务】的纯视觉特征作为靶点...")
        
        # ====================================================================
        # 🔴 完美修复：使用 processor 稳健处理可能为 list 的原始图像数据
        # ====================================================================
        target_data = next(iter(val_dataloader))
        raw_target_pixels = target_data["pixel_values"]
        
        with torch.no_grad():
            if isinstance(raw_target_pixels, list):
                dummy_text = [""] * len(raw_target_pixels)
                inputs = self.processor(text=dummy_text, images=raw_target_pixels, return_tensors="pt")
                target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            else:
                target_pixel_values = raw_target_pixels.to(self.vla.device, dtype=torch.bfloat16)
            
            # 取 batch 的第一张图作为我们的 "锚点"
            target_vision_out = self.vla.vision_backbone(target_pixel_values[0:1]) 
            z_target = F.normalize(self.vla.projector(target_vision_out).mean(dim=1), p=2, dim=-1)

        print("🎯 视觉靶点已锁定！彻底抛弃文本指令干扰。")

        # ====================================================================
        # 🔴 全局 RGB 噪声初始化 (在 [0, 1] 空间定义，方便直观限制变化幅度)
        # ====================================================================
        delta_noise_01 = torch.zeros((1, 3, 224, 224), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.Adam([delta_noise_01], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        train_iterator = iter(train_dataloader)
        
        # 提取 OpenVLA 的 mean 和 std 用于维度转换
        std_tensor = self.std[0].view(1, 3, 1, 1).to(self.vla.device)
        mean_tensor = self.mean[0].view(1, 3, 1, 1).to(self.vla.device)

        print(f"🚀 [阶段 2] 开始毫无保留的全域视觉特征碰撞攻击！")

        for i in tqdm(range(num_iter)):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            raw_pixel_values = data["pixel_values"]
            optimizer.zero_grad()
            
            # 1. 稳健提取干净的 normalized 张量
            if isinstance(raw_pixel_values, list):
                dummy_text = [""] * len(raw_pixel_values)
                inputs = self.processor(text=dummy_text, images=raw_pixel_values, return_tensors="pt")
                clean_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            else:
                clean_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.bfloat16)

            # 2. 将我们在 [0, 1] 空间优化的噪声，除以 std 转换到模型的 normalized 空间
            normalized_noise = delta_noise_01.to(torch.bfloat16) / std_tensor.to(torch.bfloat16)
            
            # 3. 把噪声加在原图上 (直接加在 Tensor 层面，无视任何数据格式问题！)
            adv_pixel_values = clean_pixel_values.clone()
            adv_pixel_values[:, :3, :, :] = adv_pixel_values[:, :3, :, :] + normalized_noise

            # 4. 提取对抗特征
            adv_vision_out = self.vla.vision_backbone(adv_pixel_values)
            z_adv = F.normalize(self.vla.projector(adv_vision_out).mean(dim=1), p=2, dim=-1)
            
            # 5. 极其纯粹的特征碰撞 Loss
            z_target_batch = z_target.expand(z_adv.shape[0], -1)
            L_align = 1.0 - F.cosine_similarity(z_adv, z_target_batch).mean()

            L_align.backward()
            optimizer.step()
            
            # 🔴 上限测试：允许极大的颜色改变 (EPSILON = 0.5 即允许 50% 的像素剧变)
            EPSILON = 0.5
            with torch.no_grad():
                delta_noise_01.data = delta_noise_01.data.clamp(-EPSILON, EPSILON)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())
            
            # 保持这些变量存在，防止你原有的 plot_loss 等函数报错
            self.train_L_align = self.train_J_loss 
            self.train_L_camou = [0] * len(self.train_J_loss)
            self.train_L_bg = [0] * len(self.train_J_loss)

            if i % 100 == 0:
                self.plot_loss()
                
                temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                os.makedirs(temp_save_dir, exist_ok=True)
                # 保存这个极端的全域噪声外挂
                torch.save(delta_noise_01.detach().cpu(), os.path.join(temp_save_dir, "full_noise.pt"))
                
                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                os.makedirs(val_related_file_path, exist_ok=True)
                
                # 记录加噪后的毁天灭地的效果
                with torch.no_grad():
                    clean_primary = clean_pixel_values[:, :3, :, :].float()
                    denorm_clean = clean_primary * std_tensor + mean_tensor
                    adv_images_01 = torch.clamp(denorm_clean + delta_noise_01.float(), 0.0, 1.0)
                    
                    for o in range(adv_images_01.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(adv_images_01[o].cpu())
                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                
                if i > 0:
                    recent_avg_loss = np.mean(self.train_J_loss[-100:])
                    print(f"\n📊 [Iter {i:04d}] 暴力注入误差 (L_align): {recent_avg_loss:.4f}")

                    if recent_avg_loss < self.best_moving_avg_loss: 
                        print(f"   🌟 破防深度加深！视觉特征偏差降至 {recent_avg_loss:.4f}")
                        self.best_moving_avg_loss = recent_avg_loss
                        
                        last_save_dir = os.path.join(self.save_dir, "last")
                        os.makedirs(last_save_dir, exist_ok=True)
                        torch.save(delta_noise_01.detach().cpu(), os.path.join(last_save_dir, "full_noise.pt"))
                        
                        last_img_dir = os.path.join(last_save_dir, "val_related_data")
                        os.makedirs(last_img_dir, exist_ok=True)
                        for o in range(adv_images_01.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(adv_images_01[o].cpu())
                            pil_img.save(os.path.join(last_img_dir, f"{str(o)}.png"))
                            
                self.save_info(path=self.save_dir)
    '''

    '''
    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=1000, 
                            lr=0.05, accumulate_steps=1, warmup=20, **kwargs):
        
        import glob
        import random
        
        print("🔍 [阶段 1] 正在加载精心挑选的【第 0 帧恶意靶点图片】...")
        target_image_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png"
        
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"找不到靶点图片：{target_image_path}，请先运行首帧提取脚本！")
            
        target_image = Image.open(target_image_path).convert("RGB")
        
        with torch.no_grad():
            inputs = self.processor(text=[""], images=[target_image], return_tensors="pt")
            target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            
            target_vision_out = self.vla.vision_backbone(target_pixel_values)
            z_target = F.normalize(self.vla.projector(target_vision_out).mean(dim=1), p=2, dim=-1)

        print("🎯 第 0 帧视觉靶点已锁定！")

        # ====================================================================
        # 🔴 核心修复：彻底抛弃旧的动态轨迹 Dataloader，自建【首帧样本池】
        # ====================================================================
        print("🔍 [阶段 1.5] 正在从 target_frames 文件夹构建【纯净首帧加载池】...")
        all_frame_paths = glob.glob("/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/*.png")
        
        if not all_frame_paths:
            raise FileNotFoundError("⚠️ target_frames 文件夹为空！请先执行首帧提取脚本。")
        print(f"✅ 成功读取 {len(all_frame_paths)} 张各任务真实的初始状态图！")

        # 默认 Batch Size (模拟原 DataLoader 的吞吐量)
        batch_size = 10
        
        # 初始化全域 RGB 噪声
        delta_noise_01 = torch.zeros((1, 3, 224, 224), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.Adam([delta_noise_01], lr=lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        std_tensor = self.std[0].view(1, 3, 1, 1).to(self.vla.device)
        mean_tensor = self.mean[0].view(1, 3, 1, 1).to(self.vla.device)

        print(f"🚀 [阶段 2] 开始绝对静态的【首帧对首帧】全域特征碰撞！")

        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            
            # ====================================================================
            # 🔴 现场攒 Batch：从我们的首帧池里随机抽 8 张不同的初始桌面
            # ====================================================================
            sampled_paths = random.choices(all_frame_paths, k=batch_size)
            images = [Image.open(p).convert("RGB") for p in sampled_paths]
            
            with torch.no_grad():
                # 送入 processor 拿到干净的标准化 Tensor
                inputs = self.processor(text=[""] * batch_size, images=images, return_tensors="pt")
                clean_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)

            # 将 [0, 1] 噪声转换到模型的 normalized 空间
            normalized_noise = delta_noise_01.to(torch.bfloat16) / std_tensor.to(torch.bfloat16)
            
            # 把噪声强行盖在干净的首帧上
            adv_pixel_values = clean_pixel_values.clone()
            adv_pixel_values[:, :3, :, :] = adv_pixel_values[:, :3, :, :] + normalized_noise

            # 提取对抗特征
            adv_vision_out = self.vla.vision_backbone(adv_pixel_values)
            z_adv = F.normalize(self.vla.projector(adv_vision_out).mean(dim=1), p=2, dim=-1)
            
            # 计算纯粹的潜空间对齐 Loss
            z_target_batch = z_target.expand(z_adv.shape[0], -1)
            L_align = 1.0 - F.cosine_similarity(z_adv, z_target_batch).mean()

            L_align.backward()
            optimizer.step()
            
            # 允许极大的颜色改变 (EPSILON = 0.5)
            EPSILON = 1.0
            with torch.no_grad():
                delta_noise_01.data = delta_noise_01.data.clamp(-EPSILON, EPSILON)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())
            self.train_L_align = self.train_J_loss 
            self.train_L_camou = [0] * len(self.train_J_loss)
            self.train_L_bg = [0] * len(self.train_J_loss)

            if i % 100 == 0:
                self.plot_loss()
                
                temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                os.makedirs(temp_save_dir, exist_ok=True)
                torch.save(delta_noise_01.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                
                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                os.makedirs(val_related_file_path, exist_ok=True)
                
                # 记录渲染后的破坏效果
                with torch.no_grad():
                    clean_primary = clean_pixel_values[:, :3, :, :].float()
                    denorm_clean = clean_primary * std_tensor + mean_tensor
                    adv_images_01 = torch.clamp(denorm_clean + delta_noise_01.float(), 0.0, 1.0)
                    
                    for o in range(adv_images_01.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(adv_images_01[o].cpu())
                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                
                if i > 0:
                    recent_avg_loss = np.mean(self.train_J_loss[-100:])
                    print(f"\n📊 [Iter {i:04d}] 首帧空间劫持误差 (L_align): {recent_avg_loss:.4f}")

                    if recent_avg_loss < self.best_moving_avg_loss: 
                        print(f"   🌟 空间诱导深度加深！Loss 降至 {recent_avg_loss:.4f}")
                        self.best_moving_avg_loss = recent_avg_loss
                        
                        last_save_dir = os.path.join(self.save_dir, "last")
                        os.makedirs(last_save_dir, exist_ok=True)
                        torch.save(delta_noise_01.detach().cpu(), os.path.join(last_save_dir, "full_noise.pt"))
                        
                        last_img_dir = os.path.join(last_save_dir, "val_related_data")
                        os.makedirs(last_img_dir, exist_ok=True)
                        for o in range(adv_images_01.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(adv_images_01[o].cpu())
                            pil_img.save(os.path.join(last_img_dir, f"{str(o)}.png"))
                            
                self.save_info(path=self.save_dir)
        
        # ====================================================================
        # 🔴 造物主模式：不再是加噪声，而是直接把这 224x224 个像素作为变量画出来！
        # 初始为全随机的纯粹雪花点 [0.0 ~ 1.0]
        # ====================================================================
        adv_pixels_01 = torch.rand((1, 3, 224, 224), device=self.vla.device, requires_grad=True)

        # 学习率开大，让它疯狂作画
        optimizer = torch.optim.Adam([adv_pixels_01], lr=0.1) 
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        std_tensor = self.std[0].view(1, 3, 1, 1).to(self.vla.device)
        mean_tensor = self.mean[0].view(1, 3, 1, 1).to(self.vla.device)

        print(f"🚀 [阶段 2] 开启【造物主模式】：白板作画生成极致幻觉马赛克！")

        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            
            # 1. 直接截断在合法的像素空间 [0, 1]
            valid_adv_pixels_01 = torch.clamp(adv_pixels_01, 0.0, 1.0)
            
            # 2. 标准化后喂给大模型
            adv_pixel_values = (valid_adv_pixels_01 - mean_tensor) / std_tensor

            # 3. 提取对抗特征
            adv_vision_out = self.vla.vision_backbone(adv_pixel_values.to(torch.bfloat16))
            z_adv = F.normalize(self.vla.projector(adv_vision_out).mean(dim=1), p=2, dim=-1)
            
            # 4. 强制对齐到靶点特征 (z_target 是你在阶段 1 提取的首帧靶点)
            L_align = 1.0 - F.cosine_similarity(z_adv, z_target).mean()

            L_align.backward()
            optimizer.step()
            
            # 🔴 直接在数值本身做裁剪，确保它永远是一张可以展示的图片
            with torch.no_grad():
                adv_pixels_01.data = adv_pixels_01.data.clamp(0.0, 1.0)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())

            if i % 100 == 0:
                # 画图和日志逻辑...
                temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                os.makedirs(temp_save_dir, exist_ok=True)
                
                # 🔴 保存这张“画”出来的终极幻觉图！
                # 注意：此时保存的就已经是绝对覆盖画面的图片了，拿去测试的时候模式一会完美读取
                torch.save(adv_pixels_01.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                
                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                os.makedirs(val_related_file_path, exist_ok=True)
                
                with torch.no_grad():
                    pil_img = torchvision.transforms.ToPILImage()(adv_pixels_01[0].cpu())
                    pil_img.save(os.path.join(val_related_file_path, f"god_mode_illusion.png"))
                
                if i > 0:
                    recent_avg_loss = np.mean(self.train_J_loss[-100:])
                    print(f"\n📊 [Iter {i:04d}] 造物幻觉误差 (L_align): {recent_avg_loss:.4f}")
    '''

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=1000, 
                            lr=0.05, accumulate_steps=1, warmup=20, **kwargs):
        
        import glob
        import random
        
        print("🔍 [阶段 1] 正在加载精心挑选的【第 0 帧恶意靶点图片】...")
        target_image_path = "/root/autodl-tmp/roboticAttack/run/GreyBox/20260308_122225/target_frames/target_scene.png"
        
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"找不到靶点图片：{target_image_path}，请先运行首帧提取脚本！")
            
        target_image = Image.open(target_image_path).convert("RGB")
        
        with torch.no_grad():
            inputs = self.processor(text=[""], images=[target_image], return_tensors="pt")
            # 官方 processor 默认输出的正是 6 通道张量
            target_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
            
            target_vision_out = self.vla.vision_backbone(target_pixel_values)
            z_target = F.normalize(self.vla.projector(target_vision_out).mean(dim=1), p=2, dim=-1)

        print("🎯 第 0 帧视觉靶点已锁定！")

        print("🔍 [阶段 1.5] 检查背景池状态...")
        all_frame_paths = glob.glob("target_frames/*.png")
        if all_frame_paths:
            print(f"✅ 成功读取 {len(all_frame_paths)} 张背景图（注：造物主模式下直接重写画布，背景不参与运算）")

        # ====================================================================
        # 🔴 造物主模式：完全白板作画，直接优化一张 224x224 的 3 通道全域图片
        # ====================================================================
        adv_pixels_01 = torch.rand((1, 3, 224, 224), device=self.vla.device, requires_grad=True)

        optimizer = torch.optim.Adam([adv_pixels_01], lr=lr) 
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup, 
            num_training_steps=int(num_iter / accumulate_steps), num_cycles=0.5, last_epoch=-1
        )

        # ====================================================================
        # 🔴 关键修复：提取两套不同的均值和方差，为 DINO 和 SigLIP 准备
        # ====================================================================
        mean0 = self.mean[0].view(1, 3, 1, 1).to(self.vla.device) # SigLIP 均值
        std0 = self.std[0].view(1, 3, 1, 1).to(self.vla.device)   # SigLIP 方差
        mean1 = self.mean[1].view(1, 3, 1, 1).to(self.vla.device) # DINO 均值
        std1 = self.std[1].view(1, 3, 1, 1).to(self.vla.device)   # DINO 方差

        print(f"🚀 [阶段 2] 开启【造物主模式】：白板作画，彻底降维打击！")

        for i in tqdm(range(num_iter)):
            optimizer.zero_grad()
            
            # 1. 确保画布在合法色彩空间 [0, 1]
            valid_adv_pixels_01 = torch.clamp(adv_pixels_01, 0.0, 1.0)
            
            # 2. 🔴【最核心修复】将一张图片分成两路，分别按不同标准归一化！
            im0 = (valid_adv_pixels_01 - mean0) / std0  # 给 SigLIP 吃的 3 通道
            im1 = (valid_adv_pixels_01 - mean1) / std1  # 给 DINO 吃的 3 通道
            
            # 3. 沿通道维度拼接，组装成 OpenVLA 极度渴望的 6 通道 [1, 6, 224, 224] 张量！
            adv_pixel_values_6c = torch.cat([im0, im1], dim=1) 

            # 4. 提取特征（绝对不会再报 split 错误了！）
            adv_vision_out = self.vla.vision_backbone(adv_pixel_values_6c.to(torch.bfloat16))
            z_adv = F.normalize(self.vla.projector(adv_vision_out).mean(dim=1), p=2, dim=-1)
            
            # 5. 强制认知对齐
            z_target_batch = z_target.expand(z_adv.shape[0], -1)
            L_align = 1.0 - F.cosine_similarity(z_adv, z_target_batch).mean()

            L_align.backward()
            optimizer.step()
            
            # 确保自身像素有效性
            with torch.no_grad():
                adv_pixels_01.data = adv_pixels_01.data.clamp(0.0, 1.0)

            if (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(L_align.item())
            self.train_L_align = self.train_J_loss 
            self.train_L_camou = [0] * len(self.train_J_loss)
            self.train_L_bg = [0] * len(self.train_J_loss)

            if i % 100 == 0:
                self.plot_loss()
                
                temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                os.makedirs(temp_save_dir, exist_ok=True)
                
                # 记录这块全域的幻觉画布
                torch.save(adv_pixels_01.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                
                val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                os.makedirs(val_related_file_path, exist_ok=True)
                
                with torch.no_grad():
                    pil_img = torchvision.transforms.ToPILImage()(adv_pixels_01[0].cpu())
                    pil_img.save(os.path.join(val_related_file_path, f"god_mode_illusion.png"))
                
                if i > 0:
                    recent_avg_loss = np.mean(self.train_J_loss[-100:])
                    print(f"\n📊 [Iter {i:04d}] 造物幻觉误差 (L_align): {recent_avg_loss:.4f}")

                    if recent_avg_loss < self.best_moving_avg_loss: 
                        print(f"   🌟 幻觉植入加深！Loss 降至 {recent_avg_loss:.4f}")
                        self.best_moving_avg_loss = recent_avg_loss
                        
                        last_save_dir = os.path.join(self.save_dir, "last")
                        os.makedirs(last_save_dir, exist_ok=True)
                        torch.save(adv_pixels_01.detach().cpu(), os.path.join(last_save_dir, "patch.pt"))
                        
                        last_img_dir = os.path.join(last_save_dir, "val_related_data")
                        os.makedirs(last_img_dir, exist_ok=True)
                        pil_img.save(os.path.join(last_img_dir, f"god_mode_illusion.png"))
                            
                self.save_info(path=self.save_dir)

        