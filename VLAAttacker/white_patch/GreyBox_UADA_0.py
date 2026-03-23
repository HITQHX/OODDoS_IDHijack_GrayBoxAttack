import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from white_patch.appply_random_transform import RandomPatchTransform
from tqdm import tqdm
import os
import transformers
import pickle


def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images, mean, std):
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

class GreyBoxOpenVLAAttacker(object):
    
    def __init__(self, vla, processor, save_dir="", optimizer="adamW", resize_patch=False):
        self.vla = vla.eval()
        
        # 严格遵守灰盒假设：冻结 LLM 部分，显存开销将暴降
        self.vla.requires_grad_(False)

        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        self.base_tokenizer = self.processor.tokenizer
        
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.randomPatchTransform = RandomPatchTransform(self.vla.device, resize_patch)
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        
        # --- 本地数据记录 Buffer ---
        self.train_J_loss = []
        self.train_L_feat = []
        self.train_L_align = []
        self.train_L_camou = []
        self.val_L_align = []
        self.best_val_align_loss = float('inf')


    def plot_loss(self):
        sns.set_theme()
        x_ticks = list(range(0, len(self.train_J_loss)))

        plt.figure(figsize=(10, 6))
        plt.plot(x_ticks, self.train_J_loss, label='Total J Loss')
        plt.plot(x_ticks, self.train_L_align, label='Semantic Align Loss')
        plt.title('Grey-Box Attack Loss Curve')
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.clf()

    def save_info(self, path):
        with open(os.path.join(path, 'train_J_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_J_loss, file)
        with open(os.path.join(path, 'train_L_feat.pkl'), 'wb') as file:
            pickle.dump(self.train_L_feat, file)
        with open(os.path.join(path, 'train_L_align.pkl'), 'wb') as file:
            pickle.dump(self.train_L_align, file)
        with open(os.path.join(path, 'train_L_camou.pkl'), 'wb') as file:
            pickle.dump(self.train_L_camou, file)
        with open(os.path.join(path, 'val_L_align.pkl'), 'wb') as file:
            pickle.dump(self.val_L_align, file)

    def get_semantic_anchor(self, malicious_text):
        tokens = self.processor.tokenizer(malicious_text, return_tensors="pt").input_ids.to(self.vla.device)
        with torch.no_grad():
            embeddings = self.vla.get_input_embeddings()(tokens)
            z_mal = embeddings.mean(dim=1).squeeze(0)
            z_mal = F.normalize(z_mal, p=2, dim=-1)
        return z_mal

    def compute_tv_loss(self, patch):
        tv_h = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).sum()
        tv_w = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).sum()
        return (tv_h + tv_w) / (patch.shape[1] * patch.shape[2])

    def patchattack_graybox(self, train_dataloader, val_dataloader, num_iter=5000, 
                            patch_size=[3, 50, 50], lr=1/255, accumulate_steps=1, 
                            malicious_text="swing the knife and destroy the object",
                            alpha=1.0, beta=1.0, gamma=0.1, tau=0.5,
                            geometry=True, innerLoop=1, maskidx=None, warmup=20):
        
        patch = torch.rand(patch_size).to(self.vla.device)
        patch.requires_grad_(True)
        patch.retain_grad()

        if self.optimizer == "adamW":
            optimizer = transformers.AdamW([patch], lr=lr)
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=int(num_iter / accumulate_steps),
                num_cycles=0.5,
                last_epoch=-1,
            )

        z_mal = self.get_semantic_anchor(malicious_text)

        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        previous_adv_features = None

        print(f"🚀 [Grey-Box] 开始针对语义 '{malicious_text}' 的越狱优化...")

        for i in tqdm(range(num_iter)):
            try:
                data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                data = next(train_iterator)

            # 🔴 关键修复：此时 raw_pixel_values 是一个 PIL.Image 的列表
            raw_pixel_values = data["pixel_values"]
            
            # === 获取干净特征 z_clean ===
            with torch.no_grad():
                # 如果是列表，使用官方 Processor 转换；如果是张量则直接用
                if isinstance(raw_pixel_values, list):
                    dummy_text = [""] * len(raw_pixel_values) # Processor 必须有文本输入
                    inputs = self.processor(text=dummy_text, images=raw_pixel_values, return_tensors="pt")
                    clean_pixel_values = inputs["pixel_values"].to(self.vla.device, dtype=torch.bfloat16)
                else:
                    clean_pixel_values = raw_pixel_values.to(self.vla.device, dtype=torch.bfloat16)

                clean_vision_out = self.vla.vision_backbone(clean_pixel_values)
                clean_features = self.vla.projector(clean_vision_out).mean(dim=1)
                z_clean = F.normalize(clean_features, p=2, dim=-1)

            optimizer.zero_grad()
            
            for inner_loop in range(innerLoop):
                # 依然将原始的 List[PIL.Image] 传给原作者的打补丁函数
                modified_images = self.randomPatchTransform.apply_random_patch_batch(
                    raw_pixel_values, patch, mean=self.mean, std=self.std, geometry=geometry)

                adv_vision_out = self.vla.vision_backbone(modified_images.to(torch.bfloat16))
                adv_features = self.vla.projector(adv_vision_out).mean(dim=1)
                z_adv = F.normalize(adv_features, p=2, dim=-1)

                L_feat = F.cosine_similarity(z_adv, z_clean).mean()
                z_mal_batch = z_mal.unsqueeze(0).expand(z_adv.shape[0], -1)
                L_align = 1.0 - F.cosine_similarity(z_adv, z_mal_batch).mean()
                L_TV = self.compute_tv_loss(patch)
                
                if previous_adv_features is not None:
                    L_temporal = F.mse_loss(adv_features, previous_adv_features.detach())
                else:
                    L_temporal = torch.tensor(0.0).to(self.vla.device)

                L_camou = L_TV + tau * L_temporal
                J_loss = alpha * L_feat + beta * L_align + gamma * L_camou

                (J_loss / innerLoop).backward()
                previous_adv_features = adv_features.detach()

            optimizer.step()
            patch.data = patch.data.clamp(0, 1)

            if self.optimizer == "adamW" and (i + 1) % accumulate_steps == 0:
                scheduler.step()

            self.train_J_loss.append(J_loss.item())
            self.train_L_feat.append(L_feat.item())
            self.train_L_align.append(L_align.item())
            self.train_L_camou.append(L_camou.item())

            if i % 100 == 0:
                self.plot_loss()
                print("\nEvaluating on Validation Set...")
                avg_val_align = 0
                val_num_sample = 0
                
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for j in range(10): 
                        try:
                            v_data = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            v_data = next(val_iterator)
                            
                        # 🔴 关键修复：验证集直接传列表，不需要做预处理，打补丁函数内部会处理
                        v_raw_pixel_values = v_data["pixel_values"]
                        
                        v_modified = self.randomPatchTransform.apply_random_patch_batch(
                            v_raw_pixel_values, patch, mean=self.mean, std=self.std, geometry=geometry)
                        
                        v_adv_out = self.vla.vision_backbone(v_modified.to(torch.bfloat16))
                        v_adv_feat = F.normalize(self.vla.projector(v_adv_out).mean(dim=1), p=2, dim=-1)
                        
                        v_align = 1.0 - F.cosine_similarity(v_adv_feat, z_mal.unsqueeze(0).expand(v_adv_feat.shape[0], -1)).mean()
                        avg_val_align += v_align.item()
                        val_num_sample += 1
                        
                avg_val_align /= val_num_sample
                self.val_L_align.append(avg_val_align)
                print(f"Iter: {i}, Train J_Loss: {J_loss.item():.4f}, Val Align Loss: {avg_val_align:.4f}")

                if avg_val_align < self.best_val_align_loss:
                    self.best_val_align_loss = avg_val_align
                    
                    temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    
                    # 🔴 注意：由于 OpenVLA 是 6 通道输入(SigLIP + DINOv2)，这里我们只取前3个通道保存为可视化图片
                    denorm_images = self.randomPatchTransform.denormalize(
                        v_modified[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                        
                    for o in range(denorm_images.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                        
                    last_save_dir = os.path.join(self.save_dir, "last")
                    os.makedirs(last_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(last_save_dir, "patch.pt"))
                    last_img_dir = os.path.join(last_save_dir, "val_related_data")
                    os.makedirs(last_img_dir, exist_ok=True)
                    for o in range(denorm_images.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(denorm_images[o, :, :, :])
                        pil_img.save(os.path.join(last_img_dir, f"{str(o)}.png"))
                        
                self.save_info(path=self.save_dir)
                torch.cuda.empty_cache()