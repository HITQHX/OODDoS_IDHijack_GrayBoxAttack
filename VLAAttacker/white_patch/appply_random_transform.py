'''
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

class RandomPatchTransform:
    def __init__(self, device, resize_patch):
        self.device = device
        self.resize_patch = resize_patch

    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self, images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images

    def apply_stealth_perspective_batch(self, images, patch, mean, std, safe_zone):
        modified_images = []
        l_bg_total = torch.tensor(0.0).to(self.device)
        
        c, h, w = patch.shape
        top_shrink = int(w * 0.15)
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        endpoints = [[top_shrink, 0], [w - top_shrink, 0], [w, h], [0, h]]
        
        patch_persp = TF.perspective(patch, startpoints, endpoints)
        mask = TF.perspective(torch.ones_like(patch), startpoints, endpoints) > 0.5
        
        xmin, ymin, xmax, ymax = safe_zone

        for im in images:
            if isinstance(im, Image.Image) or isinstance(im, np.ndarray):
                im = torchvision.transforms.ToTensor()(im).to(self.device)
            else:
                im = im.clone().to(self.device)
                
            img_c, img_h, img_w = im.shape

            # 🔴 绝对鲁棒的自适应边界：保证无论分辨率如何变化，补丁都 100% 能塞进去
            safe_xmin = max(0, min(xmin, img_w - w))
            safe_ymin = max(0, min(ymin, img_h - h))
            safe_xmax = min(xmax, img_w)
            safe_ymax = min(ymax, img_h)
            
            max_x_start = max(safe_xmin, safe_xmax - w)
            max_y_start = max(safe_ymin, safe_ymax - h)

            x = random.randint(safe_xmin, max_x_start)
            y = random.randint(safe_ymin, max_y_start)

            bg_crop = im[:, y:y+h, x:x+w]
            
            l_bg_total += F.mse_loss(patch_persp[mask], bg_crop[mask])

            im[:, y:y+h, x:x+w] = torch.where(mask, patch_persp, bg_crop)

            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            
            modified_images.append(torch.cat([im0, im1], dim=1))

        return torch.cat(modified_images, dim=0), l_bg_total / len(images)

    def simulation_table_patch_single(self, image, patch, safe_zone, fixed_xy=None):
        """用于 run_libero_eval_args_geo_batch.py 中的单张图片处理"""
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.permute(1,2,0).cpu().numpy() if image.shape[0] == 3 else image.cpu().numpy()
        else:
            image_np = image.copy()
            
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)

        c, h, w = patch.shape
        patch_255 = torch.from_numpy(np.asarray(torchvision.transforms.ToPILImage()(patch.cpu())).copy()).permute(2, 0, 1).to(self.device)
        
        top_shrink = int(w * 0.15)
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        endpoints = [[top_shrink, 0], [w - top_shrink, 0], [w, h], [0, h]]
        
        patch_persp = TF.perspective(patch_255, startpoints, endpoints)
        mask = TF.perspective(torch.ones_like(patch_255), startpoints, endpoints) > 0.5
        
        xmin, ymin, xmax, ymax = safe_zone
        img_c, img_h, img_w = img_tensor.shape
        
        safe_xmin = max(0, min(xmin, img_w - w))
        safe_ymin = max(0, min(ymin, img_h - h))
        safe_xmax = min(xmax, img_w)
        safe_ymax = min(ymax, img_h)
        max_x_start = max(safe_xmin, safe_xmax - w)
        max_y_start = max(safe_ymin, safe_ymax - h)
        
        # 🔴 核心修复：如果未传入坐标，就随机生成一个；如果传入了，就死死锁住它！
        if fixed_xy is None:
            x = random.randint(safe_xmin, max_x_start)
            y = random.randint(safe_ymin, max_y_start)
        else:
            x, y = fixed_xy
            # 同样做一下越界保护
            x = max(safe_xmin, min(x, max_x_start))
            y = max(safe_ymin, min(y, max_y_start))
        
        bg_crop = img_tensor[:, y:y+h, x:x+w]
        img_tensor[:, y:y+h, x:x+w] = torch.where(mask, patch_persp.to(img_tensor.dtype), bg_crop)
        
        # 🔴 返回渲染好的图像，并且把本次决定的坐标 (x, y) 也返回出去
        return img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8), (x, y)

    # ========================== 兼容旧接口 ==========================
    def apply_random_patch_batch(self, images, patch, mean, std, geometry=False):
        modified_images = []
        for im in images:
            if not isinstance(im, torch.Tensor):
                im = torchvision.transforms.ToTensor()(im).to(self.device)
            else:
                im = im.clone().to(self.device)
            img_c, img_h, img_w = im.shape
            p_c, p_h, p_w = patch.shape
            x = random.randint(0, max(0, img_w - p_w))
            y = random.randint(0, max(0, img_h - p_h))
            im[:, y:y+p_h, x:x+p_w] = patch
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0, im1], dim=1))
        return torch.cat(modified_images, dim=0)

    def simulation_random_patch(self, image, patch, geometry=False, colorjitter=False, angle=0, shx=0, shy=0, position=(0,0)):
        # 确保旧接口调用时不报错，只取第一个返回值(图像)
        img, _ = self.simulation_table_patch_single(image, patch, safe_zone=[position[0], position[1], position[0]+50, position[1]+50])
        return img


import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

class RandomPatchTransform:
    def __init__(self, device, resize_patch):
        self.device = device
        self.resize_patch = resize_patch

    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self, images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images

    def get_geometric_endpoints(self, x, y, w, h, img_w, img_h):
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

    def apply_stealth_perspective_batch(self, images, delta_alpha, delta_beta, mean, std, safe_zone):
        modified_images = []
        c, h, w = delta_alpha.shape # 自动适应任何传入的 h 和 w (比如 45x45)
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        xmin, ymin, xmax, ymax = safe_zone

        for im in images:
            if isinstance(im, Image.Image) or isinstance(im, np.ndarray):
                im_base = torchvision.transforms.ToTensor()(im).to(self.device)
            else:
                im_base = im.clone().to(self.device)
                
            img_c, img_h, img_w = im_base.shape

            safe_xmin = max(0, min(xmin, img_w - w))
            safe_ymin = max(0, min(ymin, img_h - h))
            safe_xmax = min(xmax, img_w)
            safe_ymax = min(ymax, img_h)
            max_x_start = max(safe_xmin, safe_xmax - w)
            max_y_start = max(safe_ymin, safe_ymax - h)

            x = random.randint(safe_xmin, max_x_start)
            y = random.randint(safe_ymin, max_y_start)

            endpoints = self.get_geometric_endpoints(x, y, w, h, img_w, img_h)
            
            alpha_persp = TF.perspective(delta_alpha, startpoints, endpoints)
            beta_persp  = TF.perspective(delta_beta, startpoints, endpoints)
            mask = TF.perspective(torch.ones_like(delta_alpha), startpoints, endpoints) > 0.5

            # =========================================================================
            # 🔴 核心修复：分离计算图，绝对禁止原位修改！
            # 1. 独立克隆背景，为 backward() 保护原始数据
            # =========================================================================
            bg_crop = im_base[:, y:y+h, x:x+w].clone() 
            
            new_crop = torch.clamp(bg_crop * (1.0 + alpha_persp) + beta_persp, 0, 1)

            # 2. 新建一张图进行像素覆盖，绝不触碰前面的张量
            im_modified = im_base.clone()
            im_modified[:, y:y+h, x:x+w] = torch.where(mask, new_crop, bg_crop)

            im0 = self.normalize(im_modified, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im_modified, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0, im1], dim=1))

        return torch.cat(modified_images, dim=0), torch.tensor(0.0).to(self.device)

    def simulation_table_patch_single(self, image, patch, safe_zone, fixed_xy=None):
        delta_alpha = patch[0:1, :, :].to(self.device)
        delta_beta  = patch[1:2, :, :].to(self.device)
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.permute(1,2,0).cpu().numpy() if image.shape[0] == 3 else image.cpu().numpy()
        else:
            image_np = image.copy()
            
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device) 
        _, h, w = delta_alpha.shape
        
        xmin, ymin, xmax, ymax = safe_zone
        img_c, img_h, img_w = img_tensor.shape
        
        safe_xmin = max(0, min(xmin, img_w - w))
        safe_ymin = max(0, min(ymin, img_h - h))
        safe_xmax = min(xmax, img_w)
        safe_ymax = min(ymax, img_h)
        max_x_start = max(safe_xmin, safe_xmax - w)
        max_y_start = max(safe_ymin, safe_ymax - h)
        
        if fixed_xy is None:
            x = random.randint(safe_xmin, max_x_start)
            y = random.randint(safe_ymin, max_y_start)
        else:
            x, y = fixed_xy
            x = max(safe_xmin, min(x, max_x_start))
            y = max(safe_ymin, min(y, max_y_start))
            
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        endpoints = self.get_geometric_endpoints(x, y, w, h, img_w, img_h)
        
        alpha_persp = TF.perspective(delta_alpha, startpoints, endpoints)
        beta_persp = TF.perspective(delta_beta, startpoints, endpoints)
        mask = TF.perspective(torch.ones_like(delta_alpha), startpoints, endpoints) > 0.5
        
        bg_crop = img_tensor[:, y:y+h, x:x+w].clone()
        
        bg_crop_float = bg_crop.float() / 255.0
        new_crop_float = torch.clamp(bg_crop_float * (1.0 + alpha_persp) + beta_persp, 0, 1)
        new_crop = (new_crop_float * 255.0).to(img_tensor.dtype)
        
        img_tensor[:, y:y+h, x:x+w] = torch.where(mask, new_crop, bg_crop)
        
        return img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8), (x, y)

    def apply_random_patch_batch(self, images, patch, mean, std, geometry=False):
        modified_images = []
        for im in images:
            if not isinstance(im, torch.Tensor):
                im = torchvision.transforms.ToTensor()(im).to(self.device)
            else:
                im = im.clone().to(self.device)
            img_c, img_h, img_w = im.shape
            p_c, p_h, p_w = patch.shape
            x = random.randint(0, max(0, img_w - p_w))
            y = random.randint(0, max(0, img_h - p_h))
            im[:, y:y+p_h, x:x+p_w] = patch
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0, im1], dim=1))
        return torch.cat(modified_images, dim=0)

    def simulation_random_patch(self, image, patch, geometry=False, colorjitter=False, angle=0, shx=0, shy=0, position=(0,0)):
        img, _ = self.simulation_table_patch_single(image, patch, safe_zone=[position[0], position[1], position[0]+50, position[1]+50])
        return img
'''

import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

class RandomPatchTransform:
    def __init__(self, device, resize_patch):
        self.device = device
        self.resize_patch = resize_patch

    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self, images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images

    def get_geometric_endpoints(self, x, y, w, h, img_w, img_h):
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

    def apply_stealth_perspective_batch(self, images, delta_alpha, delta_beta, mean, std, safe_zone):
        # 此函数为训练时光影滤镜使用，保持不变
        modified_images = []
        c, h, w = delta_alpha.shape 
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        xmin, ymin, xmax, ymax = safe_zone

        for im in images:
            if isinstance(im, Image.Image) or isinstance(im, np.ndarray):
                im_base = torchvision.transforms.ToTensor()(im).to(self.device)
            else:
                im_base = im.clone().to(self.device)
                
            img_c, img_h, img_w = im_base.shape

            safe_xmin = max(0, min(xmin, img_w - w))
            safe_ymin = max(0, min(ymin, img_h - h))
            safe_xmax = min(xmax, img_w)
            safe_ymax = min(ymax, img_h)
            max_x_start = max(safe_xmin, safe_xmax - w)
            max_y_start = max(safe_ymin, safe_ymax - h)

            x = random.randint(safe_xmin, max_x_start)
            y = random.randint(safe_ymin, max_y_start)

            endpoints = self.get_geometric_endpoints(x, y, w, h, img_w, img_h)
            
            alpha_persp = TF.perspective(delta_alpha, startpoints, endpoints)
            beta_persp  = TF.perspective(delta_beta, startpoints, endpoints)
            mask = TF.perspective(torch.ones_like(delta_alpha), startpoints, endpoints) > 0.5

            bg_crop = im_base[:, y:y+h, x:x+w].clone() 
            new_crop = torch.clamp(bg_crop * (1.0 + alpha_persp) + beta_persp, 0, 1)

            im_modified = im_base.clone()
            im_modified[:, y:y+h, x:x+w] = torch.where(mask, new_crop, bg_crop)

            im0 = self.normalize(im_modified, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im_modified, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0, im1], dim=1))

        return torch.cat(modified_images, dim=0), torch.tensor(0.0).to(self.device)

    def simulation_table_patch_single(self, image, patch, safe_zone, fixed_xy=None):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.permute(1,2,0).cpu().numpy() if image.shape[0] == 3 else image.cpu().numpy()
        else:
            image_np = image.copy()
            
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device) 
        
        # =========================================================================
        # 🔴 智能识别弹药类型：彻底解决张量维度解包崩溃的问题
        # =========================================================================
        
        # 🚀 模式一：如果收到的是全域暴力噪声上限测试 (Upper Bound [1, 3, 224, 224])
        '''
        if (patch.dim() == 4 and patch.shape[1] == 3) or (patch.dim() == 3 and patch.shape[0] == 3 and patch.shape[1] > 100):
            noise = patch.squeeze(0).to(self.device) if patch.dim() == 4 else patch.to(self.device)
            
            # LIBERO 可能给 256x256，但我们的 noise 是 224x224，强行插值对齐，保证100%不报错！
            if noise.shape[-1] != img_tensor.shape[-1] or noise.shape[-2] != img_tensor.shape[-2]:
                noise = F.interpolate(noise.unsqueeze(0), size=(img_tensor.shape[-2], img_tensor.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)
            
            # 直接在 0~1 空间强行叠加致盲噪声！
            img_float = img_tensor.float() / 255.0
            adv_img_float = torch.clamp(img_float + noise.float(), 0.0, 1.0)
            adv_img = (adv_img_float * 255.0).to(img_tensor.dtype)
            
            return adv_img.permute(1, 2, 0).cpu().numpy(), (0, 0)
        '''

        if patch.dim() == 4 and patch.shape[1] == 4:
            # 提取前三层像素和最后一层掩码
            p = patch[0, :3, :, :].to(self.device)
            m = patch[0, 3:, :, :].to(self.device)
            
            if p.shape[-1] != img_tensor.shape[-1] or p.shape[-2] != img_tensor.shape[-2]:
                p = F.interpolate(p.unsqueeze(0), size=(img_tensor.shape[-2], img_tensor.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)
                m = F.interpolate(m.unsqueeze(0), size=(img_tensor.shape[-2], img_tensor.shape[-1]), mode='nearest').squeeze(0)
            
            # 🔴 完美物理级替换！彻底没有叠加闪光弹了！
            img_float = img_tensor.float() / 255.0
            adv_img_float = (1.0 - m) * img_float + m * p
            adv_img = (adv_img_float * 255.0).to(img_tensor.dtype)
            
            return adv_img.permute(1, 2, 0).cpu().numpy(), (0, 0)
            
        # 🛡️ 模式二：如果是我们的变色龙光影滤镜 [2, 50, 50]
        elif patch.dim() == 3 and patch.shape[0] == 2:
            delta_alpha = patch[0:1, :, :].to(self.device)
            delta_beta  = patch[1:2, :, :].to(self.device)
            _, h, w = delta_alpha.shape
            
            xmin, ymin, xmax, ymax = safe_zone
            img_c, img_h, img_w = img_tensor.shape
            
            safe_xmin = max(0, min(xmin, img_w - w))
            safe_ymin = max(0, min(ymin, img_h - h))
            safe_xmax = min(xmax, img_w)
            safe_ymax = min(ymax, img_h)
            max_x_start = max(safe_xmin, safe_xmax - w)
            max_y_start = max(safe_ymin, safe_ymax - h)
            
            if fixed_xy is None:
                x = random.randint(safe_xmin, max_x_start)
                y = random.randint(safe_ymin, max_y_start)
            else:
                x, y = fixed_xy
                x = max(safe_xmin, min(x, max_x_start))
                y = max(safe_ymin, min(y, max_y_start))
                
            startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
            endpoints = self.get_geometric_endpoints(x, y, w, h, img_w, img_h)
            
            alpha_persp = TF.perspective(delta_alpha, startpoints, endpoints)
            beta_persp = TF.perspective(delta_beta, startpoints, endpoints)
            mask = TF.perspective(torch.ones_like(delta_alpha), startpoints, endpoints) > 0.5
            
            bg_crop = img_tensor[:, y:y+h, x:x+w].clone()
            
            bg_crop_float = bg_crop.float() / 255.0
            new_crop_float = torch.clamp(bg_crop_float * (1.0 + alpha_persp) + beta_persp, 0, 1)
            new_crop = (new_crop_float * 255.0).to(img_tensor.dtype)
            
            img_tensor[:, y:y+h, x:x+w] = torch.where(mask, new_crop, bg_crop)
            
            return img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8), (x, y)
            
        # 🎨 模式三：兼容最初版本的纯彩色实体 Patch [3, 50, 50]
        else:
            patch_tensor = patch.squeeze(0).to(self.device) if patch.dim() == 4 else patch.to(self.device)
            c, h, w = patch_tensor.shape
            
            xmin, ymin, xmax, ymax = safe_zone
            img_c, img_h, img_w = img_tensor.shape
            
            safe_xmin = max(0, min(xmin, img_w - w))
            safe_ymin = max(0, min(ymin, img_h - h))
            safe_xmax = min(xmax, img_w)
            safe_ymax = min(ymax, img_h)
            max_x_start = max(safe_xmin, safe_xmax - w)
            max_y_start = max(safe_ymin, safe_ymax - h)
            
            if fixed_xy is None:
                x = random.randint(safe_xmin, max_x_start)
                y = random.randint(safe_ymin, max_y_start)
            else:
                x, y = fixed_xy
                x = max(safe_xmin, min(x, max_x_start))
                y = max(safe_ymin, min(y, max_y_start))
                
            startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
            endpoints = self.get_geometric_endpoints(x, y, w, h, img_w, img_h)
            
            patch_float = patch_tensor.float() 
            patch_persp = TF.perspective(patch_float, startpoints, endpoints)
            mask = TF.perspective(torch.ones_like(patch_float[0:1]), startpoints, endpoints) > 0.5
            
            bg_crop = img_tensor[:, y:y+h, x:x+w].clone()
            patch_255 = (patch_persp * 255.0).to(img_tensor.dtype)
            
            img_tensor[:, y:y+h, x:x+w] = torch.where(mask, patch_255, bg_crop)
            
            return img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8), (x, y)

    def apply_random_patch_batch(self, images, patch, mean, std, geometry=False):
        pass

    def simulation_random_patch(self, image, patch, geometry=False, colorjitter=False, angle=0, shx=0, shy=0, position=(0,0)):
        pass