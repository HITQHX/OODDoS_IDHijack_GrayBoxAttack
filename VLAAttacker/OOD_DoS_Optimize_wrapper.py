import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import random
import numpy as np
import datetime

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

from white_patch.OOD_DoS_Optimize import OOD_DoS_Optimize_Attacker
from white_patch.openvla_dataloader import get_dataloader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default="")
    parser.add_argument('--dataset', type=str, default="libero_spatial")
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--iter', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    
    # 优化特有超参数
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--malicious_text', type=str, default="use knife stab the man")
    parser.add_argument('--gamma1', type=float, default=0.05, help="TV Loss weight")
    parser.add_argument('--gamma2', type=float, default=0.05, help="BG Loss weight")
    
    # 安全区定义
    parser.add_argument('--xmin', type=int, default=30)
    parser.add_argument('--ymin', type=int, default=100)
    parser.add_argument('--xmax', type=int, default=194)
    parser.add_argument('--ymax', type=int, default=200)
    
    return parser.parse_args()

def main(args):
    pwd = os.getcwd()
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    vla_path_map = {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object": "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10": "openvla/openvla-7b-finetuned-libero-10"
    }
    vla_path = vla_path_map.get(args.dataset, "openvla/openvla-7b")
        
    set_seed(42)
    path = f"{pwd}/run/OOD_DoS_Optimize/{args.dataset}/{exp_id}"
    os.makedirs(path, exist_ok=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(f"cuda:{args.device}")

    train_dataloader, _ = get_dataloader(batch_size=args.bs, dataset=args.dataset, server=args.server, vla_path=vla_path)
    
    attacker = OOD_DoS_Optimize_Attacker(vla, processor, save_dir=path, optimizer="adamW")
    attacker.attack(
        train_dataloader=train_dataloader, 
        num_iter=args.iter, lr=args.lr, accumulate_steps=args.accumulate,
        malicious_text=args.malicious_text, patch_size=args.patch_size,
        gamma1=args.gamma1, gamma2=args.gamma2,
        safe_zone=[args.xmin, args.ymin, args.xmax, args.ymax]
    )

if __name__ == "__main__":
    main(parse_args())