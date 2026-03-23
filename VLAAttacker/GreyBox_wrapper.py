import torch
import sys
sys.path.append("/root/autodl-tmp/roboticAttack")
from transformers import AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
import os
import numpy as np
import argparse
import random
import uuid
import datetime

# 导入我们写的灰盒攻击器
from white_patch.GreyBox_UADA import GreyBoxOpenVLAAttacker
from white_patch.openvla_dataloader import get_dataloader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    pwd = os.getcwd()
    # exp_id = str(uuid.uuid4())
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 【完全对齐 UADA_wrapper.py】硬编码的模型路径判断
    if  "bridge_orig" in args.dataset:
        vla_path = "openvla/openvla-7b"
    elif "libero_spatial" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
    elif "libero_object" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-object"
    elif "libero_goal" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-goal"
    elif "libero_10" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-10"
    else:
        assert False, "Invalid dataset"
        
    set_seed(42)
    print(f"exp_id:{exp_id}")
    
    # 【完全对齐 UADA_wrapper.py】统一的结果保存路径格式
    path = f"{pwd}/run/GreyBox/{exp_id}"
    os.makedirs(path, exist_ok=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    quantization_config = None
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)

    # 🔴 修复：补全 server 和 vla_path 参数
    train_dataloader, val_dataloader = get_dataloader(
        batch_size=args.bs, 
        dataset=args.dataset, 
        server=args.server, 
        vla_path=vla_path
    )
    
    # attacker = GreyBoxUpperAttacker(vla, processor, path, optimizer="adamW", resize_patch=args.resize_patch)
    attacker = GreyBoxOpenVLAAttacker(vla, processor, path, optimizer="adamW")

    attacker.patchattack_graybox(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        num_iter=args.iter,
        # patch_size=args.patch_size, 
        lr=args.lr,
        accumulate_steps=args.accumulate,
        # malicious_text=args.malicious_text,
        #beta=args.beta,
        #gamma=args.gamma,
        #geometry=args.geometry,
        #innerLoop=args.innerLoop,
        #maskidx=args.maskidx,
        warmup=args.warmup
    )

    print("GreyBox Attack done!")

def arg_parser():
    parser = argparse.ArgumentParser()
    # ================= 原有攻击的参数完全保留，保证 .sh 兼容 =================
    parser.add_argument('--maskidx',default='0', type=list_of_ints)
    parser.add_argument('--lr',default=1e-3, type=float)
    parser.add_argument('--device',default=1, type=int)
    parser.add_argument('--iter',default=2000, type=int) 
    parser.add_argument('--accumulate',default=1, type=int)
    parser.add_argument('--bs',default=8, type=int)
    parser.add_argument('--warmup',default=20, type=int)
    parser.add_argument('--tags',nargs='+', default=[""])
    parser.add_argument('--filterGripTrainTo1', type=str2bool, nargs='?',default=False)
    parser.add_argument('--geometry', type=str2bool, nargs='?',default=True)
    parser.add_argument('--patch_size', default='3,50,50', type=list_of_ints)
    parser.add_argument('--wandb_project', default="false", type=str)
    parser.add_argument('--wandb_entity', default="xxx", type=str)
    parser.add_argument('--innerLoop', default=50, type=int)
    parser.add_argument('--dataset', default="bridge_orig", type=str)
    parser.add_argument('--resize_patch', type=str2bool, default=False)
    parser.add_argument('--reverse_direction', type=str2bool, default=True)
    parser.add_argument('--server', default="", type=str) 

    # ================= 灰盒攻击的新增特有参数 =================
    parser.add_argument('--malicious_text', type=str, default="pick up the ramekin")
    # parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--tau', default=0.5, type=float)
    
    return parser.parse_args()

def list_of_ints(arg):
    if not arg: return []
    return list(map(int, arg.split(',')))

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    args = arg_parser()
    print(f"Parameters:\n malicious_text:{args.malicious_text}\n lr:{args.lr} \n device:{args.device} \ntags:{args.tags}")
    main(args)