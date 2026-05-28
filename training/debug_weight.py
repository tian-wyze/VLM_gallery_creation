import torch

ckpt_path = '/home/tian.liu/IDA-VLM/training/runs/20260424_005636_crossattennullslot-distractor-sft-qwen7b-WYZEv03_23_token_Qwen2.5-VL-7B-Instruct_person_expert_wyzev0323token_expert_cross_attn_lr_0.0002_bs_4_captions_False/connector.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
for k, v in ckpt.items():
    if 'expert_fuser' in k:
        print(f"{k:55s}  shape={tuple(v.shape)}  max|w|={v.abs().max().item():7.3f}  nan={torch.isnan(v).any().item()}  inf={torch.isinf(v).any().item()}")
