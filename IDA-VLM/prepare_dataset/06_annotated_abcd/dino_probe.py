"""Minimal DINOv2 forward-pass probe to isolate the cublasLtCreate crash.

Runs 3 variants:
  A) default backends, fp32
  B) default backends, bf16 autocast
  C) math-only SDPA backend, fp32  (workaround if A/B crash)
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import glob


def get_batch(n=4):
    tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    paths = sorted(glob.glob(os.path.expanduser(
        "~/bucket_data/wyze_person_v2/annotated_identities/*.jpg")))[:n]
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).cuda()


def main():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").cuda().eval()
    x = get_batch(4)

    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    with torch.no_grad():
        if mode == "A":
            print("A) default backends, fp32")
            y = model(x)
        elif mode == "B":
            print("B) default backends, bf16 autocast")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(x)
        elif mode == "C":
            print("C) math-only SDPA, fp32")
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel([SDPBackend.MATH]):
                y = model(x)
        else:
            raise SystemExit(f"unknown mode {mode}")
    print("ok shape=", y.shape)


if __name__ == "__main__":
    main()
