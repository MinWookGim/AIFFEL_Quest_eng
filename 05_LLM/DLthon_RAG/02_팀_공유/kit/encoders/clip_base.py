"""출발점 — CLIP 임베딩을 그대로 쓴다. 학습 0회.

이게 지금 업계 기본값이고, 우리 리더보드의 바닥이다.
너의 방법은 이걸 이겨야 의미가 있다.
"""
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

M = "openai/clip-vit-base-patch32"
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained(M).to(_dev).eval()
_proc = CLIPProcessor.from_pretrained(M)


def encode(images, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            b = _proc(images=images[i:i + bs], return_tensors="pt").to(_dev)
            f = _model.get_image_features(**b)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype("float32")
