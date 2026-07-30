"""실험: 내용(무엇이 그려졌나)을 일부러 부수면 그림체만 남을까?

우리 관찰 = CLIP 은 사물을 식별하는 경향이 있다. 그래서 그림을 4x4 조각으로 잘라
섞어버린다. 나무·사람·구도는 부서지고, 선맛·붓질·팔레트 같은 그림체 단서는 조각 안에 남는다.
이걸로 점수가 오르면 "CLIP 이 그동안 내용을 보고 있었다"가 숫자로 증명되는 셈이다.
섞는 순서는 고정(seed)이라 누가 돌려도 같은 결과가 나온다.
"""
import random
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

M = "openai/clip-vit-base-patch32"
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained(M).to(_dev).eval()
_proc = CLIPProcessor.from_pretrained(M)

N = 4                                   # 4x4 = 16조각
_order = list(range(N * N))
random.Random(0).shuffle(_order)        # 섞는 순서를 고정 -> 모든 그림에 같은 '파괴'를 가한다


def _scramble(im, size=224):
    # 정사각형으로 맞춘 뒤 조각내서 고정 순서로 다시 붙인다
    im = im.convert("RGB").resize((size, size))
    t = size // N
    tiles = [im.crop((c * t, r * t, (c + 1) * t, (r + 1) * t))
             for r in range(N) for c in range(N)]
    out = Image.new("RGB", (size, size))
    for i, j in enumerate(_order):
        r, c = divmod(i, N)
        out.paste(tiles[j], (c * t, r * t))
    return out


def encode(images, bs=32):
    images = [_scramble(im) for im in images]
    out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            b = _proc(images=images[i:i + bs], return_tensors="pt").to(_dev)
            f = _model.get_image_features(**b)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype("float32")
