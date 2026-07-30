"""실험: 색을 지우면 그림체가 더 잘 보일까?

CLIP 이 사물·색 분위기에 끌린다는 게 우리 관찰이다. 그럼 색 단서를 아예 지우고
(흑백 변환) 넣으면, 남는 건 선·명암·질감이니 그림체 인식이 오르지 않을까 해서 해본다.
예상: 잉크(원래 무채색)는 오르거나 그대로, 벡터(색이 정체성인 이모지들)는 떨어질 것 같다.
"""
import numpy as np
import torch
from PIL import ImageOps
from transformers import CLIPModel, CLIPProcessor

M = "openai/clip-vit-base-patch32"
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained(M).to(_dev).eval()
_proc = CLIPProcessor.from_pretrained(M)


def encode(images, bs=32):
    # 기준선과 다른 곳은 이 한 줄뿐이다: 넣기 전에 흑백으로 바꾼다
    images = [ImageOps.grayscale(im).convert("RGB") for im in images]
    out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            b = _proc(images=images[i:i + bs], return_tensors="pt").to(_dev)
            f = _model.get_image_features(**b)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype("float32")
