"""첫 번째 시도 — 내용을 버리고 '어떻게 칠했나'만 남긴다.

원리 한 줄:
    VGG 중간층의 채널끼리 상관(Gram 행렬)을 구하면 **위치 정보가 사라진다.**
    "어디에 무엇이 있나"가 지워지고 "어떤 결·질감이 얼마나 섞였나"만 남는다.
    그래서 내용은 흐려지고 그림체가 남는다.

주의:
    차원이 크다(17만). 1,259장이면 임베딩만 900MB 쯤 된다.
    저층 2개만 쓰면 1만 차원으로 줄지만 점수가 달라진다 -> LAYERS 를 바꿔 실험해 보라.
"""
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights

_dev = "cuda" if torch.cuda.is_available() else "cpu"
_vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(_dev).eval()
for _p in _vgg.parameters():
    _p.requires_grad_(False)

LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1"}   # 저층만: {1, 6}
_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_resize = T.Compose([T.Resize(320), T.CenterCrop(320)])


def _prep(img):
    return _norm(T.functional.to_tensor(_resize(img)))


def encode(images, bs=16):
    outs, maxi = [], max(LAYERS)
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = torch.stack([_prep(im) for im in images[i:i + bs]]).to(_dev)
            per = []
            for li, layer in enumerate(_vgg):
                x = layer(x)
                if li in LAYERS:
                    B, C, H, W = x.shape
                    f = x.reshape(B, C, H * W)
                    G = torch.bmm(f, f.transpose(1, 2)) / (C * H * W)   # 위치가 사라지는 자리
                    t = torch.triu_indices(C, C)                        # 대칭이라 위 삼각만
                    g = G[:, t[0], t[1]]
                    per.append(g / (g.norm(dim=1, keepdim=True) + 1e-8))
                if li >= maxi:
                    break
            v = torch.cat(per, 1)
            outs.append((v / (v.norm(dim=1, keepdim=True) + 1e-8)).cpu().numpy())
    return np.concatenate(outs).astype("float32")
