# -*- coding: utf-8 -*-
"""
이어붙인 실험 — 확률을 반올림해서 주면 도용이 덜 되나.
우리 API 는 응답에서 round(x, 4) 로 잘라 준다. 그게 방어가 되는지 본다.
"""
import torch, torch.nn.functional as F
from torchvision import datasets, transforms
import importlib.util, sys, types

spec = importlib.util.spec_from_file_location("distill", "../DP06/build/distill_experiment.py")
# 위 스크립트를 그대로 다시 돌리면 오래 걸리니 필요한 것만 다시 만든다
from app.model_utils import SimpleClassifier
sys.path.insert(0, "../DP06/build")

torch.set_num_threads(8)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=False, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=False, transform=tf)
X = torch.stack([train_ds[i][0] for i in range(20000)])
Xt = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
yt = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

teacher = SimpleClassifier(10)
teacher.load_state_dict(torch.load("models/mnist_state_dict.pth", map_location="cpu", weights_only=True))
teacher.eval()

@torch.no_grad()
def probs(Z, bs=512):
    return torch.cat([F.softmax(teacher(Z[i:i+bs]), dim=1) for i in range(0, len(Z), bs)])

P = probs(X)
t_pred = probs(Xt).argmax(1)

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def student(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def soften(p, T):
    q = p.clamp_min(1e-12).pow(1.0 / T)
    return q / q.sum(1, keepdim=True)

def run(target, T, seed=0, epochs=30):
    m = student(seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        for xb, tb in DataLoader(TensorDataset(X, target), batch_size=128, shuffle=True):
            opt.zero_grad()
            loss = F.kl_div(F.log_softmax(m(xb) / T, dim=1), tb, reduction="batchmean") * T * T
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (torch.cat([m(Xt[i:i+1024]).argmax(1) for i in range(0, len(Xt), 1024)]) == yt).float().mean().item()
        agr = (torch.cat([m(Xt[i:i+1024]).argmax(1) for i in range(0, len(Xt), 1024)]) == t_pred).float().mean().item()
    return acc, agr

print("질의 20000장, 공격자는 돌려받은 확률을 T=4 로 펴서 학습한다.")
print("다른 건 다 같고, 서버가 확률을 몇 자리까지 주느냐만 바꾼다.\n")
for label, digits in [("자르지 않음 (float 그대로)", None), ("소수 4자리 (지금 우리 API)", 4), ("소수 2자리", 2)]:
    Pr = P if digits is None else torch.round(P, decimals=digits)
    Pr = Pr / Pr.sum(1, keepdim=True).clamp_min(1e-12)
    zeros = (Pr == 0).float().mean().item() * 100
    accs = [run(soften(Pr, 4), 4.0, seed=s) for s in (0, 1)]
    acc = sum(a for a, _ in accs) / len(accs) * 100
    agr = sum(g for _, g in accs) / len(accs) * 100
    lo, hi = min(a for a, _ in accs) * 100, max(a for a, _ in accs) * 100
    print(f"{label:<26} 0으로 뭉개진 칸 {zeros:5.1f}%   정확도 {acc:5.2f}% (seed {lo:.2f}~{hi:.2f})  교사와 일치 {agr:5.2f}%")
