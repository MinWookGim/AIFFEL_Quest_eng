# -*- coding: utf-8 -*-
"""
'펴기'를 얼마나 펴는 게 좋은가 — 온도 T 를 올려가며 재본다.
질문: 펴면 좋아진다면 계속 펴면 계속 좋아지나? 부작용은 무엇인가?
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from app.model_utils import SimpleClassifier

torch.set_num_threads(8)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr = datasets.MNIST("data", train=True, download=False, transform=tf)
te = datasets.MNIST("data", train=False, download=False, transform=tf)
X  = torch.stack([tr[i][0] for i in range(20000)])
Xt = torch.stack([te[i][0] for i in range(len(te))])
yt = torch.tensor([te[i][1] for i in range(len(te))])

teacher = SimpleClassifier(10)
teacher.load_state_dict(torch.load("models/mnist_state_dict.pth", map_location="cpu", weights_only=True))
teacher.eval()

@torch.no_grad()
def probs(Z, bs=512):
    return torch.cat([F.softmax(teacher(Z[i:i+bs]), dim=1) for i in range(0, len(Z), bs)])

P = probs(X)
t_pred = probs(Xt).argmax(1)

def soften(p, T):
    q = p.clamp_min(1e-12).pow(1.0 / T)
    return q / q.sum(1, keepdim=True)

def run(target, T, seed, hard=False):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(30):
        for xb, tb in DataLoader(TensorDataset(X, target), batch_size=128, shuffle=True):
            opt.zero_grad()
            z = m(xb)
            loss = F.cross_entropy(z, tb) if hard else F.kl_div(F.log_softmax(z / T, 1), tb, reduction="batchmean") * T * T
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pr = torch.cat([m(Xt[i:i+1024]).argmax(1) for i in range(0, len(Xt), 1024)])
    return (pr == yt).float().mean().item(), (pr == t_pred).float().mean().item()

SEEDS = (0, 1, 2)
print("질의 20000 번 고정. 온도 T 만 바꾼다. seed 3개 평균.\n")
print(f"{'설정':<16}{'1등이 차지하는 비중':>20}{'정확도':>26}{'교사와 일치':>14}")

a = [run(P.argmax(1), 1.0, s, hard=True) for s in SEEDS]
acc = [x[0] for x in a]
print(f"{'label 만':<16}{'-':>20}   {sum(acc)/3*100:6.2f}% ({min(acc)*100:.2f}~{max(acc)*100:.2f})"
      f"{sum(x[1] for x in a)/3*100:12.2f}%", flush=True)

for T in (1, 2, 4, 8, 16, 32):
    S = soften(P, T)
    top = S.max(1).values.mean().item()
    a = [run(S, float(T), s) for s in SEEDS]
    acc = [x[0] for x in a]
    print(f"{'T=' + str(T):<16}{top:>20.4f}   {sum(acc)/3*100:6.2f}% ({min(acc)*100:.2f}~{max(acc)*100:.2f})"
          f"{sum(x[1] for x in a)/3*100:12.2f}%", flush=True)
