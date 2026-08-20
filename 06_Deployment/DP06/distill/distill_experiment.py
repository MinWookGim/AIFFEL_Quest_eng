# -*- coding: utf-8 -*-
"""
모델 도용(model stealing) 실험 — 응답에 무엇을 담느냐가 얼마나 새는가.

교사 = 오늘 API 뒤에 있는 MNIST 모델. 가중치는 공격자에게 안 준다.
공격자 = 이미지를 던져서 돌아온 응답만으로 자기 모델을 학습시킨다. 진짜 라벨은 모른다.
"""
import time, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from app.model_utils import SimpleClassifier

DEV = "cpu"
torch.set_num_threads(8)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=False, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=False, transform=tf)

X_train = torch.stack([train_ds[i][0] for i in range(20000)])
y_train = torch.tensor([train_ds[i][1] for i in range(20000)])
X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
print(f"질의용 이미지 {len(X_train)}장, 평가용 테스트셋 {len(X_test)}장")

# ===== 교사 (내 API 뒤의 모델) =====
teacher = SimpleClassifier(10)
teacher.load_state_dict(torch.load("models/mnist_state_dict.pth", map_location="cpu", weights_only=True))
teacher.eval()

@torch.no_grad()
def teacher_probs(X, bs=512):
    out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(teacher(X[i:i+bs]), dim=1))
    return torch.cat(out)

@torch.no_grad()
def accuracy(model, X, y, bs=1024):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item()
    return correct / len(X)

@torch.no_grad()
def agreement(model, X, t_pred, bs=1024):
    """교사와 같은 답을 내는 비율 (정답이 아니라 교사를 얼마나 베꼈나)"""
    model.eval()
    same = 0
    for i in range(0, len(X), bs):
        same += (model(X[i:i+bs]).argmax(1) == t_pred[i:i+bs]).sum().item()
    return same / len(X)

P_train = teacher_probs(X_train)
P_test = teacher_probs(X_test)
t_pred_test = P_test.argmax(1)
teacher_acc = (t_pred_test == y_test).float().mean().item()
print(f"교사 정확도(테스트 1만장): {teacher_acc*100:.2f}%")
print(f"교사 평균 확신도: {P_train.max(1).values.mean().item():.4f}\n")


class StudentMLP(nn.Module):
    """공격자의 모델. 교사와 구조가 다르다 (남의 구조는 모르니까)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)


def soften(p, T):
    """돌려받은 확률을 공격자가 부드럽게 펴는 것. p^(1/T) 를 다시 1로 맞춘다."""
    if T == 1:
        return p
    q = p.clamp_min(1e-12).pow(1.0 / T)
    return q / q.sum(1, keepdim=True)


def train_student(X, target, mode, T=1.0, epochs=30, seed=0):
    torch.manual_seed(seed)
    model = StudentMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X, target), batch_size=128, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, tb in loader:
            opt.zero_grad()
            z = model(xb)
            if mode == "hard":
                loss = F.cross_entropy(z, tb)
            else:
                loss = F.kl_div(F.log_softmax(z / T, dim=1), tb, reduction="batchmean") * (T * T)
            loss.backward()
            opt.step()
    return model


BUDGETS = [1000, 5000, 20000]
SEEDS = [0, 1, 2]
rows = []

for budget in BUDGETS:
    Xq, Pq, yq = X_train[:budget], P_train[:budget], y_train[:budget]
    settings = [
        ("hard  (label 만)",        "hard", None,             1.0),
        ("T=1   (확률 전부)",        "soft", soften(Pq, 1),    1.0),
        ("T=4   (확률을 펴서)",      "soft", soften(Pq, 4),    4.0),
        ("true  (진짜 정답, 비교용)", "hard", None,             1.0),
    ]
    for name, mode, tgt, T in settings:
        if name.startswith("hard"):
            target = Pq.argmax(1)
        elif name.startswith("true"):
            target = yq
        else:
            target = tgt
        accs, agrs = [], []
        for seed in SEEDS:
            t0 = time.time()
            m = train_student(Xq, target, mode, T=T, seed=seed)
            accs.append(accuracy(m, X_test, y_test))
            agrs.append(agreement(m, X_test, t_pred_test))
        rows.append((budget, name, accs, agrs))
        lo, hi = min(accs) * 100, max(accs) * 100
        print(f"질의 {budget:>5}  {name:<22} 정확도 {sum(accs)/len(accs)*100:5.2f}% "
              f"(seed 폭 {lo:.2f}~{hi:.2f})  교사와 일치 {sum(agrs)/len(agrs)*100:5.2f}%  "
              f"[{time.time()-t0:.0f}s/seed]", flush=True)

# ===== 진짜 이미지가 하나도 없는 공격자 =====
print("\n질의 이미지를 난수로 만들면 (공격자가 MNIST 를 아예 못 구한 경우)")
torch.manual_seed(0)
X_noise = torch.rand(20000, 1, 28, 28)
X_noise = (X_noise - 0.1307) / 0.3081
P_noise = teacher_probs(X_noise)
for name, T in [("hard  (label 만)", 1.0), ("T=1   (확률 전부)", 1.0), ("T=4   (확률을 펴서)", 4.0)]:
    if name.startswith("hard"):
        target, mode = P_noise.argmax(1), "hard"
    else:
        target, mode = soften(P_noise, int(T)), "soft"
    m = train_student(X_noise, target, mode, T=T, seed=0)
    print(f"질의 20000  {name:<22} 정확도 {accuracy(m, X_test, y_test)*100:5.2f}%  "
          f"교사와 일치 {agreement(m, X_test, t_pred_test)*100:5.2f}%", flush=True)

json.dump([{"budget": b, "setting": n, "acc": a, "agree": g} for b, n, a, g in rows],
          open("../DP06/verify/distill_results.json", "w"), ensure_ascii=False, indent=1)
print("\n결과 저장: verify/distill_results.json")
