# -*- coding: utf-8 -*-
"""
8.1 에서 본 2등 값(0.000000003)이 응답에 실려 밖으로 나갈 수 있는 값인가.
증류 실험은 확률을 메모리에서 그대로 건넸기 때문에 이건 따로 확인해야 한다.
"""
import json, torch, torch.nn.functional as F
from torchvision import datasets, transforms
from app.model_utils import SimpleClassifier

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
tr = datasets.MNIST("data", train=True, download=False, transform=tf)
m = SimpleClassifier(10)
m.load_state_dict(torch.load("models/mnist_state_dict.pth", map_location="cpu", weights_only=True))
m.eval()

X = torch.stack([tr[i][0] for i in range(300)])
with torch.no_grad():
    P = F.softmax(m(X), 1)
k = int(P.max(1).values.argmax())      # 8.1 에서 쓴 것과 같은 장 (제일 확신하는 장)
p = P[k].tolist()

print("1) 서버가 확률 열 칸을 그대로 JSON 으로 내보내면")
body = json.dumps({"probs": p})
back = json.loads(body)["probs"]
print("   ", body[:110], "...")
print(f"    2등 값  보낸 것 {sorted(p)[-2]:.12e}")
print(f"            받은 것 {sorted(back)[-2]:.12e}")
print(f"    열 칸이 그대로 복원되나: {p == back}")

print("\n2) round(x, 4) 를 거치면")
r = [round(v, 4) for v in p]
print("   ", json.dumps({"probs": r}))
print(f"    2등 값: {sorted(r)[-2]}")

print("\n3) app/image_api.py 가 실제로 내보내는 응답")
print("   ", json.dumps({"success": True, "label": int(P[k].argmax()),
                         "confidence": round(float(P[k].max()), 4), "user": "userA"}))
