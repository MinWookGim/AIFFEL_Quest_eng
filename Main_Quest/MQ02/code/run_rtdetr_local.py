"""김민욱 로컬 rtdetr-l 학습 — 팀 공통 split·설정 그대로 재사용(비교 가능하게)."""
from ultralytics import RTDETR

BASE = "/home/gmw/Documents/AIFFEL_Work/_scratch/Main_Quest/MQ02"
DATA = f"{BASE}/data/split/data.yaml"   # 이미 만든 공통 split 재사용

model = RTDETR("rtdetr-l.pt")           # COCO 사전학습 → 우리 2클래스 파인튜닝
model.train(data=DATA, epochs=50, imgsz=640, batch=16, device=0, seed=42,
            project=f"{BASE}/runs", name="rtdetr-l", exist_ok=True)

# 최종: 한 번도 안 본 test 셋 평가 (클래스별 AP 포함)
m = model.val(data=DATA, split="test")
print("=== TEST 결과 ===")
print("mAP50   :", round(float(m.box.map50), 4))
print("mAP50-95:", round(float(m.box.map), 4))
