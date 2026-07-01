"""MQ02 실테스트 데모 — yolov8s 예측을 정답(GT)과 나란히 그려 "정말 잡는지" 눈으로 확인.
- test 셋에서 damage 있는 이미지 위주로 골라 GT(초록) | 예측(빨강+신뢰도) 좌우 비교.
- 신뢰도 0.3 이상만 그려 깔끔하게(baseline val 이미지처럼 지저분하지 않게).
- 출력: results_summary/demo/*.png
"""
import os, glob
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE = "/home/gmw/Documents/AIFFEL_Work/_scratch/Main_Quest/MQ02"
IMG_DIR = f"{BASE}/data/split/test/images"
LBL_DIR = f"{BASE}/data/split/test/labels"
OUT = f"{BASE}/results_summary/demo"
os.makedirs(OUT, exist_ok=True)
NAMES = {0: "dirt", 1: "damage"}
GT_COLOR = (0, 200, 0)      # 초록 = 정답
PR_COLOR = (220, 40, 40)    # 빨강 = 예측
CONF = 0.3

model = YOLO(f"{BASE}/colab_results/yolov8s/weights/best.pt")

def read_gt(lbl_path, w, h):
    """YOLO 라벨(정규화 cx cy w h) -> 픽셀 박스 리스트."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    for line in open(lbl_path):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
        x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h)
        x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
        boxes.append((c, x1, y1, x2, y2))
    return boxes

# --- damage(1) 박스가 많은 test 이미지 우선 선별 + dirt 포함 몇 장 ---
scored = []
for lbl in glob.glob(f"{LBL_DIR}/*.txt"):
    cls = [int(float(l.split()[0])) for l in open(lbl) if l.split()]
    scored.append((lbl, cls.count(1), cls.count(0)))
# damage 2개 이상인 것 상위 4장
dmg = sorted([s for s in scored if s[1] >= 2], key=lambda x: -x[1])[:4]
# dirt 포함한 것 2장(있으면)
drt = [s for s in scored if s[2] >= 1][:2]
picked = [s[0] for s in dmg] + [s[0] for s in drt]
picked = list(dict.fromkeys(picked))[:6]   # 중복 제거, 최대 6장

def draw(img, boxes, color, with_conf=False):
    im = img.copy()
    for b in boxes:
        if with_conf:
            c, x1, y1, x2, y2, cf = b
            label = f"{NAMES[c]} {cf:.2f}"
        else:
            c, x1, y1, x2, y2 = b
            label = NAMES[c]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, label, (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return im

for lbl in picked:
    name = os.path.splitext(os.path.basename(lbl))[0]
    img_path = f"{IMG_DIR}/{name}.png"
    if not os.path.exists(img_path):
        img_path = f"{IMG_DIR}/{name}.jpg"
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    gt = read_gt(lbl, w, h)
    r = model.predict(img_path, conf=CONF, verbose=False)[0]
    pr = []
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pr.append((int(box.cls[0]), x1, y1, x2, y2, float(box.conf[0])))

    gt_im = draw(img, gt, GT_COLOR)
    pr_im = draw(img, pr, PR_COLOR, with_conf=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(gt_im); ax[0].set_title(f"Ground Truth ({len(gt)} boxes)", color="green")
    ax[1].imshow(pr_im); ax[1].set_title(f"yolov8s Prediction, conf>{CONF} ({len(pr)} boxes)", color="firebrick")
    for a in ax: a.axis("off")
    plt.suptitle(name, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/demo_{name}.png", dpi=120); plt.close()
    print(f"  {name}: GT {len(gt)} vs Pred {len(pr)}")

print("\n생성 완료:", OUT)
