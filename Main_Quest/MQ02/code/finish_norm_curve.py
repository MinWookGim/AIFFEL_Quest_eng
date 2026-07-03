# [정상주입 곡선 5점 완성]
# 정상 이미지 주입량(0/250/750/1500/2500)에 따라
#   (1) 헛경보율  = 라벨 없는 "정상" 200장에서 헛박스가 뜬 이미지 비율 (낮을수록 좋음)
#   (2) damage recall + mAP50 = test 301장 공식 val (잣대 A, ultralytics)
# 를 한 번에 재서 곡선 끝점(norm2500)까지 채운다.
# 헛경보는 CPU(기존 compare_fp_norm.py 와 동일 방식: conf 0.25, 200장), val 은 기본 디바이스.
import os, glob, json
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CR   = os.path.join(HERE, "..", "colab_results")
BASE = os.path.join(HERE, "..", "data", "NordTank586x371")
IMG, LBL = os.path.join(BASE, "images"), os.path.join(BASE, "labels")
DATA = os.path.join(HERE, "..", "data", "split", "data.yaml")

# 곡선 5점: (표시이름, 주입량N, best.pt 폴더)
ARMS = [
    ("norm0",    0,    "yolov8s"),
    ("norm250",  250,  "yolov8s_norm250"),
    ("norm750",  750,  "yolov8s_norm750"),
    ("norm1500", 1500, "yolov8s_norm1500"),
    ("norm2500", 2500, "yolov8s_norm2500"),
]
THR, N_SCAN = 0.25, 200

# 정상(라벨 없는) 이미지 200장 = 헛경보 측정용 (labels.txt 는 클래스이름파일이라 제외)
labeled = set(os.path.splitext(os.path.basename(f))[0]
              for f in glob.glob(os.path.join(LBL, "*.txt")) if os.path.basename(f) != "labels.txt")
normals = sorted(p for p in glob.glob(os.path.join(IMG, "*.png"))
                 if os.path.splitext(os.path.basename(p))[0] not in labeled)[:N_SCAN]
print(f"[준비] 정상 이미지 {len(normals)}장(헛경보용), test301 공식val 병행\n")

rows = []
for label, n, folder in ARMS:
    wp = os.path.join(CR, folder, "weights", "best.pt")
    model = YOLO(wp)

    # (1) 헛경보: 정상 200장에 예측 -> 박스가 하나라도 뜨면 헛경보 이미지
    fp_imgs = fp_boxes = 0
    for p in normals:
        r = model.predict(p, conf=THR, device="cpu", verbose=False)[0]
        nb = len(r.boxes)
        if nb:
            fp_imgs += 1; fp_boxes += nb
    fp_rate = fp_imgs / len(normals) * 100

    # (2) test301 공식 val (클래스별 AP/recall)
    v = model.val(data=DATA, split="test", verbose=False)
    per = {}
    for i, c in enumerate(v.box.ap_class_index):
        per[v.names[c]] = {"AP50": float(v.box.ap50[i]), "R": float(v.box.r[i]), "P": float(v.box.p[i])}
    dmg = per.get("damage", {})

    row = {
        "arm": label, "N": n,
        "fp_rate": round(fp_rate, 1), "fp_imgs": fp_imgs, "fp_boxes": fp_boxes,
        "mAP50": round(float(v.box.map50), 4),
        "dmg_R": round(dmg.get("R", 0.0), 4),
        "dmg_AP50": round(dmg.get("AP50", 0.0), 4),
        "dmg_P": round(dmg.get("P", 0.0), 4),
    }
    rows.append(row)
    print(f"{label:9}(N={n:4}): 헛경보 {fp_imgs:3}/{N_SCAN} ({fp_rate:4.1f}%, 박스{fp_boxes:3}) | "
          f"test mAP50 {row['mAP50']:.3f} | damage R {row['dmg_R']:.3f} P {row['dmg_P']:.3f}")

print("\n===== 정상주입 곡선 5점 =====")
hdr = f"{'주입량':10} {'헛경보율':>7} {'헛박스':>6} {'mAP50':>7} {'damage R':>9}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['arm']:10} {r['fp_rate']:6.1f}% {r['fp_boxes']:6} {r['mAP50']:7.3f} {r['dmg_R']:9.3f}")

out = os.path.join(HERE, "..", "results_summary", "norm_curve_points.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out}")
