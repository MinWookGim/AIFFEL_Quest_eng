# [재발표 보류실험 실행] 앙상블의 헛경보율 측정 — "두 마리 토끼" 폐기의 원인이었던 미측정 구멍.
# 1차 발표 후 보류했던 실험(README "필요해지면 팀원에게 부탁"). 로컬 CPU로 혼자 돌릴 수 있어서 실행.
#
# 방법 = 기존 두 잣대의 교차점을 그대로 재사용:
#   - 데이터/헛경보 정의 = compare_fp_norm.py 와 동일 (라벨 없는 정상 이미지 sorted 앞 200장,
#     "박스 1개 이상 뜨면 헛경보 이미지", conf 0.25)
#   - 앙상블 = ensemble_wbf_final.py 와 동일 (4 YOLO norm주입 + rtdetr 0.5, WBF IoU 0.55, conf 0.05 수집)
#   - 검증 게이트: 챔피언(yolov8s_n1500)@0.25 가 기존 실측 34/200(17.0%)을 재현해야 신뢰
#   - 스크리닝 운영점(0.05)에서도 병기 (앙상블 recall 0.944 가 나온 그 문턱)
# ★한계(내부 기록): 평가 200장과 norm주입 학습장이 seed 샘플이라 부분 겹침(~29/1500장 추정).
#   1차 단일 측정도 같은 조건이었으므로 "단일 vs 앙상블" 상대 비교는 공정. 절대값은 낙관 편향 가능.
# 사용: conda run -n aiffel python ensemble_fp_normals.py [스모크 장수]
import os, sys, glob, json
import numpy as np
from PIL import Image
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion

HERE = os.path.dirname(os.path.abspath(__file__))
MQ   = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG  = os.path.join(MQ, "data", "NordTank586x371", "images")
LBL  = os.path.join(MQ, "data", "NordTank586x371", "labels")
CR   = os.path.join(MQ, "colab_results")
OUT  = os.path.join(HERE, "..", "results", "ensemble_fp_normals.json")

MODELS = {  # ensemble_wbf_final.py 와 동일 구성
    "yolov8s_n1500":  os.path.join(CR, "yolov8s_norm1500", "weights", "best.pt"),
    "yolo11s_n750":   os.path.join(CR, "yolo11s_norm750",  "weights", "best.pt"),
    "yolov10s_n750":  os.path.join(CR, "yolov10s_norm750", "weights", "best.pt"),
    "yolo12s_n750":   os.path.join(CR, "yolo12s_norm750",  "weights", "best.pt"),
}
CHAMP = "yolov8s_n1500"
RTDETR_W = os.path.join(CR, "rtdetr-l", "weights", "best.pt")
RTDETR_WEIGHT = 0.5
COLLECT_CONF = 0.05          # 수집은 낮게 (앙상블 파이프라인과 동일)
WBF_IOU = 0.55
N_SCAN = 200
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# 정상 200장 = compare_fp_norm.py 와 동일 선택
labeled = set(os.path.splitext(os.path.basename(f))[0]
              for f in glob.glob(os.path.join(LBL, "*.txt")) if os.path.basename(f) != "labels.txt")
normals = sorted(p for p in glob.glob(os.path.join(IMG, "*.png"))
                 if os.path.splitext(os.path.basename(p))[0] not in labeled)[:N_SCAN]
if LIMIT:
    normals = normals[:LIMIT]
print(f"[준비] 정상 이미지 {len(normals)}장, 수집 conf>={COLLECT_CONF}, CPU", flush=True)

models = {n: YOLO(p) for n, p in MODELS.items()}
rtdetr = RTDETR(RTDETR_W)

# method -> 이미지별 최고 점수 목록 (문턱별 헛경보 계산용) / 문턱 이상 박스 수
methods = list(MODELS) + ["ENS_YOLO", "ENS_ALL"]
img_scores = {m: [] for m in methods}          # 이미지마다 [박스 점수들]
for k, p in enumerate(normals):
    W, H = Image.open(p).size
    nb, ns, nl = [], [], []
    for n, mdl in models.items():
        r = mdl.predict(p, conf=COLLECT_CONF, device="cpu", verbose=False)[0]
        b_, s_, l_ = [], [], []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            b_.append([x1/W, y1/H, x2/W, y2/H]); s_.append(float(b.conf[0])); l_.append(int(b.cls[0]))
        img_scores[n].append(s_)
        nb.append(b_); ns.append(s_); nl.append(l_)
    rr = rtdetr.predict(p, conf=COLLECT_CONF, device="cpu", verbose=False)[0]
    rb, rs, rl = [], [], []
    for b in rr.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        rb.append([x1/W, y1/H, x2/W, y2/H]); rs.append(float(b.conf[0])); rl.append(int(b.cls[0]))

    fb, fs, fl = weighted_boxes_fusion(nb, ns, nl, weights=[1]*len(models),
                                       iou_thr=WBF_IOU, skip_box_thr=0.0)
    img_scores["ENS_YOLO"].append([float(s) for s in fs])
    fb, fs, fl = weighted_boxes_fusion(nb+[rb], ns+[rs], nl+[rl],
                                       weights=[1]*len(models)+[RTDETR_WEIGHT],
                                       iou_thr=WBF_IOU, skip_box_thr=0.0)
    img_scores["ENS_ALL"].append([float(s) for s in fs])
    if (k+1) % 20 == 0:
        print(f"  {k+1}/{len(normals)}장", flush=True)

def fp_at(m, thr):
    imgs = sum(1 for ss in img_scores[m] if any(s >= thr for s in ss))
    boxes = sum(sum(1 for s in ss if s >= thr) for ss in img_scores[m])
    return imgs, boxes

print("\n===== 정상 200장 헛경보율 =====")
result = {}
for thr in (0.25, 0.05):
    print(f"\n[conf >= {thr}]")
    for m in methods:
        imgs, boxes = fp_at(m, thr)
        rate = imgs/len(normals)*100
        result[f"{m}@{thr}"] = {"fp_imgs": imgs, "rate_pct": round(rate,1), "fp_boxes": boxes}
        print(f"  {m:15}: 헛경보 이미지 {imgs:3}/{len(normals)} ({rate:5.1f}%), 헛박스 {boxes}개")

ok = result[f"{CHAMP}@0.25"]["fp_imgs"]
print(f"\n[검증 게이트] {CHAMP}@0.25 = {ok}/200 (기존 실측 34/200=17.0%와 비교해 판단)")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"n_normals": len(normals), "note": "compare_fp_norm.py 동일 200장·동일 정의, WBF는 ensemble_wbf_final.py 동일 구성", **result},
          open(OUT, "w"), ensure_ascii=False, indent=2)
print("저장:", OUT)
