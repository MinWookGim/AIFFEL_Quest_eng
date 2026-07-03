"""
MQ02 — WBF 앙상블 '최종' (실제 콜랩 4개 best.pt + rtdetr). 수치표 + 시각화용 박스 저장.

★ 프로토타입(ensemble_wbf_rtdetr.py)과 차이:
  - 모델 경로를 실제 콜랩 폴더명(yolov8s_norm1500 등)으로 교정.
  - rtdetr = 콜랩판(colab_results/rtdetr-l, ep22 mAP50 0.809).
  - 챔피언(yolov8s_norm1500) 예측과 앙상블 예측을 이미지별로 저장 -> make_ensemble_viz.py가 그림 생성.
  - AP/recall 계산 함수는 프로토타입 걸 그대로 import (동일 잣대 = 공정 비교).

지표 주의(★): 여기 AP/R/P는 내 numpy(conf>=0.05)라 ultralytics 공식 val 절대값과 다름.
  단일 vs 앙상블에 '동일 잣대'를 써서 상대비교만 유효(앙상블이 오르나 내리나).
사용: conda run -n aiffel python ensemble_wbf_final.py   (스모크: 뒤에 숫자, 예: ... 10)
"""
import os, sys, glob, pickle
import numpy as np
from PIL import Image
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion
# 프로토타입에서 검증된 계산 함수 재사용(중복 방지 + 동일 잣대)
from ensemble_wbf_rtdetr import read_gt, ap_recall

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "data", "split", "test")
IMG_DIR = os.path.join(BASE, "images")
CR = os.path.join(HERE, "..", "colab_results")

MODELS = {   # 이름 -> 실제 콜랩 best.pt (★교정된 폴더명)
    "yolov8s_n1500":  os.path.join(CR, "yolov8s_norm1500", "weights", "best.pt"),
    "yolo11s_n750":   os.path.join(CR, "yolo11s_norm750",  "weights", "best.pt"),
    "yolov10s_n750":  os.path.join(CR, "yolov10s_norm750", "weights", "best.pt"),
    "yolo12s_n750":   os.path.join(CR, "yolo12s_norm750",  "weights", "best.pt"),
}
CHAMP = "yolov8s_n1500"                                  # 단일 챔피언(비교 기준)
RTDETR_W = os.path.join(CR, "rtdetr-l", "weights", "best.pt")
RTDETR_WEIGHT = 0.5     # YOLO(1)보다 낮게 = 약한모델 거들기만
CONF = 0.05             # AP 계산용 낮은 임계값(단일/앙상블 동일)
WBF_IOU = 0.55
OUT_PKL = os.path.join(HERE, "..", "results_summary", "ensemble_wbf", "preds.pkl")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0=전체


def main():
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    bases = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(IMG_DIR, "*.png")))
    if LIMIT:
        bases = bases[:LIMIT]
    print(f"[준비] test {len(bases)}장, 모델 {list(MODELS)}+rtdetr (CPU, conf>={CONF})")
    models = {n: YOLO(p) for n, p in MODELS.items()}
    rtdetr = RTDETR(RTDETR_W)

    methods = list(MODELS) + ["ENS_YOLO", "ENS_ALL(+rtdetr)"]
    preds = {m: {0: [], 1: []} for m in methods}   # method->cls->[(img,score,box_xyxy)]
    gts = {0: {}, 1: {}}
    per_img = {}   # base -> {'wh':(W,H), 'gt':[(cls,box)], 'champ':[(box,sc,cls)], 'ens':[(box,sc,cls)]}

    for k, base in enumerate(bases):
        ip = os.path.join(IMG_DIR, base + ".png")
        W, H = Image.open(ip).size
        gt_list = read_gt(base, W, H)
        for c, x1, y1, x2, y2 in gt_list:
            gts[c].setdefault(base, []).append([x1, y1, x2, y2])

        nb, ns, nl = [], [], []       # WBF 입력(정규화) 모델별
        champ_boxes = []
        for n, mdl in models.items():
            r = mdl.predict(ip, conf=CONF, device="cpu", verbose=False)[0]
            b_, s_, l_ = [], [], []
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                sc = float(b.conf[0]); cl = int(b.cls[0])
                preds[n][cl].append((base, sc, [x1, y1, x2, y2]))
                b_.append([x1/W, y1/H, x2/W, y2/H]); s_.append(sc); l_.append(cl)
                if n == CHAMP:
                    champ_boxes.append(([x1, y1, x2, y2], sc, cl))
            nb.append(b_); ns.append(s_); nl.append(l_)

        rr = rtdetr.predict(ip, conf=CONF, device="cpu", verbose=False)[0]   # rtdetr(정규화)
        rb, rs, rl = [], [], []
        for b in rr.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rb.append([x1/W, y1/H, x2/W, y2/H]); rs.append(float(b.conf[0])); rl.append(int(b.cls[0]))

        # (A) YOLO 4종만
        fb, fs, fl = weighted_boxes_fusion(nb, ns, nl, weights=[1]*len(models),
                                           iou_thr=WBF_IOU, skip_box_thr=0.0)
        for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
            preds["ENS_YOLO"][int(cl)].append((base, float(sc), [bx*W, by*H, bx2*W, by2*H]))

        # (B) YOLO 4종 + rtdetr
        fb, fs, fl = weighted_boxes_fusion(nb+[rb], ns+[rs], nl+[rl],
                                           weights=[1]*len(models)+[RTDETR_WEIGHT],
                                           iou_thr=WBF_IOU, skip_box_thr=0.0)
        ens_boxes = []
        for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
            box = [bx*W, by*H, bx2*W, by2*H]
            preds["ENS_ALL(+rtdetr)"][int(cl)].append((base, float(sc), box))
            ens_boxes.append((box, float(sc), int(cl)))

        per_img[base] = {"wh": (W, H), "gt": gt_list,
                         "champ": champ_boxes, "ens": ens_boxes}
        if (k+1) % 50 == 0:
            print(f"  ...{k+1}/{len(bases)}")

    # ── 결과표 ─────────────────────────────────────────────
    print("\n" + "="*74)
    print(f"{'method':17} | {'mAP50':>7} | {'dirt AP':>7} {'dmg AP':>7} | {'dmg R':>7} {'dmg P':>7}")
    print("-"*74)
    table = {}
    for m in methods:
        ap0, r0, p0 = ap_recall(preds[m][0], gts[0], 0)
        ap1, r1, p1 = ap_recall(preds[m][1], gts[1], 1)
        mAP = np.nanmean([ap0, ap1])
        table[m] = dict(mAP50=mAP, dirtAP=ap0, dmgAP=ap1, dmgR=r1, dmgP=p1)
        tag = " <--" if m.startswith("ENS") else ""
        print(f"{m:17} | {mAP:7.4f} | {ap0:7.4f} {ap1:7.4f} | {r1:7.4f} {p1:7.4f}{tag}")
    print("="*74)

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"table": table, "per_img": per_img, "champ": CHAMP,
                     "methods": methods}, f)
    print(f"저장: {OUT_PKL}  (make_ensemble_viz.py 가 이걸로 그림 생성)")


if __name__ == "__main__":
    main()
