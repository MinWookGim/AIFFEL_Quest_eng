# [앙상블 프로토타입] YOLO 4종(v8s/11s/v10s/12s)을 WBF로 합쳐 단일 최고모델 대비 recall이 오르나 확인.
# - 재학습 없음: 이미 있는 best.pt 5개로 추론만 (GPU는 rtdetr 학습중 -> CPU 사용, 학습 안 건드림)
# - 평가지표(AP50/recall/precision)는 numpy로 직접 계산 (무거운 라이브러리 X)
# - 스모크: `python ensemble_wbf_prototype.py 10` -> 앞 10장만
import os, sys, glob
import numpy as np
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "data", "split", "test")
IMG_DIR, LBL_DIR = os.path.join(BASE, "images"), os.path.join(BASE, "labels")
CR = os.path.join(HERE, "..", "colab_results")
MODELS = {  # 이름 -> best.pt (강한 YOLO 4종)
    "yolov8s":  os.path.join(CR, "yolov8s",  "weights", "best.pt"),
    "yolo11s":  os.path.join(CR, "yolo11s",  "weights", "best.pt"),
    "yolov10s": os.path.join(CR, "yolov10s", "weights", "best.pt"),
    "yolo12s":  os.path.join(CR, "yolo12s",  "weights", "best.pt"),
}
RTDETR_W = os.path.join(CR, "..", "runs", "rtdetr-l", "weights", "best.pt")  # ep17 best mAP50 0.81
RTDETR_WEIGHT = 0.5   # YOLO(1)보다 낮게 = 약한모델 거들기만
CLASSES = {0: "dirt", 1: "damage"}
CONF = 0.05        # AP 계산용 낮은 임계값 (단일/앙상블 동일 적용 -> 비교 공정)
WBF_IOU = 0.55     # WBF 박스 병합 IoU
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = 전체

from PIL import Image

def read_gt(base, W, H):
    """GT 라벨(YOLO norm) -> 픽셀 xyxy 리스트 [(cls, x1,y1,x2,y2)]"""
    lp = os.path.join(LBL_DIR, base + ".txt"); out = []
    if os.path.exists(lp):
        for line in open(lp):
            p = line.split()
            if len(p) != 5: continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            out.append((c, (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H))
    return out

def iou(a, b):
    """a: (4,), b: (N,4) 픽셀 xyxy -> IoU (N,)"""
    b = np.asarray(b).reshape(-1, 4)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)

def ap_recall(preds, gts, cls, iou_thr=0.5):
    """preds: [(img, score, box_xyxy)], gts: {img:[box_xyxy]} (해당 클래스만)
       return AP50(all-point), recall, precision"""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0: return float("nan"), float("nan"), float("nan")
    preds = sorted(preds, key=lambda x: -x[1])
    matched = {img: np.zeros(len(gts.get(img, [])), bool) for img in gts}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for i, (img, sc, box) in enumerate(preds):
        gt = gts.get(img, [])
        if len(gt) == 0: fp[i] = 1; continue
        ious = iou(np.asarray(box), gt); j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img][j]:
            tp[i] = 1; matched[img][j] = True
        else:
            fp[i] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp / n_gt; prec = ctp / (ctp + cfp + 1e-9)
    # all-point AP (COCO식 monotonic envelope)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre)-1, 0, -1): mpre[k-1] = max(mpre[k-1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))
    return ap, float(rec[-1] if len(rec) else 0), float(prec[-1] if len(prec) else 0)

def main():
    bases = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(IMG_DIR, "*.png")))
    if LIMIT: bases = bases[:LIMIT]
    print(f"[준비] test {len(bases)}장, 모델 {list(MODELS)} (CPU, conf>={CONF})")
    models = {n: YOLO(p) for n, p in MODELS.items()}
    rtdetr = RTDETR(RTDETR_W)   # transformer 계열 = 다양성

    # 저장: method -> class -> preds[(img,score,box)],  gt: class -> {img:[box]}
    methods = list(MODELS) + ["ENS_YOLO", "ENS_ALL(+rtdetr)"]
    preds = {m: {0: [], 1: []} for m in methods}
    gts = {0: {}, 1: {}}

    for k, base in enumerate(bases):
        ip = os.path.join(IMG_DIR, base + ".png")
        W, H = Image.open(ip).size
        for c, x1, y1, x2, y2 in read_gt(base, W, H):
            gts[c].setdefault(base, []).append([x1, y1, x2, y2])
        # 각 모델 예측
        norm_boxes, norm_scores, norm_labels = [], [], []
        for n, mdl in models.items():
            r = mdl.predict(ip, conf=CONF, device="cpu", verbose=False)[0]
            nb, ns, nl = [], [], []
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist(); sc = float(b.conf[0]); cl = int(b.cls[0])
                preds[n][cl].append((base, sc, [x1, y1, x2, y2]))       # 단일모델 기록(픽셀)
                nb.append([x1/W, y1/H, x2/W, y2/H]); ns.append(sc); nl.append(cl)  # WBF용(정규화)
            norm_boxes.append(nb); norm_scores.append(ns); norm_labels.append(nl)
        # rtdetr 예측(정규화)
        rr = rtdetr.predict(ip, conf=CONF, device="cpu", verbose=False)[0]
        rb, rs, rl = [], [], []
        for b in rr.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rb.append([x1/W, y1/H, x2/W, y2/H]); rs.append(float(b.conf[0])); rl.append(int(b.cls[0]))
        # (A) YOLO 4종만
        fb, fs, fl = weighted_boxes_fusion(norm_boxes, norm_scores, norm_labels,
                                           weights=[1]*len(models), iou_thr=WBF_IOU, skip_box_thr=0.0)
        for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
            preds["ENS_YOLO"][int(cl)].append((base, float(sc), [bx*W, by*H, bx2*W, by2*H]))
        # (B) YOLO 4종 + rtdetr(가중치 낮춤)
        fb, fs, fl = weighted_boxes_fusion(norm_boxes+[rb], norm_scores+[rs], norm_labels+[rl],
                                           weights=[1]*len(models)+[RTDETR_WEIGHT], iou_thr=WBF_IOU, skip_box_thr=0.0)
        for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
            preds["ENS_ALL(+rtdetr)"][int(cl)].append((base, float(sc), [bx*W, by*H, bx2*W, by2*H]))
        if (k+1) % 50 == 0: print(f"  ...{k+1}/{len(bases)}")

    # 결과표
    print("\n" + "="*72)
    print(f"{'method':10} | {'mAP50':>7} | {'dirt AP':>7} {'dmg AP':>7} | {'dmg R':>7} {'dmg P':>7}")
    print("-"*72)
    for m in methods:
        ap0, r0, p0 = ap_recall(preds[m][0], gts[0], 0)
        ap1, r1, p1 = ap_recall(preds[m][1], gts[1], 1)
        mAP = np.nanmean([ap0, ap1])
        tag = " <--" if m.startswith("ENS") else ""
        print(f"{m:10} | {mAP:7.4f} | {ap0:7.4f} {ap1:7.4f} | {r1:7.4f} {p1:7.4f}{tag}")
    print("="*72)
    print("관전: ENSEMBLE 의 dmg R(damage recall)이 단일 최고보다 오르나 / mAP50 도 오르나")

if __name__ == "__main__":
    main()
