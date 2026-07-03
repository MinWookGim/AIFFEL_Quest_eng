"""
MQ02 — 두 데이터셋(우리 test / WTBD)에 대해 mAP50 + mAP50-95 함께 계산.
문서 표에 mAP50-95 컬럼을 넣기 위한 재측정. 앙상블 5모델 WBF, imgsz640.
출력: 각 데이터셋별 [method | mAP50 | mAP50-95 | dmg AP50 | dmg AP50-95 | dmg R | dmg P]
"""
import os, glob
import numpy as np
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../ensemble_bundle")
from ensemble_eval import read_gt, ap_recall

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "..", "ensemble_bundle", "weights")
CONF, WBF_IOU, RTW = 0.05, 0.55, 0.5
MODELS = {"yolov8s_n1500": "yolov8s_norm1500.pt", "yolo11s_n750": "yolo11s_norm750.pt",
          "yolov10s_n750": "yolov10s_norm750.pt", "yolo12s_n750": "yolo12s_norm750.pt"}
DATASETS = [
    ("our_test", os.path.join(HERE, "..", "data", "split", "test"), "png"),
    ("WTBD",     os.path.join(HERE, "..", "ETC", "wtbd_yolo"),       "jpg"),
]
IOUS = np.arange(0.5, 1.0, 0.05)   # 0.5,0.55,...,0.95 (10개)


def ap5095(preds, gts):
    """IoU 0.5~0.95 평균 AP (COCO식). GT 없으면 nan."""
    aps = []
    for t in IOUS:
        ap, _, _ = ap_recall(preds, gts, iou_thr=float(t))
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


def run(models, rt, img_dir, lbl_dir):
    methods = list(MODELS) + ["ENS_YOLO", "ENS_ALL(+rtdetr)"]
    preds = {m: {0: [], 1: []} for m in methods}
    gts = {0: {}, 1: {}}
    imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
    imgs = [p for p in imgs if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    from PIL import Image
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        W, H = Image.open(ip).size
        for c, x1, y1, x2, y2 in read_gt(os.path.join(lbl_dir, stem + ".txt"), W, H):
            if c in gts:
                gts[c].setdefault(stem, []).append([x1, y1, x2, y2])
        allb, alls, alll = [], [], []
        for n, m in models.items():
            r = m.predict(ip, conf=CONF, imgsz=640, verbose=False)[0]
            b, s, l = [], [], []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist(); sc = float(box.conf[0]); cl = int(box.cls[0])
                b.append([x1/W, y1/H, x2/W, y2/H]); s.append(sc); l.append(cl)
                preds[n][cl].append((stem, sc, [x1, y1, x2, y2]))
            allb.append(b); alls.append(s); alll.append(l)
        rr = rt.predict(ip, conf=CONF, imgsz=640, verbose=False)[0]
        rb, rs, rl = [], [], []
        for box in rr.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rb.append([x1/W, y1/H, x2/W, y2/H]); rs.append(float(box.conf[0])); rl.append(int(box.cls[0]))
        yb, ys, yl = weighted_boxes_fusion(allb, alls, alll, weights=[1]*4, iou_thr=WBF_IOU, skip_box_thr=0.0)
        for (bx, by, bx2, by2), sc, cl in zip(yb, ys, yl):
            preds["ENS_YOLO"][int(cl)].append((stem, float(sc), [bx*W, by*H, bx2*W, by2*H]))
        fb, fs, fl = weighted_boxes_fusion(allb+[rb], alls+[rs], alll+[rl], weights=[1]*4+[RTW],
                                           iou_thr=WBF_IOU, skip_box_thr=0.0)
        for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
            preds["ENS_ALL(+rtdetr)"][int(cl)].append((stem, float(sc), [bx*W, by*H, bx2*W, by2*H]))
    return methods, preds, gts


def main():
    models = {n: YOLO(os.path.join(WDIR, f)) for n, f in MODELS.items()}
    rt = RTDETR(os.path.join(WDIR, "rtdetr-l.pt"))
    for tag, root, ext in DATASETS:
        img_dir = os.path.join(root, "images"); lbl_dir = os.path.join(root, "labels")
        methods, preds, gts = run(models, rt, img_dir, lbl_dir)
        print(f"\n===== {tag} =====")
        print(f"{'method':17} | {'mAP50':>7} {'mAP50-95':>9} | {'dmgAP50':>8} {'dmgAP5095':>10} | {'dmgR':>6} {'dmgP':>6}")
        for m in methods:
            ap0_50, _, _ = ap_recall(preds[m][0], gts[0])
            ap1_50, r1, p1 = ap_recall(preds[m][1], gts[1])
            ap0_5095 = ap5095(preds[m][0], gts[0])
            ap1_5095 = ap5095(preds[m][1], gts[1])
            mAP50 = np.nanmean([ap0_50, ap1_50])
            mAP5095 = np.nanmean([ap0_5095, ap1_5095])
            print(f"{m:17} | {mAP50:7.4f} {mAP5095:9.4f} | {ap1_50:8.4f} {ap1_5095:10.4f} | {r1:6.3f} {p1:6.3f}")


if __name__ == "__main__":
    main()
