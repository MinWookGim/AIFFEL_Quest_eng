"""MQ02 1단계 결과 — 5종 모델 종합 비교 시각화.
- test 성능(mAP·클래스별AP·damage recall) 그룹막대 + 학습곡선 겹침.
- 라벨은 영어(matplotlib 한글폰트 이슈 회피), 설명 문서는 한글로 따로.
- 출력: results_summary/charts/*.png
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/gmw/Documents/AIFFEL_Work/_scratch/Main_Quest/MQ02"
OUT = f"{BASE}/results_summary/charts"
os.makedirs(OUT, exist_ok=True)

# --- 1) test 성능 데이터: YOLO 4종(json) + rtdetr(verify md 값 하드코딩) ---
with open(f"{BASE}/verify/colab_yolo_test_results.json") as f:
    rows = json.load(f)
# rtdetr-l 은 별도 평가라 json 에 없음 → 성적표 값(verify/rtdetr-l_test_result.md) 수기 추가
rows.append({
    "model": "rtdetr-l",
    "mAP50": 0.800, "mAP50_95": 0.440,
    "dirt":   {"AP50": 0.852, "AP50_95": 0.488, "P": 0.869, "R": 0.754},
    "damage": {"AP50": 0.748, "AP50_95": 0.392, "P": 0.762, "R": 0.674},
})

# 표시 순서: YOLO 계열(오래된→최신) 다음 rtdetr(이방인)
order = ["yolov8s", "yolo11s", "yolov10s", "yolo12s", "rtdetr-l"]
rows = sorted(rows, key=lambda r: order.index(r["model"]))
names = [r["model"] for r in rows]
# YOLO 4종은 같은 계열이라 파란톤, rtdetr 만 주황(이방인 강조)
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]

def barlabel(ax, bars, fmt="%.3f"):
    for b in bars:
        ax.annotate(fmt % b.get_height(), (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)

# --- Fig1: 전체 mAP50 & mAP50-95 ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(names)); w = 0.38
b1 = ax.bar(x-w/2, [r["mAP50"] for r in rows], w, label="mAP50", color="#55A868")
b2 = ax.bar(x+w/2, [r["mAP50_95"] for r in rows], w, label="mAP50-95", color="#C44E52")
barlabel(ax, b1); barlabel(ax, b2)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.0)
ax.set_ylabel("score (test set)"); ax.set_title("MQ02 — Overall Detection Performance (test, 5 models)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/1_overall_mAP.png", dpi=130); plt.close()

# --- Fig2: 클래스별 AP50 (dirt vs damage) ---
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x-w/2, [r["dirt"]["AP50"] for r in rows], w, label="dirt AP50", color="#8172B3")
b2 = ax.bar(x+w/2, [r["damage"]["AP50"] for r in rows], w, label="damage AP50", color="#CCB974")
barlabel(ax, b1); barlabel(ax, b2)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05)
ax.set_ylabel("AP50 (test set)")
ax.set_title("MQ02 — Per-class AP50: dirt(easy) vs damage(bottleneck)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/2_per_class_AP.png", dpi=130); plt.close()

# --- Fig3: damage recall (안전 핵심 지표) ---
fig, ax = plt.subplots(figsize=(9, 5))
dr = [r["damage"]["R"] for r in rows]
bars = ax.bar(x, dr, 0.5, color=colors)
barlabel(ax, bars)
best_i = int(np.argmax(dr))
ax.bar(x[best_i], dr[best_i], 0.5, color="#2CA02C")  # 최고값 초록 강조
ax.annotate("best (safety)", (x[best_i], dr[best_i]+0.03), ha="center", color="#2CA02C", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.0)
ax.set_ylabel("damage recall (test set)")
ax.set_title("MQ02 — Damage Recall = how few real damages are MISSED (safety-critical)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/3_damage_recall.png", dpi=130); plt.close()

# --- Fig4: 학습곡선 val mAP50 겹침 (rtdetr 발산 급락) ---
csv_paths = {
    "yolov8s":  f"{BASE}/colab_results/yolov8s/results.csv",
    "yolo11s":  f"{BASE}/colab_results/yolo11s/results.csv",
    "yolov10s": f"{BASE}/colab_results/yolov10s/results.csv",
    "yolo12s":  f"{BASE}/colab_results/yolo12s/results.csv",
    "rtdetr-l": f"{BASE}/runs/rtdetr-l/results.csv",
}
fig, ax = plt.subplots(figsize=(9, 5))
for name in order:
    df = pd.read_csv(csv_paths[name])
    df.columns = [c.strip() for c in df.columns]
    c = "#DD8452" if name == "rtdetr-l" else None
    lw = 2.2 if name == "rtdetr-l" else 1.4
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label=name, color=c, linewidth=lw)
ax.axvspan(19, 50, color="red", alpha=0.06)
ax.annotate("rtdetr-l diverges (nan)\nfrom ep19 → collapse",
            (19, 0.15), (28, 0.28), fontsize=8, color="#B22222",
            arrowprops=dict(arrowstyle="->", color="#B22222"))
ax.set_xlabel("epoch"); ax.set_ylabel("val mAP50"); ax.set_ylim(0, 1.0)
ax.set_title("MQ02 — Training Curves (val mAP50): YOLO stable vs rtdetr-l divergence")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/4_training_curves.png", dpi=130); plt.close()

print("생성 완료:", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f)
