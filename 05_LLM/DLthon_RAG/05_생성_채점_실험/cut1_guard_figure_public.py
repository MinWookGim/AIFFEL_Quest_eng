"""컷1보증 결과 확인 그림 — **공개 배포용**.

왜 따로 굽나
    `데이터_공유.md` 78행: "원본 이미지 — 공개 저장소·공개 링크에는 안 올림"
    (AI-Hub 5항 · 이라스토야 재배포 제한). 발표자료는 팀 페이지에 공개되므로 여기에 해당한다.

무엇이 다른가
    **생성물은 그대로 둔다** — 우리가 만든 것이라 제한이 없다.
    **레퍼런스 칸만** 공개 가능한 그림체(vec_undraw = unDraw 자유이용 / paint_Baroque = WikiArt 퍼블릭도메인)는
    그대로 보여주고, 제한 있는 그림체(ink_* = AI-Hub / vec_irasutoya = いらすとや)는 **글자로 대체**한다.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image

# 한글이 네모로 깨지지 않게 폰트를 잡는다 (0806 정리 노트북과 같은 방식)
for cand in ("NanumBarunGothic", "NanumGothic", "Noto Sans CJK JP"):
    if any(cand == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        plt.rcParams["axes.unicode_minus"] = False
        break

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "verify", "cut1_guard")
PACK = os.path.join(ROOT, "daypack_v2")

# 공개 배포가 되는 그림체만 원본을 보여준다 (데이터_공유.md 1절 표)
PUBLIC_OK = {"vec_undraw", "vec_twemoji", "vec_openmoji", "vec_notoemoji"} | {
    f"paint_{s}" for s in
    ("Art_Nouveau", "Baroque", "Cubism", "Impressionism", "Post_Impressionism", "Ukiyo_e")}
# 레퍼런스를 못 보여줄 때 대신 적는 설명
DESC = {"ink_m3": "AI-Hub 수묵화\n(가는 펜선 윤곽\n음영 없음 · 흰 배경)\n\n원본은 공개 배포\n제한이 있어\n글로 대신합니다"}

rows_meta = list(csv.DictReader(open(os.path.join(PACK, "meta.csv"), encoding="utf-8")))
paths = [os.path.join(PACK, r["file"]) for r in rows_meta]
style = [r["style"] for r in rows_meta]

R = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
picks = []
for tgt in ("vec_undraw", "ink_m3", "paint_Baroque"):
    c = [r for r in R if r["style"] == tgt and r["mode"] == "ind"]
    picks.append(max(c, key=lambda r: r["ref_hit"]))

fig, axes = plt.subplots(len(picks), 7, figsize=(7 * 2.3, len(picks) * 2.8))
for row, rec in enumerate(picks):
    tag = f"{rec['style']}_rep{rec['rep']}_ind"
    for col in range(7):
        ax = axes[row][col]
        ax.axis("off")
        if col < 3:
            i = rec["ref_idx"][col]
            s = style[i]
            if s in PUBLIC_OK:
                ax.imshow(Image.open(paths[i]).convert("RGB"))
                ax.set_title(f"REF {col+1}: {s}", fontsize=8,
                             color="green" if s == rec["style"] else "gray")
            else:
                # 원본을 못 싣는 자리 — 빈 칸으로 두지 않고 무엇이었는지 적는다
                ax.text(0.5, 0.5, DESC.get(s, f"{s}\n(공개 제한)"), ha="center", va="center",
                        fontsize=7.5, color="#444",
                        bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#bbb"))
                ax.set_title(f"REF {col+1}: {s}", fontsize=8,
                             color="green" if s == rec["style"] else "gray")
        else:
            k = col - 2
            p = os.path.join(OUT, f"{tag}_cut{k}.png")
            if os.path.exists(p):
                ax.imshow(Image.open(p))
            lab = rec["labels"][k - 1]
            ax.set_title(f"GEN cut{k} -> {lab}", fontsize=8,
                         color="green" if lab == rec["style"] else "red")
    axes[row][0].set_ylabel(rec["style"])
    axes[row][0].axis("on")
    axes[row][0].set_xticks([]); axes[row][0].set_yticks([])

fig.suptitle("TARGET STYLE (left 3 = retrieved references) vs GENERATED 4 cuts  "
             "[independent generation]  *public-safe version", fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, "cut1_guard_what_came_out_public.png")
fig.savefig(p, dpi=95, bbox_inches="tight")
print("저장:", p)
shown = sum(1 for rec in picks for i in rec["ref_idx"] if style[i] in PUBLIC_OK)
print(f"레퍼런스 9칸 중 원본 표시 {shown}칸 / 글자 대체 {9-shown}칸")
