"""11:30 팀 브리핑용 그림 — "생성물을 한 번도 본 적 없는 사람"에게 보여주는 한 장.

왜 필요한가
    팀 세 사람은 검색 쪽만 해왔고 4컷 생성물을 실제로 본 적이 없다.
    그 상태에서 "컷1 보증"·"라벨 일치도"를 숫자로 말하면 안 들어온다.
    -> 잘 된 편 한 줄, 안 된 편 한 줄. 그림 두 줄로 결론이 다 보인다.

★ 잘 된 편을 위에 둔다. 첫인상이 실패면 "우리 안 되는구나"로 읽힌다.
★ 이 그림은 ink 레퍼런스를 안 쓰므로 **공개 자료에도 그대로 쓸 수 있다**
  (unDraw 자유이용 · WikiArt 퍼블릭도메인).
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image

for cand in ("NanumBarunGothic", "NanumGothic", "Noto Sans CJK JP"):
    if any(cand == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        plt.rcParams["axes.unicode_minus"] = False
        break

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "verify", "cut1_guard")
PACK = os.path.join(ROOT, "daypack_v2")

meta = list(csv.DictReader(open(os.path.join(PACK, "meta.csv"), encoding="utf-8")))
paths = [os.path.join(PACK, r["file"]) for r in meta]
style = [r["style"] for r in meta]
R = json.load(open(os.path.join(SRC, "results.json"), encoding="utf-8"))


def pick(tgt, want_good):
    """★ 레퍼런스가 실제로 목표 그림체인 편만 고른다.

    안 그러면 "검색이 엉뚱한 걸 가져온 편"이 뽑혀서, 보는 사람이
    생성 문제인지 검색 문제인지 못 가른다 (실제로 paint_Baroque rep0 은
    레퍼런스 3장이 전부 먹그림이었다). 검색 탓을 못 하게 만들어야
    "생성이 병목"이라는 말이 성립한다.
    """
    c = [r for r in R if r["style"] == tgt and r["ref_hit"] >= 2]
    if not c:
        raise SystemExit(f"{tgt}: 레퍼런스가 2장 이상 맞는 편이 없다")
    return max(c, key=lambda r: r["style_precision_tail"]) if want_good \
        else min(c, key=lambda r: r["style_precision_tail"])


rows = [
    ("잘 따라온 경우", pick("paint_Baroque", True), "#1a7f37"),
    ("안 따라온 경우", pick("vec_undraw", False), "#b42318"),
]

fig, axes = plt.subplots(2, 6, figsize=(6 * 2.5, 2 * 3.2))
for r, (title, rec, color) in enumerate(rows):
    tag = f"{rec['style']}_rep{rec['rep']}_{rec['mode']}"
    # 레퍼런스 2장 — 목표 그림체가 실제로 어떻게 생겼나
    refs = [i for i in rec["ref_idx"] if style[i] == rec["style"]][:2]
    while len(refs) < 2:
        refs.append(rec["ref_idx"][len(refs)])
    for c in range(6):
        ax = axes[r][c]
        ax.axis("off")
        if c < 2:
            ax.imshow(Image.open(paths[refs[c]]).convert("RGB"))
            ax.set_title(f"찾아온 레퍼런스\n({style[refs[c]]})", fontsize=9, color="#444")
        else:
            k = c - 1
            p = os.path.join(SRC, f"{tag}_cut{k}.png")
            if os.path.exists(p):
                ax.imshow(Image.open(p))
            ax.set_title(f"생성 컷{k}", fontsize=9, color=color)
    axes[r][0].text(-0.14, 0.5, f"{title}\n목표: {rec['style']}",
                    transform=axes[r][0].transAxes, fontsize=13, color=color,
                    ha="right", va="center", weight="bold")

fig.suptitle("한 줄 대본 -> 그림체 검색 -> 4컷 생성.  같은 파이프라인, 목표 그림체만 다름",
             fontsize=14, y=1.0)
fig.text(0.5, -0.015,
         "위: 목표가 유화라서 잘 따라왔다.   아래: 목표가 납작한 벡터인데 음영 있는 일러스트로 나왔다.\n"
         "-> 레퍼런스를 정확히 찾아줘도 생성이 안 따라오는 구간이 있다. 이게 지금 병목이다.",
         ha="center", fontsize=11)
fig.tight_layout()
out = os.path.join(SRC, "브리핑_이게_4컷이다.png")
fig.savefig(out, dpi=95, bbox_inches="tight")
print("저장:", out)
for t, rec, _ in rows:
    print(f"  {t}: {rec['style']} rep{rec['rep']} {rec['mode']} "
          f"컷2~4 {rec['style_precision_tail']:.2f} / 라벨 {rec['labels']}")
