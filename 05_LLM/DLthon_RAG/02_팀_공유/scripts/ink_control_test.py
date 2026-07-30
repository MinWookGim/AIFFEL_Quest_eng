"""수묵화 통제쌍 실험 — CLIP 임베딩은 '내용'을 보는가 '화풍'을 보는가?

왜 이 실험을 하나:
  WikiArt 에서는 화풍과 내용이 뒤엉켜 있었다. 인상주의 그림은 인상주의 소재를 그리고,
  우키요에는 일본 풍경을 그린다. 그래서 "화풍을 찾았다"는 건지 "비슷한 그림을 찾았다"는 건지
  끝내 못 갈랐다.

  수묵화 데이터에는 **같은 원본을 서로 다른 기법으로 그린 세트**가 2,218개 있다.
  내용이 고정돼 있으니 남는 변수가 화풍뿐이다. 여기서 갈린다.

설계 (2요인):
  각 그림에는 content(원본 ID) 와 style(Method 1~9) 두 라벨이 붙는다.
  그림 하나를 질의로 던졌을 때 이웃으로 뭐가 오는지 본다.

    같은 내용 · 다른 화풍  <- 이게 가까우면 CLIP 은 '내용'을 본다
    다른 내용 · 같은 화풍  <- 이게 가까우면 CLIP 은 '화풍'을 본다
    다른 내용 · 다른 화풍  <- 바닥값(baseline)

이 결과가 프로젝트 설계를 바꾼다:
  '내용'을 본다면 -> 지금 임베딩으로는 화풍 검색이 안 된다. 별도 스타일 표현이 필요하다.
  '화풍'을 본다면 -> 지금 방식 그대로 간다.
"""
import os, io, re, json, glob, zipfile, random, collections
import numpy as np
from PIL import Image

random.seed(42)
ROOT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon"
INK  = f"{ROOT}/data/Ai-Hub 데이터/168.한국 전통 수묵화 화풍별 제작 데이터/01-1.정식개방데이터"
OUT  = f"{ROOT}/verify"
os.makedirs(OUT, exist_ok=True)

N_SETS   = 200   # 통제쌍 세트를 몇 개 쓸지 (세트 하나 = 같은 원본, 다른 기법 2~3장)
THUMB    = 384

# ── 1. 라벨을 읽어 (이미지파일 -> 화풍Method, 원본ID) 표를 만든다 ─────────
print("[1] 라벨 읽는 중")
meta = {}                                   # 이미지파일명 -> (method, 원본ID)
for zp in sorted(glob.glob(f"{INK}/*/02.라벨링데이터/*.zip")):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            j = json.loads(z.read(n).decode("utf-8-sig"))
            a, im = j.get("annotation", {}) or {}, j.get("images", {}) or {}
            paire = a.get("Paire")
            m     = (a.get("Paint") or {}).get("Method")
            src   = im.get("identifier")
            if paire and m and src:
                meta[os.path.basename(paire)] = (m, src)
print(f"    라벨 {len(meta):,}건")

# 원본ID 별로 묶어서, 서로 다른 기법이 2개 이상인 세트만 남긴다
by_src = collections.defaultdict(list)
for fn, (m, src) in meta.items():
    by_src[src].append((m, fn))
sets = {s: v for s, v in by_src.items() if len(set(m for m, _ in v)) >= 2}
print(f"    통제쌍 세트(같은 원본 + 다른 기법 2개 이상): {len(sets):,}개")

chosen = dict(random.sample(sorted(sets.items()), min(N_SETS, len(sets))))
wanted = {fn: (m, src) for src, v in chosen.items() for m, fn in v}
print(f"    이번 실험에 쓸 세트 {len(chosen)}개 / 이미지 {len(wanted)}장")

# ── 2. zip 에서 해당 이미지만 꺼낸다 (압축을 풀지 않는다) ─────────────────
print("[2] 이미지 추출 중 (필요한 것만)")
images, labels_style, labels_content, names = [], [], [], []
for zp in sorted(glob.glob(f"{INK}/*/01.원천데이터/*.zip")):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            if base in wanted:
                try:
                    img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                except Exception:
                    continue
                img.thumbnail((THUMB, THUMB))
                m, src = wanted[base]
                images.append(img); labels_style.append(m)
                labels_content.append(src); names.append(base)
    print(f"    {os.path.basename(zp)[:40]:42s} 누적 {len(images)}장")

y_s = np.array(labels_style)          # 화풍
y_c = np.array(labels_content)        # 내용(원본)
print(f"[2] 완료 — {len(images)}장")
print(f"    화풍 분포: {dict(sorted(collections.Counter(y_s).items()))}")

# ── 3. CLIP 임베딩 ────────────────────────────────────────────────────
import torch
from transformers import CLIPModel, CLIPProcessor
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "openai/clip-vit-base-patch32"
print(f"[3] CLIP 임베딩 (device={dev})")
model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
proc  = CLIPProcessor.from_pretrained(MODEL)
embs = []
with torch.no_grad():
    for i in range(0, len(images), 32):
        b = proc(images=images[i:i+32], return_tensors="pt").to(dev)
        f = model.get_image_features(**b)
        embs.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
E = np.concatenate(embs).astype("float32")
np.save(f"{OUT}/ink_emb.npy", E); np.save(f"{OUT}/ink_style.npy", y_s)
np.save(f"{OUT}/ink_content.npy", y_c)
print(f"[3] 완료 — {E.shape}")

# ── 4. 세 종류 쌍의 유사도 비교 ────────────────────────────────────────
print("\n[4] 세 종류 쌍의 코사인 유사도")
S = E @ E.T
N = len(E)
iu = np.triu_indices(N, k=1)
same_c = (y_c[:, None] == y_c[None, :])[iu]
same_s = (y_s[:, None] == y_s[None, :])[iu]
sim = S[iu]

groups = {
    "같은 내용 · 다른 화풍": sim[same_c & ~same_s],
    "다른 내용 · 같은 화풍": sim[~same_c & same_s],
    "다른 내용 · 다른 화풍": sim[~same_c & ~same_s],
}
for k, v in groups.items():
    print(f"    {k}  n={len(v):8,}  평균 {v.mean():.4f}  (표준편차 {v.std():.4f})")

base = groups["다른 내용 · 다른 화풍"].mean()
lift_c = groups["같은 내용 · 다른 화풍"].mean() - base
lift_s = groups["다른 내용 · 같은 화풍"].mean() - base
print(f"\n    바닥값 대비 상승폭")
print(f"      내용을 공유하면 : {lift_c:+.4f}")
print(f"      화풍을 공유하면 : {lift_s:+.4f}")
if lift_c > 0: print(f"      -> 내용 쪽이 화풍 쪽보다 {lift_c/max(lift_s,1e-9):.1f}배 강하다" if lift_s>0
                    else "      -> 화풍 공유는 오히려 유사도를 못 올린다")

# ── 5. 검색 실험 — 이웃으로 뭐가 오나 ──────────────────────────────────
print("\n[5] 검색: 그림 하나를 던지면 이웃 5장이 뭐가 오나")
S2 = S.copy(); np.fill_diagonal(S2, -np.inf)
TOPK = 5
nn = np.argsort(-S2, axis=1)[:, :TOPK]
hit_c = (y_c[nn] == y_c[:, None]).mean()
hit_s = (y_s[nn] == y_s[:, None]).mean()
# 우연 수준
cc = collections.Counter(y_c); cs = collections.Counter(y_s)
chance_c = sum(v*(v-1) for v in cc.values())/(N*(N-1))
chance_s = sum(v*(v-1) for v in cs.values())/(N*(N-1))
print(f"    이웃이 '같은 내용(원본)' 인 비율 : {hit_c:.3f}   (우연 {chance_c:.3f} -> {hit_c/chance_c:5.1f}배)")
print(f"    이웃이 '같은 화풍(기법)' 인 비율 : {hit_s:.3f}   (우연 {chance_s:.3f} -> {hit_s/chance_s:5.1f}배)")

# 내용쌍(같은 원본의 다른 기법 그림)이 몇 등으로 오나
order = np.argsort(-S2, axis=1)
ranks = []
for i in range(N):
    tw = np.where((y_c == y_c[i]) & (np.arange(N) != i))[0]
    if len(tw) == 0: continue
    pos = {v: r for r, v in enumerate(order[i])}
    ranks.append(min(pos[t] for t in tw) + 1)
ranks = np.array(ranks)
print(f"\n    '같은 원본의 다른 기법 그림' 이 몇 등으로 검색되나 (전체 {N}장 중)")
print(f"      중앙값 {np.median(ranks):.0f}등 | 1등 {np.mean(ranks==1)*100:.1f}% | 5등 안 {np.mean(ranks<=5)*100:.1f}% | 10등 안 {np.mean(ranks<=10)*100:.1f}%")

json.dump({
 "n_images": int(N), "n_sets": len(chosen),
 "sim_same_content_diff_style": round(float(groups["같은 내용 · 다른 화풍"].mean()),4),
 "sim_diff_content_same_style": round(float(groups["다른 내용 · 같은 화풍"].mean()),4),
 "sim_baseline": round(float(base),4),
 "neighbor_same_content": round(float(hit_c),4), "chance_content": round(float(chance_c),4),
 "neighbor_same_style":   round(float(hit_s),4), "chance_style":   round(float(chance_s),4),
 "content_twin_rank_median": float(np.median(ranks)),
 "content_twin_top1_pct": round(float(np.mean(ranks==1)*100),2),
}, open(f"{OUT}/ink_control_summary.json","w"), ensure_ascii=False, indent=2)

# ── 6. 눈으로 볼 그림 ─────────────────────────────────────────────────
print("\n[6] 그림 저장")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (a) 통제쌍 예시 — 행=같은 원본, 열=서로 다른 기법
idx_by_src = collections.defaultdict(list)
for i, c in enumerate(y_c): idx_by_src[c].append(i)
picks = [v for v in idx_by_src.values() if len(set(y_s[v])) >= 3][:5]
if picks:
    ncol = max(len(p) for p in picks)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.1*ncol, 3.3*len(picks)), dpi=110)
    axes = np.atleast_2d(axes)
    for r, p in enumerate(picks):
        for c in range(ncol):
            ax = axes[r, c]; ax.axis("off")
            if c < len(p):
                ax.imshow(images[p[c]])
                ax.set_title(f"Method {y_s[p[c]]}", fontsize=11)
        axes[r, 0].set_ylabel("same original", fontsize=9)
    fig.suptitle("Same original, different ink-painting Method (content fixed, style varies)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ink_control_pairs.png", bbox_inches="tight")
    print(f"    -> {OUT}/ink_control_pairs.png  (같은 원본 x 다른 기법)")

# (b) 기법별 예시 — 행=Method, 열=서로 다른 원본
fig, axes = plt.subplots(9, 4, figsize=(13, 29), dpi=95)
for r, m in enumerate(range(1, 10)):
    ids = [i for i in range(N) if y_s[i] == m][:4]
    for c in range(4):
        ax = axes[r, c]; ax.axis("off")
        if c < len(ids): ax.imshow(images[ids[c]])
        if c == 0: ax.set_title(f"Method {m}", fontsize=13, loc="left")
fig.suptitle("Ink painting samples by Method (1~9)", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT}/ink_methods_grid.png", bbox_inches="tight")
print(f"    -> {OUT}/ink_methods_grid.png  (기법 1~9 별 예시)")
print("\n끝.")
