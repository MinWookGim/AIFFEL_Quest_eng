"""
DLthon 스모크 테스트 — 학습 없이 CLIP 임베딩만으로 '화풍'이 갈리는가?

내가 확인하려는 것은 딱 하나다.
  파인튜닝을 한 번도 안 하고, 사전학습 CLIP 임베딩의 코사인 거리만으로
  같은 화풍의 그림끼리 뭉치는가?

뭉친다면 우리 프로젝트의 검색기(Retriever)는 3일 안에 만들 수 있다는 뜻이고,
안 뭉친다면 설계를 처음부터 다시 잡아야 한다.

측정은 두 가지로 한다.
  1) 정량 - precision@k : 그림 한 장을 질의로 던졌을 때 가장 가까운 k장 중
                          같은 화풍인 비율. 이게 곧 RAGAS 의 context_precision 이다.
  2) 정성 - t-SNE 그림 : 사람 눈으로 뭉침을 확인 (발표 슬라이드용)

일부러 '쉬운 화풍 5개 + 헷갈리는 쌍 1개'를 섞었다.
전부 잘 갈리는 결과만 보여주면 발표에서 오히려 못 믿기 때문이다.
"""

import os, sys, json, time, collections
import numpy as np

# 결과를 저장할 곳
OUT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon/verify"
os.makedirs(OUT, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────────────
# 쉬운 5개: 시각적으로 확 갈린다. 파이프라인이 도는지 확인용
EASY = ["Ukiyo_e", "Art_Nouveau", "Cubism", "Baroque", "Pop_Art"]
# 헷갈리는 쌍: 사람도 헷갈린다. 여기서 점수가 떨어지는 걸 보는 게 목적
HARD = ["Impressionism", "Post_Impressionism"]
TARGET_STYLES = EASY + HARD

PER_STYLE = 120        # 화풍당 몇 장 모을지
MAX_SCAN  = 95000      # 전체(81,444행)를 한 바퀴 다 훑는다
PER_ARTIST_CAP = 8     # ★한 작가당 최대 8장. 표본에 작가가 골고루 섞이게 강제한다
                       #  (v2 에서 화풍당 작가가 1~10명뿐이라 측정이 망가졌다)
TOPK      = 5          # precision@k 의 k

# ── 1. 데이터 수집 (스트리밍 — 33.7GB 를 다 받지 않는다) ─────────────
from datasets import load_dataset

print("[1] huggan/wikiart 스트리밍 시작 (전체를 받지 않고 필요한 만큼만 훑는다)")
ds = load_dataset("huggan/wikiart", split="train", streaming=True)

# style 은 ClassLabel 이라 정수로 들어온다. 이름 <-> 정수 대응표를 얻는다.
style_feat = ds.features["style"]
name2id = {n: i for i, n in enumerate(style_feat.names)}
print(f"    데이터셋의 화풍 클래스 총 {len(style_feat.names)}개")

missing = [s for s in TARGET_STYLES if s not in name2id]
if missing:
    print(f"    !! 데이터셋에 없는 이름: {missing}")
    print(f"    실제 이름 목록: {style_feat.names}")
    sys.exit(1)

want = {name2id[s]: s for s in TARGET_STYLES}
buckets  = collections.defaultdict(list)   # style_id -> [PIL image, ...]
artists  = collections.defaultdict(list)   # style_id -> [artist_id, ...]
art_count = collections.Counter()          # (style_id, artist_id) -> 몇 장 담았나

t0 = time.time()
scanned = 0
for row in ds:
    scanned += 1
    sid = row["style"]
    aid = row.get("artist", -1)
    if sid in want and len(buckets[sid]) < PER_STYLE and art_count[(sid, aid)] < PER_ARTIST_CAP:
        art_count[(sid, aid)] += 1
        # 임베딩엔 큰 해상도가 필요 없다. 여기서 바로 줄여 메모리를 아낀다.
        img = row["image"].convert("RGB")
        img.thumbnail((384, 384))
        buckets[sid].append(img)
        artists[sid].append(aid)

    if scanned % 2000 == 0:
        got = sum(len(v) for v in buckets.values())
        need = PER_STYLE * len(TARGET_STYLES)
        na = len(set(k[1] for k in art_count))
        print(f"    {scanned:6d}행 훑음 | 모은 이미지 {got}/{need} | 작가 {na}명 | {time.time()-t0:.0f}초")

    if all(len(buckets[i]) >= PER_STYLE for i in want) or scanned >= MAX_SCAN:
        break

images, labels, arts = [], [], []
for sid, name in want.items():
    for im, a in zip(buckets[sid], artists[sid]):
        images.append(im); labels.append(name); arts.append(a)

print(f"[1] 완료 — {scanned}행 훑어 {len(images)}장 확보 ({time.time()-t0:.0f}초)")
cnt = collections.Counter(labels)
for s in TARGET_STYLES:
    print(f"      {s:26s} {cnt.get(s,0):4d}장")

if len(images) < 100:
    print("!! 이미지가 너무 적다. MAX_SCAN 을 늘리거나 데이터셋을 바꿔야 한다.")
    sys.exit(1)

# ── 2. CLIP 임베딩 (학습 없음. 사전학습 가중치 그대로) ────────────────
import torch
from transformers import CLIPModel, CLIPProcessor

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[2] CLIP 임베딩 계산 (device={dev}) — 파인튜닝 없음, 사전학습 그대로")

MODEL = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
proc  = CLIPProcessor.from_pretrained(MODEL)

embs = []
BS = 32
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(images), BS):
        batch = proc(images=images[i:i+BS], return_tensors="pt").to(dev)
        f = model.get_image_features(**batch)
        f = f / f.norm(dim=-1, keepdim=True)     # 코사인 거리를 쓰려고 단위벡터로
        embs.append(f.cpu().numpy())
E = np.concatenate(embs).astype("float32")
y = np.array(labels)
A = np.array(arts)
print(f"[2] 완료 — {E.shape[0]}장 x {E.shape[1]}차원, {time.time()-t0:.0f}초")
print(f"    임베딩 용량: {E.nbytes/1024/1024:.1f}MB  (원본 이미지 대비 얼마나 작은지 보라)")

np.save(f"{OUT}/emb_wikiart_v3.npy", E)
np.save(f"{OUT}/lab_wikiart_v3.npy", y)
np.save(f"{OUT}/art_wikiart_v3.npy", A)

# ── 3. 정량 평가 — precision@k (= RAGAS 의 context_precision) ─────────
print(f"[3] precision@{TOPK} 측정 — 그림 한 장을 질의로, 가장 가까운 {TOPK}장이 같은 화풍인가")

S = E @ E.T                      # 단위벡터라 내적 = 코사인 유사도
np.fill_diagonal(S, -np.inf)     # 자기 자신은 검색 결과에서 뺀다
nn = np.argsort(-S, axis=1)[:, :TOPK]

hit = (y[nn] == y[:, None])      # (N, TOPK) 불리언
per_img = hit.mean(axis=1)

overall = per_img.mean()
print(f"\n    === 전체 precision@{TOPK} : {overall:.3f} ===\n")

rows = []
for s in TARGET_STYLES:
    m = (y == s)
    if m.sum() == 0: continue
    p = per_img[m].mean()
    grp = "쉬움" if s in EASY else "헷갈림"
    rows.append((s, grp, int(m.sum()), float(p)))
    print(f"    {s:26s} [{grp:4s}] n={m.sum():4d}  precision@{TOPK}={p:.3f}")

easy_p = np.mean([r[3] for r in rows if r[1] == "쉬움"])
hard_p = np.mean([r[3] for r in rows if r[1] == "헷갈림"])
print(f"\n    쉬운 5개 평균   : {easy_p:.3f}")
print(f"    헷갈리는 쌍 평균 : {hard_p:.3f}")
print(f"    차이            : {easy_p - hard_p:+.3f}")

# 헷갈리는 쌍이 서로를 얼마나 잡아먹는지 — 혼동의 실제 내용
print(f"\n    [헷갈리는 쌍의 이웃이 실제로 무슨 화풍이었나]")
for s in HARD:
    m = (y == s)
    neigh = y[nn[m]].ravel()
    c = collections.Counter(neigh).most_common(3)
    tot = len(neigh)
    detail = " · ".join(f"{k} {v/tot*100:.0f}%" for k, v in c)
    print(f"      {s:22s} -> {detail}")


# ── 3.5 ★작가 누수 검증 ─────────────────────────────────────────────
# 위 precision 이 부풀려졌을 수 있다. 스트리밍이 정렬돼 있으면 같은 화풍 표본에
# 같은 작가 그림이 몰린다. 그러면 "같은 화풍을 찾은 것"이 아니라
# "같은 작가를 찾은 것"이고, 화풍 검색기의 성능이 과대평가된다.
print("\n[3.5] 작가 누수 검증 — 이웃이 '같은 화풍'이라서 잡힌 건가, '같은 작가'라서 잡힌 건가")
n_art_per_style = {s_: len(set(A[y == s_])) for s_ in TARGET_STYLES}
print("    화풍별 표본에 들어간 작가 수 (적을수록 누수 위험이 크다)")
for s_ in TARGET_STYLES:
    print(f"      {s_:26s} 작가 {n_art_per_style[s_]:3d}명 / 그림 {int((y==s_).sum())}장")

same_artist = (A[nn] == A[:, None])
print(f"\n    이웃 {TOPK}장 중 '같은 작가'인 비율 : {same_artist.mean():.3f}")

# 같은 작가인 이웃을 아예 빼고 다시 잰다 = 정직한 숫자
S2 = S.copy()
S2[A[:, None] == A[None, :]] = -np.inf     # 같은 작가끼리는 서로 검색 못 하게
nn2 = np.argsort(-S2, axis=1)[:, :TOPK]
per_img2 = (y[nn2] == y[:, None]).mean(axis=1)
overall2 = per_img2.mean()
easy2 = np.mean([per_img2[y == s_].mean() for s_ in EASY])
hard2 = np.mean([per_img2[y == s_].mean() for s_ in HARD])
print(f"\n    === 같은 작가 제외 precision@{TOPK} : {overall2:.3f} ===")
print(f"        (제외 전 {overall:.3f} -> 제외 후 {overall2:.3f}, 차이 {overall2-overall:+.3f})")
print(f"        쉬움 {easy2:.3f} | 헷갈림 {hard2:.3f}")
for s_ in TARGET_STYLES:
    m_ = (y == s_)
    print(f"      {s_:26s} {per_img[m_].mean():.3f} -> {per_img2[m_].mean():.3f}")

summary_leak = {
    "same_artist_neighbor_ratio": round(float(same_artist.mean()), 4),
    "artists_per_style": {k: int(v) for k, v in n_art_per_style.items()},
    "precision_overall_artist_excluded": round(float(overall2), 4),
    "precision_easy_artist_excluded": round(float(easy2), 4),
    "precision_hard_artist_excluded": round(float(hard2), 4),
}

# ── 4. 정성 — t-SNE 그림 (발표 슬라이드용) ────────────────────────────
print("\n[4] t-SNE 그림 생성")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Z = TSNE(n_components=2, perplexity=30, init="pca",
         random_state=42, max_iter=1000).fit_transform(E)

fig, ax = plt.subplots(figsize=(11, 8.5), dpi=140)
palette = plt.get_cmap("tab10")
for i, s in enumerate(TARGET_STYLES):
    m = (y == s)
    if m.sum() == 0: continue
    hard = s in HARD
    ax.scatter(Z[m, 0], Z[m, 1], s=26, alpha=0.85,
               color=palette(i), label=f"{s}{'  (hard pair)' if hard else ''}",
               marker="^" if hard else "o",
               edgecolors="black" if hard else "none", linewidths=0.4)

ax.set_title(f"WikiArt style separation by CLIP embeddings — NO fine-tuning\n"
             f"precision@{TOPK}: overall {overall:.3f} | easy {easy_p:.3f} | hard pair {hard_p:.3f}",
             fontsize=13)
ax.legend(loc="best", fontsize=9, framealpha=0.9)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(f"{OUT}/tsne_wikiart_v3.png", bbox_inches="tight")
print(f"[4] 저장 -> {OUT}/tsne_wikiart_v3.png")

# ── 5. 발표에 그대로 쓸 요약을 파일로 ─────────────────────────────────
summary = {
    "model": MODEL,
    "finetuning": False,
    "n_images": int(len(y)),
    "n_styles": len(TARGET_STYLES),
    "per_style_target": PER_STYLE,
    "rows_scanned": scanned,
    "embedding_dim": int(E.shape[1]),
    "embedding_MB": round(E.nbytes/1024/1024, 2),
    "topk": TOPK,
    "precision_overall": round(float(overall), 4),
    "precision_easy_mean": round(float(easy_p), 4),
    "precision_hard_mean": round(float(hard_p), 4),
    "per_style": [{"style": r[0], "group": r[1], "n": r[2],
                   "precision_at_k": round(r[3], 4)} for r in rows],
}
summary.update(summary_leak)
with open(f"{OUT}/smoke_summary_v3.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[5] 요약 저장 -> {OUT}/smoke_summary_v3.json")
print("\n끝.")
