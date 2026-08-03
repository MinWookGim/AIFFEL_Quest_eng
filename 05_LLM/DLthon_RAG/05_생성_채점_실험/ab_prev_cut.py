"""앞 컷을 물리면 4컷이 정말 한 그림체로 붙나 — 2x2 실험.

주의: 이 스크립트는 v1 동결본을 건드리지 않는다. 결과만 `verify/ab_prev_cut/` 에 쌓는다.

무엇을 재나
    지금 `build/gen_style_test.py` 는 컷마다 **독립으로** 호출한다(66행 for 문).
    매 컷이 레퍼런스만 보고 그려지므로 **컷끼리 서로를 모른다.**
    앞 컷을 참조에 얹으면 붙는지 본다.

왜 2x2 인가
    "검색도 좋게, 생성도 좋게" 가 최종 목표인 건 맞다. 그런데 둘을 동시에 바꾸면
    **무엇이 효과였는지 못 가린다.** 그래서 네 칸을 다 돌려 각각의 효과와 상호작용을 같이 본다.

        생성\검색      CLIP 레퍼런스     Gram 레퍼런스
        독립(지금)         (1)              (2)
        순차(앞컷물림)      (3)              (4)

      (3)-(1) = 앞 컷 물리기 효과      (2)-(1) = 검색기 교체 효과
      (4)-(1) = 둘 다      /  (4)-(3)-(2)+(1) = 상호작용 (같이 쓰면 더 좋아지나)

채점
    오늘 실측 결론대로 **Gram 으로 채점한다** (컷 단위 라벨 정확도 CLIP 0.296 < Gram 0.440).
    생성물을 코퍼스에 질의로 던져 이웃 5장의 최다 라벨로 판정 = 역방향 채점.
    주의: 레퍼런스로 쓴 그림은 이웃 후보에서 뺀다. 안 빼면 베낄수록 점수가 오른다.

쓰는 법
    python build/ab_prev_cut.py --dry-run            # 계획과 예상 비용만 (API 안 부름)
    python build/ab_prev_cut.py --styles m3 m2 --reps 2
    python build/ab_prev_cut.py --analyze-only       # 이미 뽑아둔 결과만 다시 분석
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "daypack_v1")
OUT = os.path.join(ROOT, "verify", "ab_prev_cut")
CACHE = os.path.join(ROOT, "verify", "cache")
MODEL, SIZE, QUALITY = "gpt-image-1", "1024x1024", "medium"
N_REF = 3          # 레퍼런스 장수 (지금 파이프라인과 동일)
N_CUTS = 4
TOPK = 5
RATE = {"text_in": 5.0 / 1e6, "image_in": 10.0 / 1e6, "image_out": 40.0 / 1e6}

sys.path.insert(0, os.path.join(ROOT, "kit"))


# ─────────────────────────────────────────────────────────────
# 1. 코퍼스와 임베딩 캐시
# ─────────────────────────────────────────────────────────────
def load_pack():
    """daypack_v1 의 이미지·라벨을 읽는다. (591장 임시 코퍼스가 아니라 **본 코퍼스**)"""
    import csv
    rows = list(csv.DictReader(open(os.path.join(PACK, "meta.csv"), encoding="utf-8")))
    paths = [os.path.join(PACK, r["file"]) for r in rows]
    style = np.array([r["style"] for r in rows])
    content = np.array([r.get("content", "") for r in rows])
    return paths, style, content


def get_embeddings(kind, paths):
    """CLIP / Gram 임베딩을 캐시해 둔다. 한 번 계산하면 다음부터는 즉시 로드."""
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, f"daypack_v1_{kind}.npy")
    if os.path.exists(cache_path):
        E = np.load(cache_path)
        print(f"    [{kind}] 캐시 사용 {E.shape}")
    else:
        print(f"    [{kind}] 캐시가 없다. {len(paths)}장 계산 중… (한 번만 걸린다)")
        import importlib.util
        mod_path = os.path.join(ROOT, "kit", "encoders",
                                {"clip": "clip_base.py", "gram": "gram_vgg19.py"}[kind])
        spec = importlib.util.spec_from_file_location(f"enc_{kind}", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        imgs = [Image.open(p).convert("RGB") for p in paths]
        E = np.asarray(mod.encode(imgs), dtype="float32")
        np.save(cache_path, E)
        print(f"    [{kind}] 계산 완료 {E.shape} -> 캐시 저장")
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


def encode_one(kind, img):
    """생성된 그림 한 장을 코퍼스와 같은 방식으로 임베딩한다."""
    import importlib.util
    mod_path = os.path.join(ROOT, "kit", "encoders",
                            {"clip": "clip_base.py", "gram": "gram_vgg19.py"}[kind])
    spec = importlib.util.spec_from_file_location(f"enc1_{kind}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = np.asarray(mod.encode([img]), dtype="float32")[0]
    return v / (np.linalg.norm(v) + 1e-8)


# ─────────────────────────────────────────────────────────────
# 2. 레퍼런스 검색 — 여기가 검색 쪽 변수
# ─────────────────────────────────────────────────────────────
def retrieve_refs(E, style, content, target, query_idx, n=N_REF):
    """질의 그림 한 장을 주고 이웃 n 장을 뽑는다. 이게 생성에 물릴 레퍼런스다.

    주의: 질의 자신과 **같은 원본**은 뺀다. 안 빼면 내용이 같은 걸 뽑고 그림체를 뽑았다고 착각한다.
    주의: 라벨로 거르지 않는다. 검색기 실력이 그대로 드러나야 CLIP/Gram 비교가 성립한다.
    """
    sims = E @ E[query_idx]
    ok = content != content[query_idx]
    ok[query_idx] = False
    sims[~ok] = -np.inf
    top = np.argsort(-sims)[:n]
    hit = int((style[top] == target).sum())
    return [int(i) for i in top], hit


# ─────────────────────────────────────────────────────────────
# 3. 생성 — 여기가 생성 쪽 변수
# ─────────────────────────────────────────────────────────────
def to_file(img, name):
    b = io.BytesIO()
    img.save(b, format="PNG")
    b.name = name
    b.seek(0)
    return b


def cost_of(usage):
    if usage is None:
        return {}
    det = getattr(usage, "input_tokens_details", None)
    t_in = getattr(det, "text_tokens", 0) or 0
    i_in = getattr(det, "image_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    if not (t_in or i_in):
        t_in = getattr(usage, "input_tokens", 0) or 0
    usd = t_in * RATE["text_in"] + i_in * RATE["image_in"] + out * RATE["image_out"]
    return {"텍스트입력토큰": t_in, "이미지입력토큰": i_in, "출력토큰": out, "달러": round(usd, 4)}


PROMPT = ("Draw a NEW illustration of this scene: {scene}. "
          "Match the painting style of the reference images exactly — "
          "the brush texture, ink tone, line quality and color palette. "
          "Do not copy the content of the references, only their style.")

PROMPT_SEQ = ("Draw a NEW illustration of this scene: {scene}. "
              "Match the painting style of the reference images exactly — "
              "the brush texture, ink tone, line quality and color palette. "
              "The LAST reference image is the previous panel of this same comic: "
              "keep the identical style, characters and palette as that panel, "
              "but draw a different moment. "
              "Do not copy the content of any reference, only the style.")


def generate_edition(client, ref_imgs, scenes, sequential):
    """4컷 한 편을 생성한다.

    sequential=False : 컷마다 레퍼런스만 (지금 방식)
    sequential=True  : 컷2~4 에 **컷1 을 참조로 추가** (앞 컷 물림)

    주의: 컷1 은 두 조건에서 같은 입력을 받지만 결과는 다르다(생성이 비결정적).
      즉 컷1 차이는 잡음이다. 그래서 아래 분석에서 **컷2~4 만으로도** 따로 본다.
    """
    cuts, costs = [], []
    for k, scene in enumerate(scenes, 1):
        imgs = list(ref_imgs)
        prompt = PROMPT.format(scene=scene)
        if sequential and cuts:
            imgs = list(ref_imgs) + [cuts[0]]      # 맨 뒤에 컷1 을 붙인다
            prompt = PROMPT_SEQ.format(scene=scene)
        t0 = time.time()
        r = client.images.edit(
            model=MODEL, size=SIZE, quality=QUALITY,
            image=[to_file(x, f"r{i}.png") for i, x in enumerate(imgs)],
            prompt=prompt)
        dt = time.time() - t0
        img = Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
        cuts.append(img)
        costs.append({"컷": k, "초": round(dt, 1), **cost_of(getattr(r, "usage", None))})
    return cuts, costs


# ─────────────────────────────────────────────────────────────
# 4. 채점 — 역방향 채점 (Gram)
# ─────────────────────────────────────────────────────────────
def score_edition(cuts, Eg, style, content, target, ref_idx):
    """생성된 4컷을 코퍼스에 질의로 던져 라벨을 붙이고 지표를 낸다.

    주의: 레퍼런스로 물린 그림과 **같은 원본**은 이웃 후보에서 뺀다.
      안 빼면 레퍼런스를 베낄수록 점수가 오른다 (그림체가 아니라 복사를 재게 된다).
    """
    banned = np.zeros(len(style), dtype=bool)
    for i in ref_idx:
        banned |= (content == content[i])

    V = np.stack([encode_one("gram", c) for c in cuts])
    labels, top_hits = [], []
    for v in V:
        sims = Eg @ v
        sims[banned] = -np.inf
        top = np.argsort(-sims)[:TOPK]
        lab, cnt = np.unique(style[top], return_counts=True)
        labels.append(lab[np.argmax(cnt)])
        top_hits.append(float((style[top] == target).mean()))

    labels = np.array(labels)
    S = V @ V.T
    iu = np.triu_indices(len(cuts), k=1)
    return {
        "style_precision": float((labels == target).mean()),   # 4컷 중 목표 라벨로 판정된 비율
        "style_precision_tail": float((labels[1:] == target).mean()),  # 컷2~4 만 (컷1 잡음 제외)
        "neighbor_hit": float(np.mean(top_hits)),              # 이웃 5장 중 목표 라벨 비율 평균
        "edition_cosine": float(S[iu].mean()),                 # 컷 간 유사도 = 복붙 탐지용
        "labels": [str(x) for x in labels],
    }


# ─────────────────────────────────────────────────────────────
# 5. 실험 진행
# ─────────────────────────────────────────────────────────────
def load_scenes(n=N_CUTS):
    """장면 4개. 기존 대본이 있으면 쓰고, 없으면 기본 장면으로."""
    p = os.path.join(ROOT, "verify", "demo_stories.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        for v in d.values():
            if isinstance(v, list) and len(v) >= n:
                return [c["scene"] for c in v[:n]]
    return ["a lone traveler walking along a mountain path",
            "the traveler stops at a small stream and looks at the water",
            "the traveler meets an old man under a pine tree",
            "the two of them walk toward a distant village at dusk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", nargs="+", default=["m3", "m2"],
                    help="목표 그림체. 천장이 높은 것부터 (m3 3.34배 / m2 3.00배 / m1 2.26배)")
    ap.add_argument("--reps", type=int, default=2, help="조건당 반복 횟수")
    ap.add_argument("--retrievers", nargs="+", default=["clip", "gram"])
    ap.add_argument("--dry-run", action="store_true", help="계획·예상비용만 보고 끝낸다")
    ap.add_argument("--analyze-only", action="store_true", help="이미 뽑은 결과만 재분석")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    results_path = os.path.join(OUT, "results.json")

    conds = [(r, s) for r in a.retrievers for s in (False, True)]
    n_editions = len(conds) * len(a.styles) * a.reps
    n_images = n_editions * N_CUTS
    est = n_images * 0.046      # 실측 장당 평균

    print(f"조건 {len(conds)}개 x 그림체 {len(a.styles)}종 x 반복 {a.reps}회 = {n_editions}편")
    print(f"이미지 {n_images}장 / 예상 비용 약 ${est:.2f} / 예상 시간 약 {n_images*19.2/60:.0f}분")
    print(f"조건: " + ", ".join(f"{r}+{'순차' if s else '독립'}" for r, s in conds))
    if a.dry_run:
        print("\n--dry-run 이라 여기서 멈춘다. 실제로 돌리려면 --dry-run 을 빼라.")
        return

    print("\n[1] 코퍼스와 임베딩")
    paths, style, content = load_pack()
    print(f"    daypack_v1 {len(paths)}장 / {len(np.unique(style))}클래스")
    E = {k: get_embeddings(k, paths) for k in set(a.retrievers) | {"gram"}}
    Eg = E["gram"]

    scenes = load_scenes()
    rng = np.random.default_rng(20260803)

    if not a.analyze_only:
        from openai import OpenAI
        client = OpenAI(api_key=open(os.path.expanduser("~/.config/openai/api_key")).read().strip())

        records = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else []
        done = {(r["retriever"], r["sequential"], r["style"], r["rep"]) for r in records}

        print("\n[2] 생성")
        for target in a.styles:
            pool = np.where(style == target)[0]
            for rep in range(a.reps):
                # 질의 그림은 조건마다 **같은 것**을 쓴다. 안 그러면 비교가 안 된다
                q = int(rng.choice(pool))
                for retr, seq in conds:
                    key = (retr, seq, target, rep)
                    if key in done:
                        print(f"    건너뜀 (이미 있음) {key}")
                        continue
                    ref_idx, ref_hit = retrieve_refs(E[retr], style, content, target, q)
                    ref_imgs = [Image.open(paths[i]).convert("RGB") for i in ref_idx]
                    tag = f"{target}_rep{rep}_{retr}_{'seq' if seq else 'ind'}"
                    print(f"    {tag}  (레퍼런스 {ref_hit}/{N_REF} 장이 실제 {target})", flush=True)
                    try:
                        cuts, costs = generate_edition(client, ref_imgs, scenes, seq)
                    except Exception as e:
                        print(f"      !! 실패: {repr(e)[:160]}")
                        continue
                    for k, c in enumerate(cuts, 1):
                        c.save(os.path.join(OUT, f"{tag}_cut{k}.png"))
                    sc = score_edition(cuts, Eg, style, content, target, ref_idx)
                    rec = {"retriever": retr, "sequential": seq, "style": target, "rep": rep,
                           "query_idx": q, "ref_idx": ref_idx, "ref_hit": ref_hit,
                           "panels_produced": len(cuts),
                           "latency_sec": round(sum(c["초"] for c in costs), 1),
                           "cost_usd": round(sum(c.get("달러", 0) for c in costs), 4),
                           **sc}
                    records.append(rec)
                    json.dump(records, open(results_path, "w"), ensure_ascii=False, indent=1)
                    print(f"      style_prec {sc['style_precision']:.2f} "
                          f"(컷2~4 {sc['style_precision_tail']:.2f}) "
                          f"컷간코사인 {sc['edition_cosine']:.3f}  ${rec['cost_usd']:.3f}")

    print("\n[3] 분석")
    analyze(json.load(open(results_path, encoding="utf-8")))


def analyze(records):
    if not records:
        print("    결과가 없다.")
        return
    import itertools

    def cell(retr, seq):
        return [r for r in records if r["retriever"] == retr and r["sequential"] == seq]

    retrs = sorted({r["retriever"] for r in records})
    print(f"\n    {'조건':18s} {'n':>3s} {'style_prec':>11s} {'컷2~4':>8s} {'컷간코사인':>11s} {'초':>6s}")
    means = {}
    for retr, seq in itertools.product(retrs, (False, True)):
        c = cell(retr, seq)
        if not c:
            continue
        m = {k: float(np.mean([x[k] for x in c]))
             for k in ("style_precision", "style_precision_tail", "edition_cosine", "latency_sec")}
        means[(retr, seq)] = m
        name = f"{retr}+{'순차' if seq else '독립'}"
        print(f"    {name:18s} {len(c):3d} {m['style_precision']:11.3f} "
              f"{m['style_precision_tail']:8.3f} {m['edition_cosine']:11.3f} {m['latency_sec']:6.1f}")

    print("\n    효과 (짝지어 비교 — 같은 그림체·같은 반복끼리 뺀다):")
    for retr in retrs:
        paired_diff(records, retr, "style_precision", "앞 컷 물리기")
    if len(retrs) == 2:
        cross_diff(records, retrs, "style_precision")


def _boot_ci(d, n=10000):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else 0.0, None, None
    rng = np.random.default_rng(0)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _verdict(mean, lo, hi, label):
    if lo is None:
        print(f"      {label}: {mean:+.3f}  (표본이 적어 구간을 못 낸다)")
        return
    v = "효과 있다" if lo > 0 else ("오히려 나빠졌다" if hi < 0 else "주의: 아직 모른다 — 구간이 0을 품는다")
    print(f"      {label}: {mean:+.3f}  95% [{lo:+.3f} ~ {hi:+.3f}]  -> {v}")


def paired_diff(records, retr, key, label):
    """같은 (그림체, 반복) 짝에서 순차 - 독립."""
    d = []
    for r in records:
        if r["retriever"] != retr or r["sequential"]:
            continue
        for s in records:
            if (s["retriever"] == retr and s["sequential"]
                    and s["style"] == r["style"] and s["rep"] == r["rep"]):
                d.append(s[key] - r[key])
    if d:
        _verdict(*_boot_ci(d), f"[{retr}] {label} (n={len(d)})")


def cross_diff(records, retrs, key):
    """검색기 교체 효과 — 생성 방식을 고정하고 비교."""
    a, b = retrs
    for seq in (False, True):
        d = []
        for r in records:
            if r["retriever"] != a or r["sequential"] != seq:
                continue
            for s in records:
                if (s["retriever"] == b and s["sequential"] == seq
                        and s["style"] == r["style"] and s["rep"] == r["rep"]):
                    d.append(s[key] - r[key])
        if d:
            _verdict(*_boot_ci(d), f"검색기 {a}->{b} ({'순차' if seq else '독립'}, n={len(d)})")


if __name__ == "__main__":
    main()
