"""컷1 을 먼저 보증하면 순차의 정확도 손해가 사라지나 — 3조건 실험.

★ v1 동결본을 건드리지 않는다. 결과만 `verify/cut1_guard/` 에 쌓는다.
★ 코퍼스·검색·채점·구간 코드는 `ab_prev_cut.py` 를 그대로 import 한다.
  두 실험의 채점 규칙이 갈리면 숫자를 나란히 못 놓기 때문이다. 여기서 새로 짜는 건
  "컷1 을 여러 장 뽑아 고르는 부분" 하나뿐이다.

무엇을 묻나
    8/6 ab_prev_cut 결론: 앞 컷을 물리면(순차) 일관성은 확실히 오르고(+0.1348, 7/7 상승)
    정확도는 확실히 내린다(-0.1429, 0/7 상승).

    그런데 순차는 한 번에 두 가지 일을 한다.
        (a) 4컷을 서로 닮게 만든다        -> 일관성 이득
        (b) 컷1 의 오차를 나머지 셋에 복제한다 -> 정확도 손해
    관측된 -0.1429 는 (a)와 (b)가 **엉킨 채로 잰 합계**다.

    가설: (b)만 떼어내면 (a)만 남는다.
          즉 컷1 을 best-of-N 으로 보증해 컷1 의 오차를 줄이면
          **순차의 정확도 손해가 사라지거나 크게 줄고, 일관성 이득은 그대로 남는다.**

    근거 둘
      1. 손해의 출처가 그림으로 보인다 — `verify/ab_prev_cut/ab_compare_vec_undraw.png`
         순차 실패 편은 컷1 이 목표 그림체가 아니었고 나머지 셋이 그 어긋남을 충실히 베꼈다.
      2. 고르는 도구가 이미 실측돼 있다 — `verify/best_of_n_test.py`
         크리틱은 판정기로 쓰면 0.440(56% 오판)이지만 **선별기로 쓰면 0.704**.

조건 3개 (같은 그림체·같은 반복·같은 질의그림에서)
    ind      독립                          컷마다 레퍼런스만 (기존 기본값)
    seq      순차                          컷2~4 에 컷1 을 물림 (기존, 컷1 무보증)
    bon_seq  컷1 best-of-N -> 선별 -> 순차  ★ 새 조건

    핵심 비교는 bon_seq - seq 다. ind 는 눈금으로 같이 돌린다.

★★ 이 실험에서 제일 조용히 틀리기 쉬운 곳 — 대표 지표를 컷2~4 로 둔다

    bon_seq 의 컷1 은 **채점기가 고른 그림**이다. 그 컷1 을 같은 채점기로 채점하면
    당연히 점수가 높다. 즉 bon_seq 의 `style_precision`(4컷 전체)은 **정직한 표본이 아니다.**
    (부정행위는 아니다 — 서비스도 목표 그림체를 알고 고를 테니 선별은 시스템의 일부다.
     다만 그 컷을 "측정값"으로 쓰면 안 된다는 뜻이다.)

    반면 컷2~4 는 아무도 고르지 않았다. 고른 컷1 을 조건으로 그려졌을 뿐이다.
    -> **대표 지표 = `style_precision_tail`(컷2~4).** 4컷 전체 값도 찍되 참고로만 읽는다.

선별기
    후보 컷1 을 코퍼스에 질의로 던져 이웃 5장 중 목표 라벨의 비율이 제일 높은 것을 고른다.
    `verify/best_of_n_test.py` 가 쓴 크리틱 점수와 같은 정의다.
    ★ 동점이면 무작위로 고른다. 크리틱이 못 가른 것을 맞힌 걸로 치면 안 된다(같은 파일의 규칙).

그림체 3종 (천장 = `verify/style_ceiling_daypack_v2_gram.json`)
    vec_undraw     0.979  쉬운 판. 8/6 에 신호가 사실상 여기서만 나왔다
    paint_Baroque  0.713  ★ 새로 넣음. 8/6 은 vec+ink 뿐이라 도메인이 둘이었다
    ink_m3         0.657  어려운 판. 8/6 에 네 조건 전부 0.000 이라 바닥에 눌렸다
                          -> 그래도 남긴다. 보증한 컷1 로도 0.000 이면 "병목이 앵커가 아니라
                             생성 능력"이라는 답이 되고, 8/6 과 같은 그림체라 비교도 된다

쓰는 법
    python build/cut1_guard.py --dry-run                  # 계획·예상비용만 (API 안 부름)
    python build/cut1_guard.py                            # 실제 실행 (중간에 끊겨도 이어짐)
    python build/cut1_guard.py --analyze-only             # 뽑아둔 결과만 다시 분석
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ab_prev_cut as ab            # noqa: E402  — 코퍼스·검색·채점·구간을 그대로 쓴다

OUT = os.path.join(ROOT, "verify", "cut1_guard")
MODES = ("ind", "seq", "bon_seq")
MODE_NAME = {"ind": "독립", "seq": "순차", "bon_seq": "컷1보증+순차"}


# ─────────────────────────────────────────────────────────────
# 1. 선별기 — 크리틱을 "고르는 데" 쓴다
# ─────────────────────────────────────────────────────────────
def banned_mask(style, content, ref_idx):
    """레퍼런스로 물린 그림과 같은 원본을 이웃 후보에서 뺀다.

    ★ `ab.score_edition` 안의 규칙과 **같아야 한다.** 선별과 채점이 다른 판을 보면
      "고를 땐 높았는데 잴 땐 낮은" 일이 생기고 원인을 못 찾는다.
    ★★ group 이 빈 값이면 안 묶는다 (`kit/score.py` 40행. v2 의 vec_* 698장이 빈 값이다).
    """
    banned = np.zeros(len(style), dtype=bool)
    for i in ref_idx:
        if content[i]:
            banned |= (content == content[i])
        banned[i] = True
    return banned


def critic_score(img, Eg, style, banned, target):
    """이 그림이 목표 그림체'다움' 점수. 이웃 TOPK 장 중 목표 라벨의 비율."""
    v = ab.encode_one("gram", img)
    sims = Eg @ v
    sims[banned] = -np.inf
    top = np.argsort(-sims)[:ab.TOPK]
    return float((style[top] == target).mean())


# ─────────────────────────────────────────────────────────────
# 2. 생성
# ─────────────────────────────────────────────────────────────
def gen_one(client, imgs, prompt):
    """그림 한 장. ab_prev_cut 과 같은 모델·크기·품질을 쓴다."""
    t0 = time.time()
    r = client.images.edit(
        model=ab.MODEL, size=ab.SIZE, quality=ab.QUALITY,
        image=[ab.to_file(x, f"r{i}.png") for i, x in enumerate(imgs)],
        prompt=prompt)
    dt = time.time() - t0
    img = Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
    return img, {"초": round(dt, 1), **ab.cost_of(getattr(r, "usage", None))}


def make_edition(client, ref_imgs, scenes, mode, ctx, rng):
    """4컷 한 편. mode 에 따라 컷1 을 만드는 방식만 갈린다.

    반환 diag 에 후보 점수를 전부 남긴다. **선별기가 실제로 무슨 일을 했는지**
    나중에 확인하려면 이게 있어야 한다 (후보 점수가 전부 같으면 그냥 무작위로 고른 것이다).
    """
    Eg, style, content, ref_idx, target, n_best = ctx
    costs, diag = [], {}

    if mode == "bon_seq":
        banned = banned_mask(style, content, ref_idx)
        cands, cand_costs = [], []
        for _ in range(n_best):
            img, c = gen_one(client, ref_imgs, ab.PROMPT.format(scene=scenes[0]))
            cands.append(img)
            cand_costs.append(c)
        costs.extend(cand_costs)
        sc = [critic_score(im, Eg, style, banned, target) for im in cands]
        best = float(np.max(sc))
        # ★ 동점이면 무작위. 크리틱이 못 가른 것을 맞힌 걸로 치면 안 된다
        pick = int(rng.choice(np.flatnonzero(np.asarray(sc) == best)))
        diag = {
            "cut1_cand_scores": [round(s, 3) for s in sc],
            "cut1_picked": pick,
            "cut1_spread": round(float(max(sc) - min(sc)), 3),   # 0 이면 선별기가 못 갈랐다
            "cut1_tied": bool(sum(s == best for s in sc) > 1),
        }
        cuts = [cands[pick]]
        diag["_cands"] = cands          # 저장용. 기록에는 안 남긴다
        sequential = True
    else:
        img, c = gen_one(client, ref_imgs, ab.PROMPT.format(scene=scenes[0]))
        costs.append(c)
        cuts = [img]
        sequential = (mode == "seq")

    for scene in scenes[1:]:
        imgs = list(ref_imgs) + [cuts[0]] if sequential else list(ref_imgs)
        prompt = (ab.PROMPT_SEQ if sequential else ab.PROMPT).format(scene=scene)
        img, c = gen_one(client, imgs, prompt)
        costs.append(c)
        cuts.append(img)

    return cuts, costs, diag


# ─────────────────────────────────────────────────────────────
# 3. 실험 진행
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="daypack_v2")
    ap.add_argument("--styles", nargs="+",
                    default=["vec_undraw", "paint_Baroque", "ink_m3"])
    ap.add_argument("--reps", type=int, default=3, help="그림체당 반복 횟수")
    ap.add_argument("--retriever", default="gram",
                    help="검색기 하나로 고정한다. 8/6 실측에서 clip->gram 교체 효과가 "
                         "안 보였으므로, 조건 수를 늘리는 대신 그 예산을 반복에 쓴다")
    ap.add_argument("--n-best", type=int, default=3, help="컷1 후보 장수 (best-of-N 의 N)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    ab.PACK_NAME = a.pack
    ab.PACK = os.path.join(ROOT, a.pack)
    if not os.path.isdir(ab.PACK):
        raise SystemExit(f"코퍼스 폴더가 없다: {ab.PACK}")

    os.makedirs(OUT, exist_ok=True)
    results_path = os.path.join(OUT, "results.json")

    paths, style, content = ab.load_pack()
    missing = [s for s in a.styles if s not in set(style)]
    if missing:
        raise SystemExit(f"{a.pack} 에 없는 그림체다: {missing}")

    per_pair = ab.N_CUTS * 2 + (a.n_best + ab.N_CUTS - 1)   # ind 4 + seq 4 + bon_seq (N+3)
    n_pairs = len(a.styles) * a.reps
    n_images = per_pair * n_pairs
    est = n_images * 0.046

    print(f"코퍼스 {a.pack} — {len(paths)}장 / {len(np.unique(style))}클래스")
    print(f"그림체 {len(a.styles)}종 x 반복 {a.reps}회 = {n_pairs}쌍, 쌍마다 조건 3개")
    print(f"    ind {ab.N_CUTS}장 + seq {ab.N_CUTS}장 + bon_seq {a.n_best + ab.N_CUTS - 1}장 "
          f"(컷1 후보 {a.n_best} + 컷2~4) = 쌍당 {per_pair}장")
    print(f"이미지 {n_images}장 / 예상 비용 약 ${est:.2f} / 예상 시간 약 {n_images*19.2/60:.0f}분")
    print(f"검색기 {a.retriever} 고정 · 대표 지표 = style_precision_tail(컷2~4)")
    if a.dry_run:
        print("\n--dry-run 이라 여기서 멈춘다.")
        return

    print("\n[1] 임베딩")
    E = {k: ab.get_embeddings(k, paths) for k in {a.retriever, "gram"}}
    Eg = E["gram"]

    scenes = ab.load_scenes()
    rng = np.random.default_rng(20260809)

    if not a.analyze_only:
        from openai import OpenAI
        client = OpenAI(api_key=open(os.path.expanduser("~/.config/openai/api_key")).read().strip())

        records = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else []
        done = {(r["style"], r["rep"], r["mode"]) for r in records}

        print("\n[2] 생성")
        for target in a.styles:
            pool = np.where(style == target)[0]
            for rep in range(a.reps):
                # ★ 질의 그림은 세 조건이 **같은 것**을 쓴다. 안 그러면 짝지어 비교가 성립 안 한다
                q = int(rng.choice(pool))
                ref_idx, ref_hit = ab.retrieve_refs(E[a.retriever], style, content, target, q)
                ref_imgs = [Image.open(paths[i]).convert("RGB") for i in ref_idx]
                ctx = (Eg, style, content, ref_idx, target, a.n_best)

                for mode in MODES:
                    if (target, rep, mode) in done:
                        print(f"    건너뜀 (이미 있음) {target} rep{rep} {mode}")
                        continue
                    tag = f"{target}_rep{rep}_{mode}"
                    print(f"    {tag}  (레퍼런스 {ref_hit}/{ab.N_REF} 장이 실제 {target})", flush=True)
                    try:
                        cuts, costs, diag = make_edition(client, ref_imgs, scenes, mode, ctx, rng)
                    except Exception as e:
                        print(f"      !! 실패: {repr(e)[:160]}")
                        continue

                    for im, k in zip(diag.pop("_cands", []), range(1, 99)):
                        im.save(os.path.join(OUT, f"{tag}_cand{k}.png"))
                    for k, c in enumerate(cuts, 1):
                        c.save(os.path.join(OUT, f"{tag}_cut{k}.png"))

                    sc = ab.score_edition(cuts, Eg, style, content, target, ref_idx)
                    rec = {"mode": mode, "style": target, "rep": rep,
                           "retriever": a.retriever, "query_idx": q,
                           "ref_idx": ref_idx, "ref_hit": ref_hit,
                           "panels_produced": len(cuts),
                           "latency_sec": round(sum(c["초"] for c in costs), 1),
                           "cost_usd": round(sum(c.get("달러", 0) for c in costs), 4),
                           **diag, **sc}
                    records.append(rec)
                    json.dump(records, open(results_path, "w"), ensure_ascii=False, indent=1)
                    extra = ""
                    if mode == "bon_seq":
                        extra = (f"  [컷1 후보 {rec['cut1_cand_scores']} -> #{rec['cut1_picked']}"
                                 f"{' 동점무작위' if rec['cut1_tied'] else ''}]")
                    print(f"      컷2~4 {sc['style_precision_tail']:.2f} "
                          f"(4컷 {sc['style_precision']:.2f}) "
                          f"컷간코사인 {sc['edition_cosine']:.3f}  ${rec['cost_usd']:.3f}{extra}")

    print("\n[3] 분석")
    analyze(json.load(open(results_path, encoding="utf-8")))


# ─────────────────────────────────────────────────────────────
# 4. 분석
# ─────────────────────────────────────────────────────────────
def paired(records, key, mode_b, mode_a):
    """같은 (그림체, 반복) 짝에서 mode_b - mode_a."""
    idx = {(r["style"], r["rep"], r["mode"]): r for r in records}
    d = []
    for (s, rep, m), r in idx.items():
        if m != mode_a:
            continue
        o = idx.get((s, rep, mode_b))
        if o is not None:
            d.append(o[key] - r[key])
    return d


def analyze(records):
    if not records:
        print("    결과가 없다.")
        return
    ab.add_domain_precision(records)

    print(f"\n    {'조건':16s} {'n':>3s} {'컷2~4★':>8s} {'4컷':>7s} "
          f"{'domain':>8s} {'컷간코사인':>11s} {'초':>6s} {'$':>7s}")
    for mode in MODES:
        c = [r for r in records if r["mode"] == mode]
        if not c:
            continue
        m = {k: float(np.mean([x[k] for x in c]))
             for k in ("style_precision_tail", "style_precision", "domain_precision",
                       "edition_cosine", "latency_sec", "cost_usd")}
        print(f"    {MODE_NAME[mode]:16s} {len(c):3d} {m['style_precision_tail']:8.3f} "
              f"{m['style_precision']:7.3f} {m['domain_precision']:8.3f} "
              f"{m['edition_cosine']:11.3f} {m['latency_sec']:6.1f} {m['cost_usd']:7.3f}")

    print("\n    그림체별 컷2~4:")
    for tgt in sorted({r["style"] for r in records}):
        row = []
        for mode in MODES:
            c = [r for r in records if r["mode"] == mode and r["style"] == tgt]
            row.append(f"{MODE_NAME[mode]} {np.mean([x['style_precision_tail'] for x in c]):.3f}"
                       if c else "-")
        print(f"      {tgt:<16} " + "   ".join(row))

    print("\n    ★ 대표 지표 = 컷2~4 정확도 (bon_seq 의 컷1 은 채점기가 고른 것이라 뺀다)")
    for b, a_, label in (("seq", "ind", "순차 - 독립           (8/6 재현)"),
                         ("bon_seq", "seq", "컷1보증 - 순차        ★ 가설의 핵심"),
                         ("bon_seq", "ind", "컷1보증+순차 - 독립   (넘었나)")):
        d = paired(records, "style_precision_tail", b, a_)
        if d:
            ab._verdict(*ab._boot_ci(d), f"{label} (n={len(d)})")

    print("\n    일관성 (컷간코사인) — 보증해도 일관성 이득이 남아 있나")
    for b, a_, label in (("seq", "ind", "순차 - 독립        "),
                         ("bon_seq", "ind", "컷1보증+순차 - 독립")):
        d = paired(records, "edition_cosine", b, a_)
        if d:
            ab._verdict(*ab._boot_ci(d), f"{label} (n={len(d)})")

    bon = [r for r in records if r["mode"] == "bon_seq" and "cut1_spread" in r]
    if bon:
        tied = sum(r["cut1_tied"] for r in bon)
        print(f"\n    선별기가 실제로 갈랐나: {len(bon)}편 중 동점(무작위 선택) {tied}편, "
              f"후보 점수 폭 평균 {np.mean([r['cut1_spread'] for r in bon]):.3f}")
        print("      (폭이 0 에 가까우면 크리틱이 후보를 못 가른 것이다 — 그러면 이 실험은"
              " best-of-N 을 잰 게 아니라 그냥 컷1 을 한 번 더 뽑은 것에 가깝다)")


if __name__ == "__main__":
    main()
