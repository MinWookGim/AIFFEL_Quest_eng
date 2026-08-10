"""그림체별 천장을 잰다 — 생성 점수를 읽을 자(尺)를 먼저 만드는 스크립트.

왜 필요한가
    `ab_prev_cut.py` 가 내놓는 style_precision 은 절대값으로 읽으면 안 된다.
    8/3 실측에서 **진짜 코퍼스 그림조차 0.500 이 천장**이었다. 생성물이 0.45 를 받았을 때
    "실패"가 아니라 "천장의 90%"일 수 있다는 뜻이다.

    그래서 각 그림체마다 **진짜 그림이 받는 점수**를 먼저 재둔다. 이게 그 그림체의 천장이고,
    생성 점수는 이 천장으로 나눠 읽는다.

    ★ v1(9클래스)에서 잰 옛 배수(m3 3.34 / m2 3.00 / m1 2.26)는 여기서 재사용하지 않는다.
      코퍼스가 daypack_v2(2,522장 / 20클래스)로 바뀌어 눈금 자체가 달라졌다.

무엇을 재나
    코퍼스 그림 한 장을 질의로 던져 이웃 5장을 찾고(자기 자신·같은 원본 제외),
    그 5장의 **최다 라벨**이 원래 라벨과 같은지 본다. `ab_prev_cut.py` 의 채점과 똑같은 방식이라
    나온 숫자를 생성 점수와 바로 나란히 놓을 수 있다.

    같이 나오는 것:
      ceiling      최다 라벨이 맞은 비율 = 그 그림체의 천장 (생성 점수의 분모)
      neighbor_hit 이웃 5장 중 같은 그림체 비율 평균 (더 부드러운 눈금)
      prior        그 그림체가 코퍼스에서 차지하는 비율 = 무작위로 찍었을 때의 기대값
      lift         ceiling / prior

쓰는 법
    python build/style_ceiling.py                      # daypack_v2 · Gram · 전 클래스
    python build/style_ceiling.py --kind clip          # CLIP 으로도 재서 비교
    python build/style_ceiling.py --sample 40          # 클래스당 40장만 (빠르게)
    python build/style_ceiling.py --pack daypack_v1    # 옛 조건 재현
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_prev_cut as ab          # 코퍼스 로딩·임베딩 캐시를 그대로 쓴다 (두 벌 만들지 않는다)

TOPK = 5


def boot_ci(x, n=10000, seed=0):
    """부트스트랩 95% 구간. 표본이 2개 미만이면 구간을 안 낸다."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else 0.0), None, None
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def measure(E, style, content, target, idxs):
    """진짜 그림들을 질의로 던져 그 그림체의 천장을 잰다.

    ★ 자기 자신과 **같은 원본**은 이웃 후보에서 뺀다. 안 빼면 같은 그림의 다른 판본을
      찾아놓고 그림체를 맞혔다고 착각한다 (ab_prev_cut 채점과 같은 규칙).
    ★★ **group 이 빈 값이면 안 묶는다** (`kit/score.py` 40행 규칙).
       daypack_v2 의 vec_* 698장은 group 이 전부 '' 이라, 빈 값을 같은 원본으로 보면
       vec 다섯 클래스가 통째로 후보에서 빠져 천장이 0.000 으로 나온다 (2026-08-06 실측으로 잡음).
    """
    correct, hits = [], []
    for q in idxs:
        sims = E @ E[q]
        banned = (content == content[q]) & (content != "")   # 빈 값은 묶지 않는다
        banned[q] = True                                     # 자기 자신은 언제나 제외
        sims[banned] = -np.inf
        top = np.argsort(-sims)[:TOPK]
        lab, cnt = np.unique(style[top], return_counts=True)
        correct.append(float(lab[np.argmax(cnt)] == target))
        hits.append(float((style[top] == target).mean()))
    return correct, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="daypack_v2")
    ap.add_argument("--kind", default="gram", choices=["gram", "clip"],
                    help="채점 임베딩. 8/3 실측 결론대로 기본은 gram")
    ap.add_argument("--sample", type=int, default=0,
                    help="클래스당 질의 장수 (0 = 전부). 표본이 줄면 구간이 넓어진다")
    ap.add_argument("--styles", nargs="+", default=None, help="비우면 전 클래스")
    a = ap.parse_args()

    ab.PACK_NAME = a.pack
    ab.PACK = os.path.join(ab.ROOT, a.pack)
    if not os.path.isdir(ab.PACK):
        raise SystemExit(f"코퍼스 폴더가 없다: {ab.PACK}")

    paths, style, content = ab.load_pack()
    n_class = len(np.unique(style))
    print(f"코퍼스 {a.pack} — {len(paths)}장 / {n_class}클래스 / 채점 {a.kind}")

    E = ab.get_embeddings(a.kind, paths)

    targets = a.styles or sorted(set(str(x) for x in style))
    rng = np.random.default_rng(20260806)

    rows = []
    print(f"\n{'그림체':<26} {'n':>4} {'천장':>7} {'95% 구간':>18} {'이웃적중':>8} {'무작위':>7} {'배수':>6}")
    for t in targets:
        pool = np.where(style == t)[0]
        if len(pool) == 0:
            print(f"    {t}: 없는 클래스라 건너뛴다")
            continue
        idxs = pool if not a.sample else rng.choice(pool, min(a.sample, len(pool)), replace=False)
        correct, hits = measure(E, style, content, t, idxs)
        m, lo, hi = boot_ci(correct)
        prior = len(pool) / len(style)       # 무작위로 찍었을 때 이 라벨이 나올 기대값
        ci = f"[{lo:.3f} ~ {hi:.3f}]" if lo is not None else "(표본 부족)"
        print(f"{t:<26} {len(idxs):>4} {m:>7.3f} {ci:>18} "
              f"{np.mean(hits):>8.3f} {prior:>7.4f} {m/prior:>6.2f}")
        rows.append({"style": t, "n": int(len(idxs)), "ceiling": m,
                     "ci95_low": lo, "ci95_high": hi,
                     "neighbor_hit": float(np.mean(hits)),
                     "prior": prior, "lift": m / prior})

    out = os.path.join(ab.ROOT, "verify", f"style_ceiling_{a.pack}_{a.kind}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"pack": a.pack, "kind": a.kind, "topk": TOPK,
               "n_images": len(paths), "n_class": n_class, "rows": rows},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n저장: {out}")

    if rows:
        best = sorted(rows, key=lambda r: -r["ceiling"])[:5]
        print("\n천장이 높은 순 (ab_prev_cut --styles 후보):")
        for r in best:
            print(f"    {r['style']:<26} 천장 {r['ceiling']:.3f}  배수 {r['lift']:.2f}")
        print("\n★ 생성 점수는 이 천장으로 나눠 읽는다. 절대값끼리 비교하지 말 것.")


if __name__ == "__main__":
    main()
