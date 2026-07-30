"""두 방법 중 어느 쪽이 진짜 나은지 판정한다 — 짝지어 비교.

쓰는 법
    python compare.py "CLIP 기준선" "Gram VGG19 (4층)"
    (이름은 score.py 에 --name 으로 준 그 이름. runs/ 폴더의 파일을 찾는다)

왜 이게 따로 필요한가
    score.py 가 찍어주는 95% 구간은 **한 방법이 혼자 흔들리는 폭**이다.
    두 방법을 비교할 때 그 구간을 겹쳐보는 건 지나치게 보수적이다 —
    **두 방법이 같은 그림들로 채점받았다는 사실**을 안 쓰기 때문이다.

    같은 질의에서 A와 B의 점수를 **짝지어** 빼면, 그림 난이도 때문에 생기는 출렁임이
    상쇄되고 **방법의 차이만** 남는다. 그래서 훨씬 예민하게 갈린다.
    (어려운 그림은 둘 다 못 맞히고, 쉬운 그림은 둘 다 맞힌다. 그 공통분모를 지우는 것)

판정
    차이의 95% 구간이 **0을 포함하면 "아직 모른다"**. 포함하지 않으면 이긴 것이다.
"""
import os, sys, glob, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
BOOT = 10000


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name)[:40]


def load(name):
    p = os.path.join(RUNS, f"{slug(name)}.npy")
    if not os.path.exists(p):
        have = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{RUNS}/*.npy"))]
        raise SystemExit(f"'{name}' 기록이 없다.\n  있는 것: {have}")
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="방법 A 이름")
    ap.add_argument("b", help="방법 B 이름")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    if len(A) != len(B):
        raise SystemExit(f"질의 수가 다르다 ({len(A)} vs {len(B)}). "
                         "★코퍼스가 다른 두 기록은 비교할 수 없다 (규칙 2)")

    d = B - A                      # 질의별 차이
    mean_d = float(d.mean())

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\nA  {args.a:28s} p@5 {A.mean():.4f}")
    print(f"B  {args.b:28s} p@5 {B.mean():.4f}")
    print(f"\n차이 (B - A)            {mean_d:+.4f}")
    print(f"차이의 95% 구간          {lo:+.4f} ~ {hi:+.4f}")

    # 몇 개 질의에서 뒤집혔나 — 평균만 보면 안 보이는 것
    win, lose, tie = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    print(f"질의별 승패             B승 {win} / A승 {lose} / 무 {tie}")

    if lo > 0:
        print(f"\n판정: **B 가 이겼다.** (구간이 0 위에 있다)")
    elif hi < 0:
        print(f"\n판정: **A 가 이겼다.** (구간이 0 아래에 있다)")
    else:
        print(f"\n판정: **아직 모른다.** 구간이 0을 품는다 — 이 차이는 출렁임으로 설명된다")

    json.dump({"A": args.a, "B": args.b, "meanA": float(A.mean()), "meanB": float(B.mean()),
               "diff": mean_d, "ci95": [float(lo), float(hi)],
               "win": win, "lose": lose, "tie": tie,
               "verdict": "B" if lo > 0 else ("A" if hi < 0 else "무승부")},
              open(os.path.join(RUNS, f"compare_{slug(args.a)}_vs_{slug(args.b)}.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
