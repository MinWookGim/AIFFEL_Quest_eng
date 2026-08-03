"""컷 간 consistency 를 **순위 기반**으로 다시 잰다.

v1(절대 코사인 평균)은 실패했다:
    같은 그림체 4장 중앙 0.668  vs  다른 그림체 4장 중앙 0.653  -> 차이 0.015, 분포가 겹침.
    수묵화끼리는 코사인이 전부 0.65 근처라 공통 도메인 신호가 다 먹는다.

그런데 score.py 의 p@5 는 무작위 대비 2.05배가 나온다. 차이는 **절대값이냐 순위냐** 다.
그래서 여기서는 컷마다 코퍼스에서 이웃을 찾아 **라벨을 투표로 붙이고**,
4컷의 라벨이 서로 같은지를 본다. 즉 역방향 채점을 4번 하고 일치를 세는 것이다.

주의: v1 의 버그도 고쳤다: style 배열이 이미 1-based 인데 +1 을 또 해서 이름이 한 칸 밀렸다
  (m1 이 m2 로 찍힘). 이제 라벨을 그대로 쓴다.
"""
import numpy as np

RNG = np.random.default_rng(20260803)
TRIALS = 2000
K = 4      # 한 편의 컷 수
TOPK = 5   # 라벨을 투표로 정할 때 볼 이웃 수


def load():
    E = np.load(__import__("os").environ.get("EMB","verify/ink_emb.npy")).astype("float32")
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E, np.load("verify/ink_style.npy"), np.load("verify/ink_content.npy", allow_pickle=True)


def predict_labels(E, style, content):
    """모든 그림에 대해 '이웃 투표로 매긴 라벨'을 미리 계산해 둔다.

    자기 자신과 **같은 원본**은 후보에서 뺀다 (score.py 와 같은 규칙).
    안 빼면 그림체가 아니라 내용이 같은 걸 찾고도 맞힌 것이 된다.
    """
    S = E @ E.T
    pred = np.empty(len(E), dtype=style.dtype)
    for i in range(len(E)):
        ok = content != content[i]
        ok[i] = False
        sims = S[i].copy()
        sims[~ok] = -np.inf
        top = np.argpartition(-sims, TOPK)[:TOPK]
        top = top[np.argsort(-sims[top])][:TOPK]
        labels, counts = np.unique(style[top], return_counts=True)
        pred[i] = labels[np.argmax(counts)]      # 최다 득표 라벨
    return pred


def sample_same_style(style, content):
    """같은 그림체 4장 (원본은 서로 다르게)."""
    for _ in range(200):
        s = RNG.choice(np.unique(style))
        pool = np.where(style == s)[0]
        if len(pool) < K:
            continue
        seen, idx = set(), []
        for i in RNG.permutation(pool):
            if content[i] in seen:
                continue
            seen.add(content[i]); idx.append(int(i))
            if len(idx) == K:
                return idx, s
    return None, None


def sample_diff_style(style):
    ss = RNG.choice(np.unique(style), size=K, replace=False)
    return [int(RNG.choice(np.where(style == s)[0])) for s in ss]


def edition_scores(pred, idx, requested=None):
    """한 편(4컷)의 점수 두 개.

    all_agree     : 4컷의 예측 라벨이 전부 같은가 (0/1)   <- 엄격
    agree_ratio   : 최다 득표 라벨을 따르는 컷의 비율      <- 부분점수
    style_precision: 요청한 라벨과 맞은 컷의 비율 (요청이 있을 때만)
    """
    labels = pred[idx]
    vals, cnts = np.unique(labels, return_counts=True)
    top = cnts.max()
    out = {"all_agree": int(top == K), "agree_ratio": top / K}
    if requested is not None:
        out["style_precision"] = float((labels == requested).mean())
    return out


def main():
    E, style, content = load()
    print(f"코퍼스 {len(E)}장 / 그림체 {len(np.unique(style))}종 / 원본 {len(np.unique(content))}종")
    print(f"라벨 값: {sorted(np.unique(style).tolist())}")
    pred = predict_labels(E, style, content)
    print(f"컷 단위 라벨 정확도 (이웃 {TOPK}장 투표): {(pred == style).mean():.3f}"
          f"   / 무작위 {1/len(np.unique(style)):.3f}\n")

    same_agree, same_ratio, same_prec = [], [], []
    for _ in range(TRIALS):
        idx, s = sample_same_style(style, content)
        if idx is None:
            continue
        r = edition_scores(pred, idx, requested=s)
        same_agree.append(r["all_agree"]); same_ratio.append(r["agree_ratio"])
        same_prec.append(r["style_precision"])

    diff_agree, diff_ratio = [], []
    for _ in range(TRIALS):
        r = edition_scores(pred, sample_diff_style(style))
        diff_agree.append(r["all_agree"]); diff_ratio.append(r["agree_ratio"])

    def line(name, agree, ratio):
        a, r = np.array(agree), np.array(ratio)
        print(f"  {name:20s} n={len(a):5d}   4컷 전부일치 {a.mean():.3f}   "
              f"일치비율 중앙 {np.median(r):.3f}  5~95% [{np.percentile(r,5):.3f} ~ {np.percentile(r,95):.3f}]")

    print("편 단위 (4컷 묶음):")
    line("(1) 같은 그림체", same_agree, same_ratio)
    line("(2) 다른 그림체", diff_agree, diff_ratio)

    sp = np.array(same_prec)
    print(f"\n  요청 라벨과 일치(style precision) 같은그림체 편: 중앙 {np.median(sp):.3f}"
          f"   5~95% [{np.percentile(sp,5):.3f} ~ {np.percentile(sp,95):.3f}]")

    gap = np.mean(same_agree) - np.mean(diff_agree)
    print(f"\n갈림 정도: 전부일치율 {np.mean(same_agree):.3f} vs {np.mean(diff_agree):.3f}"
          f"  (차이 {gap:+.3f})")
    print("v1(절대 코사인)은 차이 0.015 로 못 갈랐다. 위 차이와 비교할 것.")


if __name__ == "__main__":
    main()


def quartiles():
    """게이트 임계값용 사분위 — 5~95% 만으로는 하한을 못 정한다."""
    import os
    E, style, content = load()
    pred = predict_labels(E, style, content)
    sp, ag = [], []
    for _ in range(TRIALS):
        idx, s = sample_same_style(style, content)
        if idx is None:
            continue
        r = edition_scores(pred, idx, requested=s)
        sp.append(r["style_precision"]); ag.append(r["agree_ratio"])
    sp, ag = np.array(sp), np.array(ag)
    name = os.environ.get("EMB", "?").split("/")[-1]
    print(f"[{name}] 진짜 그림 4장짜리 편 {len(sp)}개")
    for label, a in [("style_precision", sp), ("agree_ratio", ag)]:
        q = np.percentile(a, [10, 25, 50, 75, 90])
        print(f"   {label:16s} 10%={q[0]:.3f} 25%={q[1]:.3f} 중앙={q[2]:.3f} 75%={q[3]:.3f} 90%={q[4]:.3f}")
    print(f"   4컷 중 최소 2컷이 요청 라벨: {(sp>=0.5).mean():.3f}   최소 1컷: {(sp>=0.25).mean():.3f}")
