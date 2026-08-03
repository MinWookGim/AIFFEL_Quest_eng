"""채점기를 여러 개 모아 투표시키면 나아지나 — 실측한다.

투표(앙상블)가 먹히려면 조건이 하나 있다. **서로 다른 데서 틀려야 한다.**
다 같은 그림에서 틀리면 몇 개를 모아도 결과가 같다. 그래서 먼저 오류가 겹치는지부터 본다.

방법 두 가지를 비교한다
  hard voting : 각자 라벨을 하나씩 내고 다수결        (동점이면 판정 보류)
  soft voting : 유사도를 정규화해 더한 뒤 그걸로 이웃을 찾음 (순위 융합)
"""
import numpy as np

TOPK = 5


def load(path):
    E = np.load(path).astype("float32")
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


def sim_matrix(E, content):
    """자기 자신과 같은 원본은 후보에서 뺀 유사도 행렬."""
    S = E @ E.T
    same = content[:, None] == content[None, :]
    S[same] = -np.inf
    return S


def predict(S, style):
    """이웃 TOPK 장의 최다 득표 라벨."""
    pred = np.empty(len(S), dtype=style.dtype)
    for i in range(len(S)):
        top = np.argpartition(-S[i], TOPK)[:TOPK]
        top = top[np.argsort(-S[i][top])][:TOPK]
        lab, cnt = np.unique(style[top], return_counts=True)
        pred[i] = lab[np.argmax(cnt)]
    return pred


def zscore(S):
    """채점기마다 유사도 스케일이 다르다. 더하려면 같은 자로 맞춰야 한다.
    -inf(제외 대상)는 건드리지 않는다."""
    ok = np.isfinite(S)
    m, s = S[ok].mean(), S[ok].std()
    Z = np.full_like(S, -np.inf)
    Z[ok] = (S[ok] - m) / (s + 1e-8)
    return Z


def main():
    style = np.load("verify/ink_style.npy")
    content = np.load("verify/ink_content.npy", allow_pickle=True)
    n_class = len(np.unique(style))

    scorers = {}
    for name, path in [("CLIP", "verify/ink_emb.npy"), ("Gram", "verify/ink_gram.npy")]:
        scorers[name] = sim_matrix(load(path), content)

    preds = {k: predict(S, style) for k, S in scorers.items()}
    hits = {k: (p == style) for k, p in preds.items()}

    print(f"코퍼스 {len(style)}장 / {n_class}클래스 / 무작위 {1/n_class:.3f}\n")
    print("단독 성능 (컷 단위 라벨 정확도):")
    for k, h in hits.items():
        print(f"  {k:6s} {h.mean():.3f}")

    # ── 오류가 겹치나 ─────────────────────────────
    a, b = hits["CLIP"], hits["Gram"]
    both = (a & b).mean()
    only_a = (a & ~b).mean()
    only_b = (~a & b).mean()
    neither = (~a & ~b).mean()
    print(f"\n오류 겹침 (앙상블이 먹힐 조건):")
    print(f"  둘 다 맞힘      {both:.3f}")
    print(f"  CLIP 만 맞힘    {only_a:.3f}   <- Gram 이 놓친 것을 CLIP 이 건짐")
    print(f"  Gram 만 맞힘    {only_b:.3f}   <- 그 반대")
    print(f"  둘 다 틀림      {neither:.3f}  <- 여기는 투표로 못 구한다")
    print(f"\n  한쪽이라도 맞힌 비율(이론상 상한) {both+only_a+only_b:.3f}")
    print(f"  단독 최고                        {max(h.mean() for h in hits.values()):.3f}")
    print(f"  -> 투표로 건질 수 있는 여지       {both+only_a+only_b - max(h.mean() for h in hits.values()):+.3f}")

    # ── soft voting: 유사도를 z 정규화해 더한다 ──
    Z = sum(zscore(S) for S in scorers.values())
    pred_soft = predict(Z, style)
    print(f"\nsoft voting (유사도 순위 융합) {(pred_soft == style).mean():.3f}")

    # 가중치를 바꿔가며 — Gram 이 더 강하니 비중을 달리 줘 본다
    print("  가중치별 (CLIP:Gram):")
    for w in [0.25, 0.5, 1.0, 2.0, 4.0]:
        Zw = zscore(scorers["CLIP"]) + w * zscore(scorers["Gram"])
        print(f"    1:{w:<4} {(predict(Zw, style) == style).mean():.3f}")

    # ── hard voting 은 2표라 동점이 문제 ──────────
    tie = (preds["CLIP"] != preds["Gram"]).mean()
    print(f"\nhard voting: 두 채점기가 다른 답을 내는 비율 {tie:.3f}")
    print("  -> 2표는 동점이 나면 못 정한다. 다수결을 하려면 채점기가 최소 3개 필요하다.")


if __name__ == "__main__":
    main()
