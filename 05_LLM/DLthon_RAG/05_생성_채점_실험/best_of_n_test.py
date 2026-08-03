"""크리틱을 '고르는 데' 쓰면 쓸 만한가 — best-of-N 실측.

배경
    크리틱(Gram)의 컷 단위 라벨 정확도는 0.440 이다. 진짜 m3 그림의 56%를 "아니다"로 잘못 짚는다.
    그래서 **버리는 데(재생성 루프)** 쓰면 잘 그린 것을 버린다.
    반면 **고르는 데(선택 루프)** 쓰면 틀려도 최악이 무작위 뽑기다. 밑질 게 없다.

무엇을 재나
    "m3 로 그려줘" 하고 N장을 받았다고 치자. 그중 진짜 m3 는 한 장뿐이고 나머지는 딴 그림체다.
    크리틱이 m3 점수가 제일 높은 것을 고른다. **진짜 m3 를 고를 확률**이 무작위(1/N)보다 높은가.

    * 코퍼스 그림으로 시뮬레이션한다. 생성 API 를 안 쓴다.
    * 크리틱 점수 = 그 그림의 이웃 TOPK 장 중 목표 라벨(m3)의 비율. score.py 와 같은 방식.
"""
import numpy as np

TOPK = 5
TRIALS = 3000
RNG = np.random.default_rng(20260803)


def load(path, content):
    E = np.load(path).astype("float32")
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S = E @ E.T
    S[content[:, None] == content[None, :]] = -np.inf   # 자기·같은 원본 제외
    return S


def critic_scores(S, style, targets):
    """모든 그림 x 모든 라벨에 대해 '그 라벨다움' 점수를 미리 계산한다."""
    out = np.zeros((len(S), len(targets)), dtype="float32")
    t_index = {t: j for j, t in enumerate(targets)}
    for i in range(len(S)):
        top = np.argpartition(-S[i], TOPK)[:TOPK]
        top = top[np.argsort(-S[i][top])][:TOPK]
        lab, cnt = np.unique(style[top], return_counts=True)
        for l, c in zip(lab, cnt):
            out[i, t_index[l]] = c / TOPK
    return out


def run(name, S, style):
    targets = np.unique(style)
    CS = critic_scores(S, style, targets)
    t_index = {t: j for j, t in enumerate(targets)}

    for N in (2, 3, 5):
        win = 0
        for _ in range(TRIALS):
            target = RNG.choice(targets)
            j = t_index[target]
            # 진짜 목표 그림 1장 + 딴 그림체 N-1장
            real = int(RNG.choice(np.where(style == target)[0]))
            others = [int(RNG.choice(np.where(style == s)[0]))
                      for s in RNG.choice(targets[targets != target], size=N - 1, replace=False)]
            cand = [real] + others
            scores = CS[cand, j]
            # 동점이면 무작위로 하나 — 크리틱이 못 가른 것을 맞힌 걸로 치면 안 된다
            best = int(RNG.choice(np.flatnonzero(scores == scores.max())))
            win += (cand[best] == real)
        print(f"  {name:5s} N={N}   진짜를 고를 확률 {win/TRIALS:.3f}   "
              f"무작위 {1/N:.3f}   배수 {(win/TRIALS)/(1/N):.2f}")


def main():
    style = np.load("verify/ink_style.npy")
    content = np.load("verify/ink_content.npy", allow_pickle=True)
    print(f"코퍼스 {len(style)}장 / {len(np.unique(style))}클래스\n")
    print("후보 N장 중 목표 그림체가 1장. 크리틱이 그걸 골라내나:")
    for name, path in [("CLIP", "verify/ink_emb.npy"), ("Gram", "verify/ink_gram.npy")]:
        run(name, load(path, content), style)


if __name__ == "__main__":
    main()
