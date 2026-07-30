# -*- coding: utf-8 -*-
"""수경님 제안(Macro mAP@5) 검토용 — 지금 지표(micro p@5)와 나란히 실측 비교.

질문: 지표를 바꾸면 방법들의 순위가 바뀌는가? (바뀌면 지표 선택이 중대사, 안 바뀌면 취향 문제)

- micro p@5  : 지금 리더보드 (질의=그림 한 장씩, 전부 평균 -> 큰 클래스가 지배)
- Macro p@5  : 클래스별 평균을 먼저 내고 그 평균 (클래스 한 표씩 -> 불균형 보정)
- Macro mAP@5: 상위 5장 안에서 '맞는 걸 앞에 놨는지' 순서까지 반영한 AP@5 를 클래스별 평균
누수 규칙은 score.py 와 동일(자기자신 + 같은 group 제외).
"""
import os, sys, csv, json, importlib.util
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "kit")
PACK = os.path.join(ROOT, "daypack_v2")
K = 5

rows = list(csv.DictReader(open(os.path.join(PACK, "meta.csv"), encoding="utf-8")))
print(f"코퍼스 {len(rows)}장 로드 중...")
imgs = [Image.open(os.path.join(PACK, r["file"])).convert("RGB") for r in rows]
style = np.array([r["style"] for r in rows])
group = np.array([r["group"] or "" for r in rows])

def load_encode(path):
    spec = importlib.util.spec_from_file_location("enc_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.encode

def metrics(E):
    # score.py 의 precision_at_k 와 같은 규칙 + 순서 보존해서 AP@5 까지
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    N = len(E)
    p5 = np.zeros(N)
    ap5 = np.zeros(N)
    for i in range(N):
        ok = (group != group[i]) | (group == "")
        ok[i] = False
        sims = E @ E[i]
        sims[~ok] = -np.inf
        top = np.argpartition(-sims, K)[:K]
        top = top[np.argsort(-sims[top])]              # 유사도 순 정렬 = 순위
        rel = (style[top] == style[i]).astype(float)   # 순위별 정답 여부
        p5[i] = rel.mean()
        # AP@5: 정답이 나온 순위 r 마다 그 시점 정밀도 P(r) 를 더해 5로 나눔 (정답 풀은 5개 이상)
        cum = np.cumsum(rel)
        ranks = np.arange(1, K + 1)
        ap5[i] = float((rel * cum / ranks).sum() / K)
    return p5, ap5

def macro(v):
    return float(np.mean([v[style == s].mean() for s in sorted(set(style))]))

METHODS = [
    ("CLIP 기준선", "encoders/clip_base.py"),
    ("민욱-흑백CLIP", "encoders/clip_gray.py"),
    ("민욱-패치셔플CLIP", "encoders/clip_patchshuffle.py"),
    ("Gram VGG19", "encoders/gram_vgg19.py"),
]

# 리더보드의 micro p@5 와 대조해 계산이 같은 규칙인지 검증
lb = {r["이름"]: float(r["p@5"]) for r in csv.DictReader(open(os.path.join(KIT, "leaderboard.csv"), encoding="utf-8"))
      if r["코퍼스"] == "daypack_v2"}
lb_map = {"CLIP 기준선": "CLIP 기준선 v2", "Gram VGG19": "Gram VGG19 v2"}

out = {}
print(f"\n{'방법':22s} {'micro p@5':>10s} {'Macro p@5':>10s} {'Macro mAP@5':>12s}")
for name, rel_path in METHODS:
    encode = load_encode(os.path.join(KIT, rel_path))
    E = np.asarray(encode(imgs), dtype="float32")
    p5, ap5 = metrics(E)
    micro = float(p5.mean())
    board = lb.get(lb_map.get(name, name))
    if board is not None:
        assert abs(micro - board) < 5e-4, f"{name}: micro {micro:.4f} != 리더보드 {board:.4f} (규칙 어긋남)"
    out[name] = {"micro_p5": round(micro, 4), "macro_p5": round(macro(p5), 4),
                 "macro_map5": round(macro(ap5), 4)}
    r = out[name]
    print(f"{name:22s} {r['micro_p5']:>10.4f} {r['macro_p5']:>10.4f} {r['macro_map5']:>12.4f}")

for key in ["micro_p5", "macro_p5", "macro_map5"]:
    order = sorted(out, key=lambda n: -out[n][key])
    print(f"{key:11s} 순위: {' > '.join(order)}")

json.dump(out, open(os.path.join(ROOT, "verify", "지표비교_macro_map.json"), "w"),
          ensure_ascii=False, indent=2)
print("\n저장: verify/지표비교_macro_map.json (리더보드 micro 와 대조 검증 통과)")
