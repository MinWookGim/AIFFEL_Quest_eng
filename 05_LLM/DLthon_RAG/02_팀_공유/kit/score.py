"""그림체 검색기 채점기 — 규격만 지키면 뭘 꽂든 점수가 나온다.

쓰는 법
    python score.py encoders/clip_base.py --name "CLIP 기준선"

너가 할 일은 딱 하나다. 아래 규격의 파일 하나를 만들어서 넣으면 된다.

    # encoders/내방법.py
    def encode(images):        # images: PIL.Image 리스트
        return vectors         # numpy (N, D) float32.  정규화는 여기서 안 해도 된다

무엇을 재나
    그림 한 장을 질의로 던져 **가장 가까운 5장**을 찾고,
    그중 **같은 기법(그림체)** 이 몇 장인지 센다. 1,259장 전부에 대해 반복해 평균낸다.

★ 두 가지를 반드시 빼고 잰다 (안 빼면 그림체가 아니라 딴 걸 재게 된다)
    1) 자기 자신
    2) **같은 원본을 다른 기법으로 그린 그림** (= 내용이 같은 그림)
       이걸 안 빼면 "내용이 비슷한 것"을 찾고도 점수가 오른다.
       우리 실측에서 내용은 무작위의 102배로 딸려 왔다. 그림체는 1.5배였다.

★ 숫자 하나만 보고 이겼다고 하지 마라
    같은 코드를 두 번 돌려도 숫자는 흔들린다. 그래서 **95% 구간**을 같이 찍는다.
    두 방법을 비교할 땐 구간이 겹치는지부터 본다. 겹치면 "아직 모른다"가 정답이다.
    (짝지어 비교하려고 질의별 점수를 runs/ 에 남긴다)
"""
import os, sys, csv, json, time, argparse, importlib.util
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.environ.get("DAYPACK", os.path.join(HERE, "..", "daypack_v1"))
TOPK = 5
BOOT = 1000          # 95% 구간을 낼 때 다시 뽑는 횟수


def load_pack():
    """meta.csv 를 읽어 이미지·정답라벨을 가져온다.

    group  = 이웃 후보에서 빼야 할 묶음 (같은 원본 / 같은 작가). 비어 있으면 안 뺀다
    domain = ink / paint / vector. 도메인 안쪽 성적을 따로 보려고 쓴다
    (v1 meta.csv 는 content 컬럼만 있어서 그걸 group 으로 읽는다)
    """
    rows = list(csv.DictReader(open(os.path.join(PACK, "meta.csv"), encoding="utf-8")))
    imgs = [Image.open(os.path.join(PACK, r["file"])).convert("RGB") for r in rows]
    style = np.array([r["style"] for r in rows])
    group = np.array([r.get("group", r.get("content", "")) or "" for r in rows])
    domain = np.array([r.get("domain", "") or "" for r in rows])
    ver = json.load(open(os.path.join(PACK, "VERSION.json"), encoding="utf-8"))
    return imgs, style, group, domain, ver


def load_encoder(path):
    """사용자가 만든 encode() 를 불러온다."""
    spec = importlib.util.spec_from_file_location("user_encoder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_encoder"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "encode"):
        raise SystemExit(f"{path} 에 encode(images) 함수가 없다. 규격을 보라.")
    return mod.encode


def precision_at_k(E, style, group, k=TOPK, restrict=None):
    """질의별 precision@k 와, 그 질의의 '찍었을 때 기대값'을 같이 돌려준다.

    restrict 를 주면(도메인 배열) **같은 도메인 안에서만** 이웃을 찾는다.
    전체 판에서는 "수묵화냐 이모지냐" 같은 공짜 문제가 섞여 점수가 부푼다.
    """
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)   # 방어적 정규화
    N = len(E)
    hits = np.zeros(N)
    chance = np.zeros(N)
    for i in range(N):
        # 자기 자신 + 같은 묶음(같은 원본 / 같은 작가)은 후보에서 뺀다
        ok = (group != group[i]) | (group == "")
        ok[i] = False
        if restrict is not None:
            ok &= (restrict == restrict[i])
        sims = E @ E[i]
        sims[~ok] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])][:k]
        hits[i] = np.mean(style[top] == style[i])
        pool = style[ok]
        chance[i] = np.mean(pool == style[i]) if len(pool) else 0.0
    return hits, chance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encoder", help="encode(images) 가 든 .py 경로")
    ap.add_argument("--name", default=None, help="리더보드에 적힐 이름")
    ap.add_argument("--note", default="", help="한 줄 메모 (뭘 바꿨나)")
    a = ap.parse_args()
    name = a.name or os.path.basename(a.encoder)

    imgs, style, group, domain, ver = load_pack()
    print(f"[1] 코퍼스 {ver['version']} — {len(imgs)}장 / {ver['n_styles']}클래스")

    encode = load_encoder(a.encoder)
    print(f"[2] {name} 로 임베딩 뽑는 중…")
    t0 = time.time()
    E = np.asarray(encode(imgs), dtype="float32")
    dt = time.time() - t0
    if E.ndim != 2 or len(E) != len(imgs):
        raise SystemExit(f"encode() 가 (N, D) 를 안 돌려줬다. 받은 모양: {E.shape}")
    print(f"    {E.shape[0]}x{E.shape[1]}  ({dt:.1f}초, {E.nbytes/1e6:.0f}MB)")

    print("[3] 채점 중 (자기 자신·같은 묶음 제외)")
    hits, chance = precision_at_k(E, style, group)
    overall, ch = float(hits.mean()), float(chance.mean())

    # 95% 구간 — 질의를 다시 뽑아가며 평균이 얼마나 흔들리는지 본다
    rng = np.random.default_rng(0)
    boots = np.array([hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n{'클래스':10s} {'p@5':>8s} {'무작위':>8s} {'배수':>7s}")
    per = {}
    for s in sorted(set(style)):
        m = style == s
        p, c = hits[m].mean(), chance[m].mean()
        per[s] = round(float(p), 4)
        print(f"{s:10s} {p:8.4f} {c:8.4f} {p/c if c else 0:7.2f}")
    print("-" * 36)
    print(f"{'전체':10s} {overall:8.4f} {ch:8.4f} {overall/ch if ch else 0:7.2f}")

    # ★도메인 안쪽 성적 — 여기가 진짜 어려운 부분이다
    doms = sorted(set(domain) - {""})
    dom_res = {}
    if doms:
        hin, cin = precision_at_k(E, style, group, restrict=domain)
        print(f"\n{'도메인':10s} {'전체판':>9s} {'도메인안':>9s} {'무작위':>9s} {'배수':>7s}")
        for d in doms:
            m = domain == d
            p_all, p_in, c_in = hits[m].mean(), hin[m].mean(), cin[m].mean()
            dom_res[d] = {"전체판": round(float(p_all), 4), "도메인안": round(float(p_in), 4),
                          "무작위": round(float(c_in), 4),
                          "배수": round(float(p_in / c_in), 2) if c_in else 0}
            print(f"{d:10s} {p_all:9.4f} {p_in:9.4f} {c_in:9.4f} "
                  f"{p_in/c_in if c_in else 0:7.2f}")
        print("  -> '전체판'이 높고 '도메인안'이 낮으면, 도메인 구분이라는 **공짜 문제**로 점수를 번 것이다.")
    print(f"\n★ 95% 구간  {lo:.4f} ~ {hi:.4f}   (폭 {hi-lo:.4f})")
    print(f"  -> 다른 방법과 이 폭보다 작게 차이 나면 **아직 이긴 게 아니다.**")

    # 질의별 점수 저장 — 나중에 두 방법을 짝지어 비교할 때 쓴다
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    slug = "".join(ch2 if ch2.isalnum() else "_" for ch2 in name)[:40]
    np.save(os.path.join(HERE, "runs", f"{slug}.npy"), hits)

    # 리더보드에 한 줄 붙인다
    lb = os.path.join(HERE, "leaderboard.csv")
    new = not os.path.exists(lb)
    with open(lb, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["이름", "코퍼스", "p@5", "무작위", "배수", "95%하한", "95%상한",
                        "차원", "초", "메모"])
        w.writerow([name, ver["version"], f"{overall:.4f}", f"{ch:.4f}",
                    f"{overall/ch if ch else 0:.2f}", f"{lo:.4f}", f"{hi:.4f}",
                    E.shape[1], f"{dt:.1f}", a.note])
    print(f"\n리더보드에 추가됨 -> {lb}")
    json.dump({"name": name, "corpus": ver["version"], "p5": overall, "chance": ch,
               "ci95": [float(lo), float(hi)], "per_class": per, "per_domain": dom_res,
               "dim": int(E.shape[1]), "seconds": round(dt, 1), "note": a.note},
              open(os.path.join(HERE, "runs", f"{slug}.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
