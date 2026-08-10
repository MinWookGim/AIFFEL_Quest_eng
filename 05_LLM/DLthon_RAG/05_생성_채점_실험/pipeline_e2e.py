"""한 줄 이야기 -> 4컷 -> 채점까지 한 번에 도는 뼈대 (end-to-end 완주용).

왜 이걸 만드나
    8/6 에 팀 repo 를 전수로 읽어보니 **아무도 끝까지 못 돌렸다.** 각자 자기 토막만 있다.

        한 줄 이야기 -> 대본 -> 그림체 검색 -> 4컷 생성 -> 채점 -> (루프)
                         |        |            |         |
                     희연님 확인  진현·수경님    나만 있음   나만 실측
                     (mock 루프)  (실측 완료)

    진현님 `src/pipeline/comic_pipeline.py` 는 그래프 뼈대가 완성돼 있고 어댑터 4개를
    주입받게 설계돼 있다. 그중 `retrieve_references` 만 구현돼 있고 나머지 셋이 비어 있다.
    내가 8/3 에 만든 생성·채점 코드가 정확히 그 빈 칸이라, 여기서 이어 붙인다.

무엇을 지켰나
    - **남의 파일을 고치지 않는다.** 진현님 repo 는 Windows·uv 환경이라 그대로는 못 돌린다.
      그래서 노드 이름과 의존성 주입 **모양만** 그대로 따르고 구현은 여기 새로 둔다.
      나중에 진현님 그래프에 이 어댑터들을 그대로 꽂을 수 있다.
    - 코퍼스는 **daypack_v2** (팀 표준, 2,522장 / 20클래스).
    - 채점은 Gram 역방향 + **천장 대비 정규화**. 절대값으로 읽지 않는다
      (8/3 실측: 진짜 그림도 천장이 있다 -> `build/style_ceiling.py`).

★ 아직 비어 있는 칸 (일부러 남긴다 — 8/10 에 팀이 붙일 살집)
    - 컷 간 일관성 판정: 지금은 복붙 상한(코사인 0.95)만 본다. 절대 코사인은 8/3 실측에서
      같은 그림체 0.668 vs 다른 그림체 0.653 으로 **판별에 실패**했다. 더 나은 판정기 필요
    - 재생성 루프: 지금 채점기로는 켜면 안 된다 (판정 정확도 0.440 = 56% 오판).
      크리틱을 **선별기**로 쓰는 best-of-N 이 0.693 이라 그쪽이 먼저다
    - 텍스트 쿼리 경로: 지금은 이미지 질의만. 수경님 실측에서 한국어 텍스트 쿼리가
      영어의 1/3 이라(0.1027 vs 0.3327) 별도 설계가 필요하다

쓰는 법
    python build/pipeline_e2e.py --dry-run          # 돈 안 쓰고 계획·검색·채점 눈금까지만
    python build/pipeline_e2e.py --mock             # 생성만 자리표시 이미지로 (무료 완주)
    python build/pipeline_e2e.py                    # 진짜 1회 완주 (약 $0.18 + 대본 LLM)
    python build/pipeline_e2e.py --sequential       # 앞 컷을 물려서
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_prev_cut as ab          # 코퍼스 로딩·임베딩 캐시·생성 프롬프트를 재사용한다

ROOT = ab.ROOT
OUT = os.path.join(ROOT, "verify", "e2e")
N_REF, N_CUTS, TOPK = 3, 4, 5


# ─────────────────────────────────────────────────────────────
# 진현님 comic_pipeline.py 와 같은 모양의 state / 의존성 묶음
# ─────────────────────────────────────────────────────────────
class ComicState(TypedDict, total=False):
    story: str
    style_image_path: str
    cuts: list          # 4컷 장면 문장
    references: list    # 검색된 레퍼런스 (dict)
    generated_images: list
    evaluation: dict
    retry_count: int


@dataclass(frozen=True)
class PipelineDependencies:
    """네 어댑터를 밖에서 꽂는다. 검색기·생성기를 갈아도 그래프는 안 바뀐다."""
    write_cuts: Callable[[str], list]
    retrieve_references: Callable[[str, int], list]
    generate_images: Callable[[list, list], list]
    evaluate_images: Callable[[list, list, list], dict]
    reference_k: int = N_REF


# ─────────────────────────────────────────────────────────────
# 노드 1 — 대본 작성
# ─────────────────────────────────────────────────────────────
MOCK_CUTS = [
    "a lone traveler walking along a mountain path at dawn",
    "the traveler stops at a small stream and looks at the water",
    "the traveler meets an old man under a pine tree",
    "the two of them walk toward a distant village at dusk",
]


def make_write_cuts(use_llm=True):
    """한 줄 이야기 -> 4컷 장면. 키가 없거나 실패하면 목(mock)으로 내려간다.

    ★ 목이든 LLM 이든 **출력 모양이 같아야** 뒷 노드가 신경 쓸 게 없다 (희연님 빌드업과 같은 원칙).
    """
    def write_cuts(story: str) -> list:
        if not use_llm:
            return list(MOCK_CUTS)
        try:
            from openai import OpenAI
            key = open(os.path.expanduser("~/.config/openai/api_key")).read().strip()
            client = OpenAI(api_key=key)
            r = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.7,
                messages=[{"role": "user", "content":
                           "다음 이야기를 4컷 만화의 장면 묘사로 만들어라. "
                           '반드시 JSON 배열만 출력: ["장면1", "장면2", "장면3", "장면4"]. '
                           "각 장면은 그림 한 장으로 그릴 수 있게 구체적으로, **영어로** 쓴다.\n"
                           f"이야기: {story}"}])
            txt = r.choices[0].message.content.strip()
            txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            cuts = json.loads(txt)
            assert len(cuts) == N_CUTS, f"{len(cuts)}컷이 나왔다"
            return cuts
        except Exception as e:
            print(f"    대본 LLM 실패 -> 목 대본으로 간다: {repr(e)[:120]}")
            return list(MOCK_CUTS)
    return write_cuts


# ─────────────────────────────────────────────────────────────
# 노드 2 — 그림체 검색 (진현님이 이미 구현해둔 자리)
# ─────────────────────────────────────────────────────────────
def make_retrieve_references(E, style, content, paths):
    """예시 그림 한 장 -> 같은 그림체 레퍼런스 Top-k.

    ★ 질의와 **같은 원본**은 뺀다. 안 빼면 내용이 같은 걸 뽑아놓고 그림체를 뽑았다고 착각한다.
    ★ 라벨로 거르지 않는다. 검색기 실력이 그대로 드러나야 한다.
    ★★ group 이 빈 값이면 안 묶는다 (`kit/score.py` 40행 규칙. vec_* 698장이 빈 값이다).
    """
    file_to_idx = {os.path.abspath(p): i for i, p in enumerate(paths)}

    def retrieve_references(style_image_path: str, reference_k: int = N_REF) -> list:
        q = file_to_idx.get(os.path.abspath(style_image_path))
        if q is None:
            raise SystemExit(f"코퍼스 밖 이미지는 아직 못 받는다(임베딩 경로 미구현): {style_image_path}")
        sims = E @ E[q]
        ok = (content != content[q]) | (content == "")
        ok[q] = False
        sims[~ok] = -np.inf
        top = np.argsort(-sims)[:reference_k]
        return [{"rank": r + 1, "image_path": paths[i], "score": float(sims[i]),
                 "style": str(style[i])} for r, i in enumerate(top)]
    return retrieve_references


# ─────────────────────────────────────────────────────────────
# 노드 3 — 이미지 생성 (지금까지 나만 갖고 있던 칸)
# ─────────────────────────────────────────────────────────────
def make_generate_images(mock=False, sequential=False, outdir=OUT):
    """4컷을 생성한다.

    sequential=False : 컷마다 레퍼런스만 (지금 파이프라인 방식)
    sequential=True  : 컷2~4 에 **컷1 을 참조로 추가** -> 4컷이 서로를 알게 된다

    ★ 8/3 발견의 핵심이 여기다. 지금 `gen_style_test.py` 는 컷마다 독립 호출이라
      앞 컷이 다음 컷에 안 들어간다. 컷 간 일관성은 채점 문제가 아니라 생성 구조 문제였다.
    """
    def generate_images(cuts: list, references: list) -> list:
        os.makedirs(outdir, exist_ok=True)
        if mock:
            # 무료 완주용 자리표시. 뼈대가 도는지만 본다.
            out = []
            for k, scene in enumerate(cuts, 1):
                img = Image.new("RGB", (512, 512), (232, 232, 232))
                d = ImageDraw.Draw(img)
                d.rectangle([6, 6, 505, 505], outline=(120, 120, 120))
                d.text((16, 16), f"cut {k} (mock)", fill=(0, 0, 0))
                for j in range(10):
                    d.text((16, 56 + j * 18), scene[j * 46:(j + 1) * 46], fill=(70, 70, 70))
                p = os.path.join(outdir, f"e2e_mock_cut{k}.png")
                img.save(p)
                out.append(p)
            return out

        from openai import OpenAI
        client = OpenAI(api_key=open(os.path.expanduser("~/.config/openai/api_key")).read().strip())
        ref_imgs = [Image.open(r["image_path"]).convert("RGB") for r in references]
        made, paths_out, total = [], [], 0.0
        for k, scene in enumerate(cuts, 1):
            imgs = list(ref_imgs)
            prompt = ab.PROMPT.format(scene=scene)
            if sequential and made:
                imgs = list(ref_imgs) + [made[0]]      # 맨 뒤에 컷1 을 붙인다
                prompt = ab.PROMPT_SEQ.format(scene=scene)
            t0 = time.time()
            r = client.images.edit(
                model=ab.MODEL, size=ab.SIZE, quality=ab.QUALITY,
                image=[ab.to_file(x, f"r{i}.png") for i, x in enumerate(imgs)],
                prompt=prompt)
            img = Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
            made.append(img)
            p = os.path.join(outdir, f"e2e_cut{k}.png")
            img.save(p)
            paths_out.append(p)
            c = ab.cost_of(getattr(r, "usage", None)) or {}
            total += c.get("달러", 0)
            print(f"    컷{k} 생성 {time.time()-t0:.1f}초  ${c.get('달러', 0):.3f}")
        print(f"    생성 합계 ${total:.3f}")
        return paths_out
    return generate_images


# ─────────────────────────────────────────────────────────────
# 노드 4 — 채점 (Gram 역방향 + 천장 대비 정규화)
# ─────────────────────────────────────────────────────────────
def make_evaluate_images(Eg, style, content, target, ceiling=None):
    """생성된 4컷을 코퍼스에 질의로 던져 라벨을 붙인다.

    ★ 레퍼런스로 물린 그림과 같은 원본은 이웃 후보에서 뺀다. 안 빼면 베낄수록 점수가 오른다.
    ★ style_precision 은 절대값으로 읽지 않는다. 진짜 그림이 받는 점수(천장)로 나눠 읽는다.
    """
    def evaluate_images(generated_paths: list, references: list, cuts: list) -> dict:
        # ★ group 이 빈 값이면 안 묶는다 (kit/score.py 40행 규칙)
        banned = np.zeros(len(style), dtype=bool)
        idx_of = {p: j for j, p in enumerate(ab_paths_cache)}
        for r in references:
            i = idx_of.get(r["image_path"])
            if i is None:
                continue
            if content[i]:
                banned |= (content == content[i])
            banned[i] = True                     # 레퍼런스 그림 자체는 언제나 뺀다

        V = np.stack([ab.encode_one("gram", Image.open(p).convert("RGB")) for p in generated_paths])
        labels, hits = [], []
        for v in V:
            sims = Eg @ v
            sims[banned] = -np.inf
            top = np.argsort(-sims)[:TOPK]
            lab, cnt = np.unique(style[top], return_counts=True)
            labels.append(str(lab[np.argmax(cnt)]))
            hits.append(float((style[top] == target).mean()))

        labels = np.array(labels)
        S = V @ V.T
        iu = np.triu_indices(len(V), k=1)
        prec = float((labels == target).mean())
        out = {
            "target": target,
            "style_precision": prec,
            "style_precision_tail": float((labels[1:] == target).mean()),  # 컷1 잡음 제외
            "neighbor_hit": float(np.mean(hits)),
            "edition_cosine": float(S[iu].mean()),
            "labels": labels.tolist(),
        }
        if ceiling:
            out["ceiling"] = ceiling
            out["ceiling_ratio"] = prec / ceiling if ceiling else None
        # ★ 복붙 탐지만 한다. 절대 코사인으로 "일관성 있다/없다"를 판정하지 않는다
        #   (8/3 실측: 같은 그림체 0.668 vs 다른 그림체 0.653 -> 판별 실패)
        out["copy_paste_flag"] = bool(out["edition_cosine"] > 0.95)
        return out
    return evaluate_images


ab_paths_cache: list = []      # evaluate 에서 레퍼런스 경로 -> 인덱스를 찾으려고 둔다


# ─────────────────────────────────────────────────────────────
# 그래프 조립 — 진현님 build_comic_pipeline 과 같은 순서
# ─────────────────────────────────────────────────────────────
def run_pipeline(deps: PipelineDependencies, story: str, style_image_path: str) -> ComicState:
    """랭그래프가 있으면 그래프로, 없으면 같은 순서로 그냥 부른다.

    ★ 지금은 되돌아가는 엣지가 없다 = 사실상 파이프라인이다. 루프를 붙이는 건
      채점기가 판정기로 쓸 만해진 다음이다 (지금 판정 정확도 0.440 = 56% 오판).
    """
    state: ComicState = {"story": story, "style_image_path": style_image_path, "retry_count": 0}
    print("\n[1] 대본 작성")
    state["cuts"] = deps.write_cuts(story)
    for k, c in enumerate(state["cuts"], 1):
        print(f"    {k}컷: {c[:88]}")

    print("\n[2] 그림체 검색")
    state["references"] = deps.retrieve_references(style_image_path, deps.reference_k)
    for r in state["references"]:
        print(f"    {r['rank']}. {r['style']:<16} {os.path.basename(r['image_path'])[:52]}  {r['score']:.3f}")

    print("\n[3] 이미지 생성")
    state["generated_images"] = deps.generate_images(state["cuts"], state["references"])

    print("\n[4] 채점")
    state["evaluation"] = deps.evaluate_images(
        state["generated_images"], state["references"], state["cuts"])
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="daypack_v2")
    ap.add_argument("--story", default="산길을 걷던 나그네가 노인을 만나 마을까지 함께 간다")
    ap.add_argument("--target", default="ink_m3", help="목표 그림체 (질의 그림을 여기서 고른다)")
    ap.add_argument("--kind", default="gram", choices=["gram", "clip"], help="검색 임베딩")
    ap.add_argument("--sequential", action="store_true", help="앞 컷을 물려서 생성")
    ap.add_argument("--mock", action="store_true", help="생성만 자리표시로 (무료 완주)")
    ap.add_argument("--no-llm", action="store_true", help="대본도 목으로 (완전 무료)")
    ap.add_argument("--dry-run", action="store_true", help="검색·눈금까지만 하고 생성 전에 멈춘다")
    a = ap.parse_args()

    ab.PACK_NAME = a.pack
    ab.PACK = os.path.join(ROOT, a.pack)
    os.makedirs(OUT, exist_ok=True)

    paths, style, content = ab.load_pack()
    if a.target not in set(style):
        raise SystemExit(f"{a.pack} 에 없는 그림체다: {a.target}")
    print(f"코퍼스 {a.pack} — {len(paths)}장 / {len(np.unique(style))}클래스 / 목표 {a.target}")

    global ab_paths_cache
    ab_paths_cache = paths

    E = ab.get_embeddings(a.kind, paths)
    Eg = E if a.kind == "gram" else ab.get_embeddings("gram", paths)

    # 천장을 읽어온다. 없으면 style_ceiling.py 를 먼저 돌리라고 알린다
    ceil_path = os.path.join(ROOT, "verify", f"style_ceiling_{a.pack}_gram.json")
    ceiling = None
    if os.path.exists(ceil_path):
        rows = json.load(open(ceil_path, encoding="utf-8"))["rows"]
        hit = [r for r in rows if r["style"] == a.target]
        if hit:
            ceiling = hit[0]["ceiling"]
            print(f"천장({a.target}) = {ceiling:.3f}  -> 생성 점수는 이 값으로 나눠 읽는다")
    if ceiling is None:
        print(f"천장 미측정. `python build/style_ceiling.py` 를 먼저 돌리면 정규화해서 읽을 수 있다")

    rng = np.random.default_rng(20260806)
    q = int(rng.choice(np.where(style == a.target)[0]))
    print(f"질의 그림: {os.path.basename(paths[q])}")

    deps = PipelineDependencies(
        write_cuts=make_write_cuts(use_llm=not a.no_llm),
        retrieve_references=make_retrieve_references(E, style, content, paths),
        generate_images=make_generate_images(mock=a.mock, sequential=a.sequential),
        evaluate_images=make_evaluate_images(Eg, style, content, a.target, ceiling),
    )

    if a.dry_run:
        print("\n--dry-run: 대본·검색까지만 보고 생성 전에 멈춘다")
        cuts = deps.write_cuts(a.story)
        for k, c in enumerate(cuts, 1):
            print(f"    {k}컷: {c[:88]}")
        refs = deps.retrieve_references(paths[q], N_REF)
        for r in refs:
            print(f"    {r['rank']}. {r['style']:<16} {os.path.basename(r['image_path'])[:52]}  {r['score']:.3f}")
        hit = sum(r["style"] == a.target for r in refs)
        print(f"\n    레퍼런스 {hit}/{N_REF} 장이 실제 {a.target}")
        print(f"    여기서 멈춘다. 진짜로 돌리려면 --dry-run 을 빼라 (약 $0.18)")
        return

    state = run_pipeline(deps, a.story, paths[q])
    ev = state["evaluation"]
    print(f"\n    라벨 판정: {ev['labels']}")
    print(f"    style_precision {ev['style_precision']:.3f} (컷2~4 {ev['style_precision_tail']:.3f})")
    if ev.get("ceiling"):
        print(f"    천장 {ev['ceiling']:.3f} 대비 {ev['ceiling_ratio']:.1%}")
    print(f"    이웃적중 {ev['neighbor_hit']:.3f} / 컷간코사인 {ev['edition_cosine']:.3f}"
          f"{'  ★복붙 의심' if ev['copy_paste_flag'] else ''}")

    rec = {"story": a.story, "target": a.target, "retriever": a.kind,
           "sequential": a.sequential, "mock": a.mock,
           "query": os.path.basename(paths[q]),
           "cuts": state["cuts"],
           "references": [os.path.basename(r["image_path"]) for r in state["references"]],
           "evaluation": ev}
    p = os.path.join(OUT, f"e2e_{a.target}_{'seq' if a.sequential else 'ind'}.json")
    json.dump(rec, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
