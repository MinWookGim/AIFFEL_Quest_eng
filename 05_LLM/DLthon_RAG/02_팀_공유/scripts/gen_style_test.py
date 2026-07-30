"""생성 붙여보기 — 레퍼런스를 물리면 정말 그 그림체로 그려지나?

발표 하루 전까지 이 프로젝트는 그림을 단 한 장도 생성한 적이 없었다.
"검색한 레퍼런스를 생성에 물리면 된다"는 말만 있고 근거가 없었다. 그걸 지금 확인한다.

  A안  말로만    프롬프트에 "수묵화 화풍" 이라고만 쓴다        (레퍼런스 없음)
  B안  레퍼런스  검색된 수묵화 3장을 참조로 같이 넣는다        (우리 주장)

그리고 나온 그림을 **우리 지표로 직접 잰다** — 수묵화 코퍼스 591장 임베딩과의 거리.
눈으로 "비슷해 보인다"로 끝내지 않는다.
"""
import os, io, json, base64, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon"
OUT  = f"{ROOT}/verify"
MODEL = "gpt-image-1"
SIZE, QUALITY = "1024x1024", "medium"

# ── 1. 레퍼런스 = 어제 데모에서 실제로 검색된 수묵화 컷 ──────────────
# demo_코믹_수묵화.png 의 조판 좌표(PAD 26, PW 460, PH 320, GAP 18, y 78)에서 그대로 잘라낸다.
demo = Image.open(f"{OUT}/demo_코믹_수묵화.png")
PAD, PW, PH, GAP, Y = 26, 460, 320, 18, 78
refs = [demo.crop((PAD + i*(PW+GAP), Y, PAD + i*(PW+GAP) + PW, Y + PH)).convert("RGB")
        for i in range(3)]
for i, r in enumerate(refs):
    r.save(f"{OUT}/gen_ref{i+1}.png")
print(f"[1] 레퍼런스 {len(refs)}장 준비 (어제 검색된 수묵화 컷)")

# ── 2. 장면 = 어제 LLM 이 쓴 그 대본 그대로 ──────────────────────────
cuts = json.load(open(f"{OUT}/demo_stories.json"))["코믹"][:2]
for c in cuts:
    print(f"    컷: \"{c['dialogue']}\"  / {c['scene']}")

from openai import OpenAI
client = OpenAI(api_key=open(os.path.expanduser("~/.config/openai/api_key")).read().strip())

def to_file(img, name):
    b = io.BytesIO(); img.save(b, format="PNG"); b.name = name; b.seek(0)
    return b

def save_b64(b64, path):
    Image.open(io.BytesIO(base64.b64decode(b64))).save(path)
    return path

# ── 비용 측정 ────────────────────────────────────────────────────────
# 토큰 수는 API 응답의 usage 에서 그대로 받는 **실측값**이다.
# 달러 환산 단가는 내가 적어 넣은 값이므로 발표 전에 요금 페이지로 한 번 확인할 것.
RATE = {"text_in": 5.0/1e6, "image_in": 10.0/1e6, "image_out": 40.0/1e6}   # $/token

def cost_of(usage):
    if usage is None: return None
    det = getattr(usage, "input_tokens_details", None)
    t_in = getattr(det, "text_tokens", 0) or 0
    i_in = getattr(det, "image_tokens", 0) or 0
    out  = getattr(usage, "output_tokens", 0) or 0
    if not (t_in or i_in):                      # details 가 없으면 전부 텍스트로 본다
        t_in = getattr(usage, "input_tokens", 0) or 0
    usd = t_in*RATE["text_in"] + i_in*RATE["image_in"] + out*RATE["image_out"]
    return {"텍스트입력토큰": t_in, "이미지입력토큰": i_in, "출력토큰": out, "달러": round(usd, 4)}

results = {}   # (컷번호, 방식) -> 파일경로
costs   = []   # 실측 비용 기록

for k, c in enumerate(cuts, 1):
    # A안 — 말로만
    try:
        print(f"\n[2-{k}A] 말로만 생성")
        t0 = time.time()
        r = client.images.generate(
            model=MODEL, size=SIZE, quality=QUALITY,
            prompt=f"Korean traditional ink wash painting style. {c['scene']}")
        dt = time.time() - t0
        results[(k, "A")] = save_b64(r.data[0].b64_json, f"{OUT}/gen_cut{k}_A_말로만.png")
        cc = cost_of(getattr(r, "usage", None))
        costs.append({"컷": k, "방식": "A 말로만", "초": round(dt, 1), **(cc or {})})
        print(f"     -> {results[(k,'A')]}  ({dt:.1f}초, {cc})")
    except Exception as e:
        print("     !! 실패:", repr(e)[:200])

    # B안 — 검색된 레퍼런스를 물려서
    try:
        print(f"[2-{k}B] 레퍼런스 물려서 생성")
        t0 = time.time()
        r = client.images.edit(
            model=MODEL, size=SIZE, quality=QUALITY,
            image=[to_file(x, f"ref{i}.png") for i, x in enumerate(refs)],
            prompt=("Draw a NEW illustration of this scene: " + c["scene"] +
                    ". Match the painting style of the reference images exactly — "
                    "the brush texture, ink tone, line quality and color palette. "
                    "Do not copy the content of the references, only their style."))
        dt = time.time() - t0
        results[(k, "B")] = save_b64(r.data[0].b64_json, f"{OUT}/gen_cut{k}_B_레퍼런스.png")
        cc = cost_of(getattr(r, "usage", None))
        costs.append({"컷": k, "방식": "B 레퍼런스", "초": round(dt, 1), **(cc or {})})
        print(f"     -> {results[(k,'B')]}  ({dt:.1f}초, {cc})")
    except Exception as e:
        print("     !! 실패:", repr(e)[:200])

# ── 2.5 비용 요약 (발표용) ───────────────────────────────────────────
if costs:
    print("\n[2.5] 실측 비용")
    print(f"    {'방식':14s} {'초':>5s} {'텍스트입력':>10s} {'이미지입력':>10s} {'출력':>8s} {'$':>8s}")
    for c in costs:
        print(f"    컷{c['컷']} {c['방식']:10s} {c.get('초',0):5.1f} "
              f"{c.get('텍스트입력토큰',0):10d} {c.get('이미지입력토큰',0):10d} "
              f"{c.get('출력토큰',0):8d} {c.get('달러',0):8.4f}")
    tot = sum(c.get("달러", 0) for c in costs)
    print(f"    합계 {len(costs)}장 = ${tot:.4f}   (한 장 평균 ${tot/len(costs):.4f})")
    print(f"    4컷 한 편 = 약 ${tot/len(costs)*4:.3f}  /  100편 = 약 ${tot/len(costs)*4*100:.1f}")
    json.dump({"단가($/token)": RATE, "건별": costs,
               "장당평균$": round(tot/len(costs), 4),
               "4컷한편$": round(tot/len(costs)*4, 4)},
              open(f"{OUT}/gen_cost.json", "w"), ensure_ascii=False, indent=2)

if not results:
    raise SystemExit("생성이 하나도 안 됐다. 여기서 멈춘다.")

# ── 3. 우리 지표로 잰다 — 수묵화 코퍼스 591장과 얼마나 가까운가 ──────
print("\n[3] CLIP 으로 측정 (수묵화 코퍼스 591장 기준)")
import torch
from transformers import CLIPModel, CLIPProcessor
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(M).to(dev).eval()
proc = CLIPProcessor.from_pretrained(M)

E = np.load(f"{OUT}/ink_emb.npy")          # 수묵화 591장 (이미 L2 정규화됨)

def embed(img):
    with torch.no_grad():
        b = proc(images=[img], return_tensors="pt").to(dev)
        f = model.get_image_features(**b)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]

def score(img):
    v = embed(img); s = E @ v
    return float(s.mean()), float(np.sort(s)[-5:].mean())

rows = []
for i, r in enumerate(refs, 1):            # 기준선: 진짜 수묵화(검색된 레퍼런스)
    m, t = score(r); rows.append((f"레퍼런스{i} (진짜 수묵화)", m, t))
for (k, w), p in sorted(results.items()):
    m, t = score(Image.open(p).convert("RGB"))
    rows.append((f"컷{k} {'말로만(A)' if w=='A' else '레퍼런스물림(B)'}", m, t))

print(f"\n    {'':28s} 코퍼스평균  최근접5평균")
for n, m, t in rows:
    print(f"    {n:28s}   {m:.4f}      {t:.4f}")

json.dump([{"항목": n, "코퍼스평균": round(m, 4), "최근접5평균": round(t, 4)} for n, m, t in rows],
          open(f"{OUT}/gen_style_scores.json", "w"), ensure_ascii=False, indent=2)

# ── 4. 비교 그림 (발표 부록용) ───────────────────────────────────────
def font(sz, bold=False):
    for p in [f"/usr/share/fonts/truetype/nanum/NanumBarunGothic{'Bold' if bold else ''}.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

CW, CH, G, TOP = 340, 340, 16, 96
cols = [("검색된 레퍼런스", refs[0]),
        ("A. 말로만 시켰을 때", Image.open(results[(1, "A")]).convert("RGB") if (1, "A") in results else None),
        ("B. 레퍼런스 물렸을 때", Image.open(results[(1, "B")]).convert("RGB") if (1, "B") in results else None)]
cols = [(t, im) for t, im in cols if im is not None]
W = 26*2 + CW*len(cols) + G*(len(cols)-1); H = TOP + CH + 76
cv = Image.new("RGB", (W, H), "#fff"); d = ImageDraw.Draw(cv)
d.text((26, 20), "검색한 레퍼런스를 생성에 물리면 그 그림체로 그려지나", fill="#14273d", font=font(30, True))
d.text((26, 60), f"장면은 똑같습니다: \"{cuts[0]['dialogue']}\"", fill="#666", font=font(21))
for i, (t, im) in enumerate(cols):
    x = 26 + i*(CW+G)
    im = im.resize((CW, CH), Image.LANCZOS)
    cv.paste(im, (x, TOP)); d.rectangle([x, TOP, x+CW, TOP+CH], outline="#222", width=2)
    d.text((x+4, TOP+CH+12), t, fill="#111", font=font(22, True))
cv.save(f"{OUT}/gen_style_compare.png")
print(f"\n    -> {OUT}/gen_style_compare.png")
print("\n끝.")
