"""팀 배포용 데이터 팩(daypack v1) 만들기.

원본 AI-Hub 수묵화 데이터는 50GB 라 코랩에서 못 쓴다.
여기서 **평가에 필요한 만큼만** 뽑아 수백 MB 짜리 팩으로 만든다.

★ v1 코퍼스를 여기서 못 박는다.
   코퍼스가 바뀌면 이전 점수와 비교할 수 없다(팀 규칙 2).
   그래서 팩에 VERSION 을 박고, 바꿀 일이 생기면 v2 로 새로 만든다.

★ 왜 수묵화 기법 9종만 쓰나
   - 라벨이 데이터에 이미 붙어 있다 (사람이 라벨링할 필요 0)
   - **같은 원본을 다른 기법으로 그린 통제쌍**이 있다 -> 내용과 그림체를 분리해 잴 수 있다
   - 우리가 재본 것 중 제일 어렵다(같은 기법 이웃이 무작위의 1.5배).
     쉬운 클래스(이모지 등)를 섞으면 방법을 안 고쳐도 평균 점수가 올라간다 -> v1 에서 제외
"""
import os, io, json, glob, csv, zipfile, random, collections
from PIL import Image

random.seed(20260729)          # 팩을 다시 만들어도 같은 그림이 나오게 고정
ROOT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon"
INK  = f"{ROOT}/data/Ai-Hub 데이터/168.한국 전통 수묵화 화풍별 제작 데이터/01-1.정식개방데이터"
OUT  = f"{ROOT}/daypack_v1"

PER_STYLE = 140        # 기법 하나당 몇 장
THUMB     = 384        # 긴 변 기준 축소 (임베딩엔 이 정도면 충분)
JPEG_Q    = 88

os.makedirs(f"{OUT}/images", exist_ok=True)

# ── 1. 라벨을 읽는다: 이미지파일 -> (기법 Method, 원본ID) ─────────────
print("[1] 라벨 읽는 중")
meta = {}
for zp in sorted(glob.glob(f"{INK}/*/02.라벨링데이터/*.zip")):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            j = json.loads(z.read(n).decode("utf-8-sig"))
            a  = j.get("annotation", {}) or {}
            im = j.get("images", {}) or {}
            paire = a.get("Paire")                      # 실제 그림 파일명
            m     = (a.get("Paint") or {}).get("Method")  # 기법 번호 1~9
            src   = im.get("identifier")                 # 원본ID (같은 그림인지)
            if paire and m and src:
                meta[os.path.basename(paire)] = (int(m), str(src))
print(f"    라벨 {len(meta):,}건")

# ── 2. 기법별로 골고루, 그리고 한 원본이 한쪽에 몰리지 않게 뽑는다 ────
by_style = collections.defaultdict(list)
for fn, (m, src) in meta.items():
    by_style[m].append((fn, src))

want = {}                                   # 파일명 -> (기법, 원본ID)
for m in sorted(by_style):
    rows = by_style[m][:]
    random.shuffle(rows)
    # 같은 원본이 한 기법 안에서 여러 장 들어가지 않게 (내용 편중 방지)
    seen_src, picked = set(), []
    for fn, src in rows:
        if src in seen_src:
            continue
        seen_src.add(src); picked.append((fn, src))
        if len(picked) >= PER_STYLE:
            break
    for fn, src in picked:
        want[fn] = (m, src)
    print(f"    기법 {m}: {len(picked)}장 (후보 {len(rows)})")

print(f"    총 {len(want)}장 뽑음")

# ── 3. zip 에서 그 파일만 꺼내 축소 저장 (50GB 를 풀지 않는다) ────────
print("[2] 이미지 추출·축소 중")
rows, got = [], 0
for zp in sorted(glob.glob(f"{INK}/*/01.원천데이터/*.zip")):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if b not in want:
                continue
            m, src = want[b]
            try:
                img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            except Exception:
                continue
            img.thumbnail((THUMB, THUMB))
            d = f"{OUT}/images/m{m}"
            os.makedirs(d, exist_ok=True)
            outname = f"m{m}_{os.path.splitext(b)[0]}.jpg"
            img.save(f"{d}/{outname}", quality=JPEG_Q)
            rows.append({"file": f"images/m{m}/{outname}", "style": f"m{m}", "content": src})
            got += 1
            if got % 200 == 0:
                print(f"    {got}장…")
print(f"    저장 {got}장")

# ── 4. meta.csv — 채점기가 읽는 유일한 정답표 ────────────────────────
with open(f"{OUT}/meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "style", "content"])
    w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["file"]))

cnt = collections.Counter(r["style"] for r in rows)
json.dump({"version": "daypack_v1", "n_images": len(rows),
           "n_styles": len(cnt), "per_style": dict(sorted(cnt.items())),
           "thumb_px": THUMB, "seed": 20260729,
           "source": "AI-Hub 168 한국 전통 수묵화 화풍별 제작 데이터 (개방데이터)"},
          open(f"{OUT}/VERSION.json", "w"), ensure_ascii=False, indent=2)

mb = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(f"{OUT}/images") for f in fs) / 1e6
print(f"\n끝. {len(rows)}장 / {len(cnt)}클래스 / {mb:.0f}MB -> {OUT}")
print("    클래스별:", dict(sorted(cnt.items())))
