"""팀 배포용 코퍼스 v2 — 난이도 스펙트럼을 갖춘 판을 만든다.

v1 은 수묵화 기법 9종만 넣었다. 어렵기만 하고 **중간이 비어 있었다.**
v2 는 세 도메인을 섞어서, 쉬운 것부터 거의 못 가르는 것까지 한 판에 담는다.

  ink     수묵화 기법 9종          어렵다 (색 단서가 없다. 붓질과 번짐만)
  paint   WikiArt 사조 6종         중간 (Impressionism vs Post_Impressionism 이 특히)
  vector  공개 라이선스 벡터 5종    쉽다 (평면 색, 굵은 윤곽)

★ 왜 쉬운 걸 일부러 넣나
   "쉬운 클래스를 넣으면 평균이 오른다"는 함정 때문에 v1 에서는 뺐다.
   그런데 실제 서비스는 **전 코퍼스에서** 그림체를 찾아야 하니 쉬운 것도 판에 있어야 맞다.
   대신 채점기가 **클래스별·도메인별로 따로** 찍게 하고, 평균 하나로 자랑하지 않는다.

★ 누수 차단 — meta.csv 의 `group` 컬럼
   같은 group 인 그림은 이웃 후보에서 뺀다. 도메인마다 "새는 구멍"이 다르다.
     ink    group = 원본ID   (같은 그림을 다른 기법으로 그린 통제쌍이 있다)
     paint  group = 작가ID   (사조 하나에 작가가 몇 명뿐이라 작가를 맞히게 된다)
     vector group = 없음     (서로 독립된 그림들)

★ 걸러낸 것 (표본을 눈으로 봐서 잡은 것들)
   - 이모지 세트의 **국기**: 그림체가 아니라 색 사각형이라 어느 세트든 똑같이 생겼다
   - **humaaans 통째 제외**: 전신과 신체 부위 조각이 섞였는데 파일명으로 못 가른다
   - 이라스토야의 **선화(塗り絵)**: 채도로 걸러낸다 (컬러판과 사실상 다른 그림체)
"""
import os, re, csv, glob, json, shutil, random, collections
import numpy as np
from PIL import Image

random.seed(20260730)
ROOT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon"
V1   = f"{ROOT}/daypack_v1"
OPEN = f"{ROOT}/data/open_styles"
OUT  = f"{ROOT}/daypack_v2"
THUMB, JPEG_Q = 384, 88

PER_PAINT  = 120     # WikiArt 사조당
PER_VECTOR = 140     # 벡터 세트당
ARTIST_CAP = 8       # ★한 작가가 한 사조를 독차지하지 못하게
MAX_SCAN   = 95000

PAINT_STYLES = ["Ukiyo_e", "Baroque", "Cubism", "Art_Nouveau",
                "Impressionism", "Post_Impressionism"]
VECTOR_SETS  = ["twemoji", "openmoji", "notoemoji", "undraw", "irasutoya"]

FLAG = re.compile(r"1f1[ef][0-9a-f]", re.I)     # 국기 = 지역표시 문자 U+1F1E6~1F1FF
rows = []

os.makedirs(f"{OUT}/images", exist_ok=True)


def save(img, cls, name, style, domain, group):
    d = f"{OUT}/images/{cls}"
    os.makedirs(d, exist_ok=True)
    img = img.convert("RGB")
    img.thumbnail((THUMB, THUMB))
    img.save(f"{d}/{name}", quality=JPEG_Q)
    rows.append({"file": f"images/{cls}/{name}", "style": style,
                 "domain": domain, "group": group})


# ── 1. ink — v1 을 그대로 가져온다 (이미 뽑아놨으니 다시 안 뽑는다) ──
print("[1] ink — daypack_v1 재사용")
for r in csv.DictReader(open(f"{V1}/meta.csv", encoding="utf-8")):
    src = f"{V1}/{r['file']}"
    cls = f"ink_{r['style']}"
    d = f"{OUT}/images/{cls}"
    os.makedirs(d, exist_ok=True)
    dst = f"{d}/{os.path.basename(r['file'])}"
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    rows.append({"file": f"images/{cls}/{os.path.basename(r['file'])}",
                 "style": cls, "domain": "ink", "group": f"ink_{r['content']}"})
print(f"    {len(rows)}장")

# ── 2. paint — WikiArt 스트리밍 (33GB 를 다 받지 않는다) ─────────────
print(f"[2] paint — WikiArt 사조 {len(PAINT_STYLES)}종 (작가당 최대 {ARTIST_CAP}장)")
from datasets import load_dataset

ds = load_dataset("huggan/wikiart", split="train", streaming=True)
names = ds.features["style"].names
want = {names.index(s): s for s in PAINT_STYLES if s in names}
missing = [s for s in PAINT_STYLES if s not in names]
if missing:
    print(f"    !! 데이터셋에 없는 사조: {missing}")

buck = collections.defaultdict(list)      # style_id -> [(img, artist), ...]
cap = collections.Counter()               # (style_id, artist) -> 장수
scanned = 0
for row in ds:
    scanned += 1
    sid, aid = row["style"], row.get("artist", -1)
    if sid in want and len(buck[sid]) < PER_PAINT and cap[(sid, aid)] < ARTIST_CAP:
        cap[(sid, aid)] += 1
        im = row["image"].convert("RGB")
        im.thumbnail((THUMB, THUMB))
        buck[sid].append((im, aid))
    if scanned % 5000 == 0:
        got = sum(len(v) for v in buck.values())
        print(f"    {scanned:6d}행 | 모은 것 {got}/{PER_PAINT*len(want)}")
    if scanned >= MAX_SCAN or all(len(buck[s]) >= PER_PAINT for s in want):
        break

for sid, style in want.items():
    for i, (im, aid) in enumerate(buck[sid]):
        save(im, f"paint_{style}", f"{style}_{i:03d}.jpg",
             f"paint_{style}", "paint", f"artist_{aid}")
    print(f"    {style}: {len(buck[sid])}장 (작가 {len(set(a for _, a in buck[sid]))}명)")

# ── 3. vector — 공개 라이선스 세트, 걸러가며 ─────────────────────────
print("[3] vector — 공개 라이선스 세트")


def saturation(img):
    """선화(윤곽선만)는 채도가 거의 0 이다. 컬러판과 가르는 데 쓴다."""
    a = np.asarray(img.convert("HSV"), dtype="float32")
    return float(a[:, :, 1].mean())


for s in VECTOR_SETS:
    fs = sorted(glob.glob(f"{OPEN}/{s}/*.jpg"))
    kept, dropped_flag, dropped_line = [], 0, 0
    for f in fs:
        b = os.path.basename(f)
        if FLAG.search(b):                       # 국기 제외
            dropped_flag += 1
            continue
        if s == "irasutoya":
            if saturation(Image.open(f)) < 12:   # 선화 제외
                dropped_line += 1
                continue
        kept.append(f)
    random.shuffle(kept)
    kept = kept[:PER_VECTOR]
    for f in kept:
        save(Image.open(f), f"vec_{s}", os.path.basename(f), f"vec_{s}", "vector", "")
    note = f" (국기 {dropped_flag} 제외" + (f", 선화 {dropped_line} 제외" if dropped_line else "") + ")"
    print(f"    {s}: {len(kept)}장{note if dropped_flag or dropped_line else ''}")

# ── 4. meta.csv / VERSION.json ───────────────────────────────────────
rows.sort(key=lambda r: r["file"])
with open(f"{OUT}/meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "style", "domain", "group"])
    w.writeheader(); w.writerows(rows)

cnt = collections.Counter(r["style"] for r in rows)
dom = collections.Counter(r["domain"] for r in rows)
json.dump({"version": "daypack_v2", "n_images": len(rows), "n_styles": len(cnt),
           "domains": dict(sorted(dom.items())), "per_style": dict(sorted(cnt.items())),
           "thumb_px": THUMB, "seed": 20260730, "artist_cap": ARTIST_CAP,
           "filters": ["이모지 국기(U+1F1E6~1F1FF) 제외",
                       "이라스토야 선화(채도<12) 제외",
                       "humaaans 제외(전신·부위 혼재, 파일명으로 못 가름)"],
           "sources": ["AI-Hub 168 수묵화(개방데이터)", "huggan/wikiart",
                       "twemoji CC BY 4.0", "OpenMoji CC BY-SA 4.0",
                       "Noto Emoji CC BY 4.0", "unDraw(표기 불요)",
                       "いらすとや(비상업)"]},
          open(f"{OUT}/VERSION.json", "w"), ensure_ascii=False, indent=2)

mb = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(f"{OUT}/images") for f in fs) / 1e6
print(f"\n끝. {len(rows)}장 / {len(cnt)}클래스 / {mb:.0f}MB -> {OUT}")
print("    도메인별:", dict(sorted(dom.items())))
