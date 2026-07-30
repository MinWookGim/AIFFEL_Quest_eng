"""팀에 나눠줄 수 있는 코퍼스만 골라 공유팩을 만든다.

★왜 daypack_v2 를 그대로 못 주나
   AI-Hub 데이터 이용정책 5항:
     "제공 받은 AI데이터 등을 (…) 승인을 받지 않은 다른 법인, 단체 또는 개인에게
      열람하게 하거나 제공, 양도, 대여, 판매하여서는 안됩니다."
   -> ink 도메인(AI-Hub 168 수묵화)은 **팀원에게 넘기는 것 자체가 금지**다.
      팀원은 각자 AI-Hub 에서 본인 명의로 받아야 한다.
      다행히 `make_daypack.py` 가 seed 고정이라, 각자 돌리면 **똑같은 팩이 재현**된다.

   이라스토야도 소재 자체의 재배포를 막고 있어 제외한다.

★그래서 공유팩에 들어가는 것
   paint   WikiArt 사조 6종        — 퍼블릭 도메인 회화
   vector  twemoji · openmoji · notoemoji · undraw   — CC BY / CC BY-SA / 자유 이용

   ink 와 irasutoya 는 **빠진다.** 그래서 공유팩 점수는 daypack_v2 점수와 비교할 수 없다(규칙 2).
"""
import os, csv, json, shutil, collections

ROOT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon"
SRC  = f"{ROOT}/daypack_v2"
OUT  = f"{ROOT}/daypack_share_v1"

EXCLUDE_DOMAIN = {"ink"}                 # AI-Hub — 제공 금지
EXCLUDE_STYLE  = {"vec_irasutoya"}       # 소재 재배포 금지

rows = []
for r in csv.DictReader(open(f"{SRC}/meta.csv", encoding="utf-8")):
    if r["domain"] in EXCLUDE_DOMAIN or r["style"] in EXCLUDE_STYLE:
        continue
    dst = f"{OUT}/{r['file']}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(f"{SRC}/{r['file']}", dst)
    rows.append(r)

with open(f"{OUT}/meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "style", "domain", "group"])
    w.writeheader(); w.writerows(rows)

cnt = collections.Counter(r["style"] for r in rows)
dom = collections.Counter(r["domain"] for r in rows)
json.dump({"version": "daypack_share_v1", "n_images": len(rows), "n_styles": len(cnt),
           "domains": dict(sorted(dom.items())), "per_style": dict(sorted(cnt.items())),
           "제외": ["ink (AI-Hub 이용정책 5항 — 제3자 제공 금지)",
                   "vec_irasutoya (소재 재배포 제한)"],
           "출처_표기": {
               "WikiArt": "huggan/wikiart (퍼블릭 도메인 회화)",
               "Twemoji": "Twemoji, CC BY 4.0",
               "OpenMoji": "OpenMoji, CC BY-SA 4.0",
               "Noto Emoji": "Google Noto Emoji, CC BY 4.0",
               "unDraw": "unDraw (자유 이용, 표기 불요)"},
           "주의": "daypack_v2 점수와 비교 금지 — 코퍼스 구성이 다르다(팀 규칙 2)"},
          open(f"{OUT}/VERSION.json", "w"), ensure_ascii=False, indent=2)

# 출처 표기 파일 — CC BY / CC BY-SA 는 표기가 의무다
open(f"{OUT}/LICENSE_출처.md", "w", encoding="utf-8").write("""# 출처 표기

이 폴더의 이미지는 아래 출처에서 받아 384px 로 축소한 것이다.

| 세트 | 출처 | 라이선스 | 표기 의무 |
|---|---|---|---|
| paint_* | huggan/wikiart | 퍼블릭 도메인 회화 | - |
| vec_twemoji | Twemoji (jdecked/twemoji) | **CC BY 4.0** | 필요 |
| vec_openmoji | OpenMoji (hfg-gmuend/openmoji) | **CC BY-SA 4.0** | 필요 + 동일조건 |
| vec_notoemoji | Google Noto Emoji | **CC BY 4.0** | 필요 |
| vec_undraw | unDraw | 자유 이용 | 불요 |

## 여기 없는 것
- **ink (흑백 회화 기법 9종)** — AI-Hub 168 한국 전통 수묵화 화풍별 제작 데이터.
  **이용정책 5항에 따라 제3자에게 제공할 수 없다.** 각자 AI-Hub 에서 본인 명의로 받은 뒤
  `build/make_daypack.py` 를 돌리면 같은 팩이 재현된다(seed 고정).
  2차 저작물에도 **한국지능정보사회진흥원 사업결과임**을 밝혀야 한다(이용정책 1항).
- **vec_irasutoya** — いらすとや. 소재 자체의 재배포가 제한된다. `build/fetch_irasutoya.py` 로 각자 수집.
""")

mb = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(f"{OUT}/images") for f in fs) / 1e6
print(f"공유팩 {len(rows)}장 / {len(cnt)}클래스 / {mb:.0f}MB -> {OUT}")
print("  도메인별:", dict(sorted(dom.items())))
print("  클래스별:", dict(sorted(cnt.items())))
