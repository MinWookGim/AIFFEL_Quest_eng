"""いらすとや(irasutoya) 그림을 '그림체 클래스' 하나로 모은다.

왜 필요하나
    그림체 코퍼스에 **뚜렷하게 다른 화풍**이 하나 더 붙으면 검색기 평가가 다양해진다.
    이라스토야는 선이 굵고 색이 평평한 특유의 그림체라 클래스로 쓰기 좋다.

★ 예의를 지키는 수집 (남의 서버를 긁는 일이다)
    - robots.txt 확인함: 비어 있다(제한 선언 없음). 그래도 스스로 제한을 건다
    - 요청 사이 **1.2초 이상** 쉰다. 동시 요청 안 한다
    - **총 250장까지만.** 전부 긁지 않는다
    - 사이트맵을 쓴다(검색 페이지를 훑지 않는다 = 서버 부담이 적은 정식 경로)

★ 라이선스
    이라스토야는 한 제작물에 **21점 이상 쓰는 상업 이용만 유료**다.
    우리는 비상업 교육·연구 용도이고, **원본 이미지는 깃허브에 커밋하지 않는다.**
    올리는 건 임베딩(숫자)뿐이다. 출처는 문서에 남긴다.
"""
import os, re, io, time, random, urllib.request, urllib.parse
from PIL import Image

OUT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon/data/open_styles/irasutoya"
SIZE, LIMIT, DELAY = 384, 250, 1.2
UA = {"User-Agent": "Mozilla/5.0 (compatible; aiffel-dlthon-research; non-commercial)"}
SITEMAP_PAGES = 32

os.makedirs(OUT, exist_ok=True)


def get(url, timeout=25):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


# ── 1. 사이트맵에서 글 주소를 모은다 ──────────────────────────────────
print("[1] 사이트맵에서 글 주소 수집")
posts = []
for pg in range(1, SITEMAP_PAGES + 1):
    try:
        xml = get(f"https://www.irasutoya.com/sitemap.xml?page={pg}").decode("utf-8", "ignore")
    except Exception as e:
        print(f"    page{pg} 실패: {e}")
        continue
    found = re.findall(r"<loc>([^<]+)</loc>", xml)
    posts += found
    print(f"    page{pg}: {len(found)}건 (누적 {len(posts)})")
    time.sleep(DELAY)

posts = sorted(set(posts))
print(f"    총 글 {len(posts):,}개")

random.seed(20260729)                       # 다시 돌려도 같은 표본
random.shuffle(posts)

# ── 2. 글 하나당 그림 한 장 ──────────────────────────────────────────
print(f"\n[2] 그림 내려받기 (최대 {LIMIT}장, 요청 간 {DELAY}초)")
# ★본문 그림은 <div class="separator"> 안의 <a href> 에 있고 호스트가
#  blogger.googleusercontent.com 이다. bp.blogspot.com 은 사이트 UI 버튼이라 걸러야 한다.
#  (처음에 이걸 헷갈려서 "ランダム" 버튼만 212장 받았다)
SEP = re.compile(r'<div class="separator".*?</div>', re.S)
IMG = re.compile(r'href="(https?://blogger\.googleusercontent\.com/img/[^"]+?\.(?:png|jpg|jpeg))"', re.I)
got, tried = 0, 0

for url in posts:
    if got >= LIMIT:
        break
    tried += 1
    try:
        html = get(url).decode("utf-8", "ignore")
    except Exception:
        time.sleep(DELAY); continue

    # 본문 블록 안의 원본 그림만 고른다 (사이트 UI 버튼 제외)
    cands = []
    for blk in SEP.findall(html):
        cands += IMG.findall(blk)
    if not cands:
        time.sleep(DELAY); continue
    src = cands[0]                     # 글 하나당 대표 그림 한 장

    slug = os.path.splitext(os.path.basename(urllib.parse.urlparse(url).path))[0]
    dst = os.path.join(OUT, f"{slug}.jpg")
    if os.path.exists(dst):
        got += 1; continue

    try:
        raw = get(src)
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))    # 투명배경 -> 흰 배경
        im = Image.alpha_composite(bg, im).convert("RGB")
        im.thumbnail((SIZE, SIZE))
        im.save(dst, quality=92)
        got += 1
        if got % 25 == 0:
            print(f"    {got}장 (시도 {tried})")
    except Exception:
        pass
    time.sleep(DELAY)

print(f"\n끝. {got}장 -> {OUT}")
print("출처: いらすとや (https://www.irasutoya.com/) — 비상업 교육·연구 용도, 원본 미배포")
