"""공개 라이선스 일러스트 세트를 받아 '화풍 클래스'로 정리한다.

폴더 하나 = 화풍 하나. 세트 이름이 곧 라벨이라 라벨링 비용이 0이다.

SVG 는 cairosvg 로 원하는 크기로 렌더링한다.
(원본이 벡터라 해상도를 우리가 정할 수 있다 - 저해상도 PNG 를 늘려 쓰는 것과 다르다)
"""
import os, io, json, time, urllib.request, urllib.error, urllib.parse
import cairosvg
from PIL import Image

OUT_ROOT = "/home/gmw/Documents/AIFFEL_Work/_scratch/ETC/DLthon/data/open_styles"
SIZE = 384
UA = {"User-Agent": "Mozilla/5.0 (aiffel-dlthon-research)"}

# ★깃허브 API 는 무인증이면 시간당 60회라 금방 403 이 난다.
#  gh CLI 토큰을 붙이면 5,000회로 늘어난다. (없으면 그냥 무인증으로 간다)
import subprocess
try:
    _tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                          timeout=10).stdout.strip()
    if _tok:
        UA["Authorization"] = f"Bearer {_tok}"
        print("[인증] gh 토큰 사용 (한도 5,000/시간)")
except Exception:
    print("[인증] gh 토큰 없음 - 무인증(60/시간)으로 진행")

def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(1.5)

def list_dir(repo, path, ref="main"):
    """깃허브 디렉터리 목록. 1000개 넘으면 잘리므로 그 경우는 git tree API 로."""
    out, page = [], 1
    while True:
        u = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(path)}?ref={ref}&per_page=100&page={page}"
        try:
            data = json.loads(get(u))
        except urllib.error.HTTPError as e:
            print(f"    ! {repo}/{path} 목록 실패 ({e.code})"); return out
        if not isinstance(data, list) or not data: break
        out += data
        if len(data) < 100: break
        page += 1
        if page > 40: break
    return out

def list_tree(repo, prefix, ref="main"):
    """★깃허브 contents API 는 폴더당 1000개에서 잘린다. 트리 API 로 한 번에 받는다."""
    u = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    try:
        tree = json.loads(get(u)).get("tree", [])
    except Exception as e:
        print(f"    ! {repo} 트리 실패: {e}"); return []
    out = []
    for t in tree:
        if t.get("type") != "blob":
            continue
        path = t["path"]
        if prefix and not path.startswith(prefix):
            continue
        out.append({"name": os.path.basename(path), "type": "file",
                    "download_url": f"https://raw.githubusercontent.com/{repo}/{ref}/"
                                    + urllib.parse.quote(path)})
    return out


def render_svg(raw, path):
    png = cairosvg.svg2png(bytestring=raw, output_width=SIZE, output_height=SIZE)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))   # 투명배경 -> 흰 배경
    Image.alpha_composite(bg, im).convert("RGB").save(path, quality=92)

def harvest(name, repo, path, ref="main", limit=300, exts=(".svg",)):
    d = os.path.join(OUT_ROOT, name); os.makedirs(d, exist_ok=True)
    files = [f for f in list_tree(repo, path, ref)
             if f["name"].lower().endswith(exts)]
    print(f"  {name:14s} 후보 {len(files)}개 -> {min(limit,len(files))}개 받는다")
    n = 0
    step = max(1, len(files) // limit) if len(files) > limit else 1
    for f in files[::step][:limit]:
        dst = os.path.join(d, os.path.splitext(f["name"])[0] + ".jpg")
        if os.path.exists(dst): n += 1; continue
        try:
            raw = get(f["download_url"])
            if f["name"].lower().endswith(".svg"): render_svg(raw, dst)
            else:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im = im.resize((SIZE, SIZE), Image.LANCZOS); im.save(dst, quality=92)
            n += 1
        except Exception as e:
            pass
        if n % 50 == 0 and n: print(f"      {n}장...")
    print(f"  {name:14s} 완료 {n}장")
    return n

os.makedirs(OUT_ROOT, exist_ok=True)
print("공개 라이선스 화풍 세트 수집")
print(f"저장 위치: {OUT_ROOT}\n")

got = {}
# 1) Twemoji - CC BY 4.0. 굵은 아웃라인 이모지
got["twemoji"] = harvest("twemoji", "jdecked/twemoji", "assets/svg", limit=300)
# 2) Humaaans - CC BY 4.0 (Calinou 미러). 조립형 플랫 인물
# ★기본 브랜치가 master 다(main 아님). 그리고 SVG 는 "Flat Assets/Humaaans" 안에 있다.
got["humaaans"] = harvest("humaaans", "Calinou/humaaans", "Flat Assets",
                          ref="master", limit=300)
# 3) unDraw - 자유 이용(출처 표기 불요). 단색 포인트컬러 플랫 일러스트
got["undraw"] = harvest("undraw", "mkhairi/undraw", "vendor/assets/images/undraw",
                        ref="master", limit=300)
# 4) OpenMoji - CC BY-SA 4.0. 가는 윤곽선 + 채색. twemoji 와 확실히 다른 그림체
got["openmoji"] = harvest("openmoji", "hfg-gmuend/openmoji", "color/svg",
                          ref="master", limit=300)
# 5) Noto Emoji - CC BY 4.0. 둥글고 매끈한 채색
got["notoemoji"] = harvest("notoemoji", "googlefonts/noto-emoji", "svg",
                           ref="main", limit=300)

print("\n" + "=" * 54)
for k, v in got.items():
    print(f"  {k:14s} {v:4d}장")
print("=" * 54)
json.dump(got, open(f"{OUT_ROOT}/_counts.json", "w"), indent=2)
print("\n라이선스 메모:")
print("  twemoji  CC BY 4.0  (그래픽) - 출처 표기 필요")
print("  humaaans CC BY 4.0  (Calinou/humaaans 미러) - 출처 표기 필요")
print("\n* 원본 이미지는 repo 에 커밋하지 않는다. 임베딩만 올린다.")
