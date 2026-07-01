"""
MQ02 풍력터빈 손상탐지 — 공통 데이터 split + data.yaml 생성기 (팀 전원 공유용)

★ 왜 이 스크립트가 팀의 "공통 기준"인가:
  - 모델만 다르게 비교하려면 split이 모두 동일해야 한다.
  - seed 고정 + 파일명 정렬 후 섞기 → 누가 어디서 돌려도 "똑같은 split"이 재현된다.
  - 그래서 data.yaml을 공유하는 게 아니라 "이 스크립트"를 공유한다.
    각자 자기 데이터 경로로 한 번 실행하면 동일한 split + 자기 환경용 data.yaml이 생긴다.

★ 설계 결정 (근거):
  - labeled 이미지만 사용(2,995장) → 캐글 예제(baseline)와 같은 조건이라 비교가 공정.
    (정상 10,475장을 배경 음성으로 넣는 건 '개선 실험'으로 따로. 아래 INCLUDE_NORMALS 참고)
  - labels.txt 는 클래스이름 파일이라 제외.
  - Dirt가 1:15로 희귀 → 'Dirt 포함 여부'로 층화(stratify)해서 train/val/test에 고르게 분배.
  - 80/10/10, seed=42. test는 최종 평가용으로 남겨둠(예제엔 없던 부분).
"""
import os, glob, random, shutil

# ── 경로 (★환경마다 바꿀 곳은 여기 "두 줄"이 전부) ──────────────────────────
# 이 스크립트엔 GPU/ROCm 특화 코드가 전혀 없다(순수 파일 복사·분할). 어디서 돌려도 동일.
SRC = "/home/gmw/Documents/AIFFEL_Work/_scratch/Main_Quest/MQ02/data/NordTank586x371"
OUT = "/home/gmw/Documents/AIFFEL_Work/_scratch/Main_Quest/MQ02/data/split"
# ▶ 캐글:  SRC="/kaggle/input/yolo-annotated-wind-turbines-586x371/NordTank586x371"
#          OUT="/kaggle/working/split"
# ▶ 코랩:  SRC="/content/NordTank586x371"   (캐글에서 받아 압축 푼 위치)
#          OUT="/content/split"

SEED = 42
RATIO = (0.8, 0.1, 0.1)        # train / val / test
INCLUDE_NORMALS = 0            # 0=baseline(labeled만). >0 이면 정상이미지를 그 수만큼 train 배경으로 추가(개선실험용)

def main():
    img_dir = os.path.join(SRC, "images")
    lbl_dir = os.path.join(SRC, "labels")

    # 1) annotation 파일만 추림 (labels.txt = 클래스이름 파일이라 제외)
    label_files = sorted(
        f for f in glob.glob(os.path.join(lbl_dir, "*.txt"))
        if os.path.basename(f) != "labels.txt"
    )

    # 2) 각 라벨에 매칭되는 이미지 찾기 + Dirt 포함 여부(층화 키)
    items = []  # (base, img_path, lbl_path, has_dirt)
    for lf in label_files:
        base = os.path.splitext(os.path.basename(lf))[0]
        img_path = os.path.join(img_dir, base + ".png")
        if not os.path.exists(img_path):
            continue
        has_dirt = False
        with open(lf) as fh:
            for line in fh:
                if line.strip().startswith("0 "):
                    has_dirt = True; break
        items.append((base, img_path, lf, has_dirt))

    # 3) Dirt 포함/미포함 두 그룹으로 나눠 각각 같은 비율로 split (= 층화)
    dirt   = sorted([it for it in items if it[3]], key=lambda x: x[0])
    nodirt = sorted([it for it in items if not it[3]], key=lambda x: x[0])

    def split_group(group):
        g = group[:]
        random.Random(SEED).shuffle(g)      # seed 고정 → 어디서 돌려도 동일
        n = len(g); n_tr = int(n*RATIO[0]); n_va = int(n*RATIO[1])
        return g[:n_tr], g[n_tr:n_tr+n_va], g[n_tr+n_va:]

    d_tr, d_va, d_te = split_group(dirt)
    n_tr, n_va, n_te = split_group(nodirt)
    splits = {"train": d_tr+n_tr, "val": d_va+n_va, "test": d_te+n_te}

    # 4) 폴더 구조로 복사 (dataset/{train,val,test}/{images,labels})
    if os.path.exists(OUT): shutil.rmtree(OUT)
    for sp, rows in splits.items():
        os.makedirs(os.path.join(OUT, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT, sp, "labels"), exist_ok=True)
        for base, img_path, lbl_path, _ in rows:
            shutil.copy2(img_path, os.path.join(OUT, sp, "images", base+".png"))
            shutil.copy2(lbl_path, os.path.join(OUT, sp, "labels", base+".txt"))

    # 5) data.yaml 작성 (ultralytics가 읽는 설정)
    yaml_path = os.path.join(OUT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("nc: 2\n")
        f.write("names:\n  0: dirt\n  1: damage\n")

    # 6) 검증 출력 — split별 Dirt/Damage 분포가 고른지 확인
    def stat(rows):
        di = sum(1 for r in rows if r[3]); return len(rows), di, len(rows)-di
    print("="*56)
    for sp in ("train","val","test"):
        tot, di, no = stat(splits[sp])
        print(f"{sp:5} 총 {tot:4}장  | Dirt포함 {di:4}  | Dirt없음 {no:4}  ({di/tot*100:.0f}% dirt)")
    print("="*56)
    print("data.yaml ->", yaml_path)

if __name__ == "__main__":
    main()
