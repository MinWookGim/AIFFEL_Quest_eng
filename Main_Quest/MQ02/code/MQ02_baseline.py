"""
================================================================
 MQ02 풍력터빈 손상탐지 — 팀 공통 베이스 코드 (이 파일 하나면 됩니다)
================================================================
 하는 일:  데이터 split(자동)  →  내 모델 학습  →  결과 출력
 사용법 :  맨 위 [설정]에서 "MODEL"과 "SRC 경로"만 내 환경에 맞게 바꾸고 실행!
           (캐글/코랩/로컬 경로 예시는 SRC 주석 참고)
 ★ 학습 설정(epochs/imgsz/batch/seed)은 팀 전원 동일해야 비교가 공정합니다 — 건드리지 마세요.
"""
import os, glob, random, shutil
import torch
# GPU 있으면 GPU(0), 없으면 CPU 자동 선택 — LMS/코랩/캐글/로컬 어디서나 그냥 돌아가게
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ══════════════════ [설정] 여기 두 개만 바꾸세요 ══════════════════
MODEL = "yolov8s.pt"   # ★ 내가 배정받은 모델. 예: yolo11s.pt / yolov10s.pt / yolo12s.pt / rtdetr-l.pt

SRC = "/home/gmw/Documents/AIFFEL_Work/_scratch/Main_Quest/MQ02/data/NordTank586x371"   # ★ 데이터 위치
#  ▶ 캐글:  SRC = "/kaggle/input/yolo-annotated-wind-turbines-586x371/NordTank586x371"
#  ▶ 코랩:  SRC = "/content/NordTank586x371"

OUT = "./split"        # split 결과가 저장될 곳 (보통 안 바꿔도 됨)
# ─────────────────── 아래 설정은 전원 동일, 건드리지 마세요 ───────────────────
EPOCHS, IMGSZ, BATCH, SEED = 50, 640, 16, 42
RATIO = (0.8, 0.1, 0.1)   # train / val / test
# ════════════════════════════════════════════════════════════════


def prepare_split():
    """데이터를 train/val/test로 나눠 split 폴더 + data.yaml 생성.
       Dirt가 희귀(1:15)해서 'Dirt 포함 여부'로 층화 분할한다. seed 고정 → 모두 동일."""
    img_dir, lbl_dir = os.path.join(SRC, "images"), os.path.join(SRC, "labels")
    # labels.txt 는 클래스이름 파일이라 제외
    label_files = sorted(f for f in glob.glob(os.path.join(lbl_dir, "*.txt"))
                         if os.path.basename(f) != "labels.txt")
    items = []  # (base, img, lbl, has_dirt)
    for lf in label_files:
        base = os.path.splitext(os.path.basename(lf))[0]
        img = os.path.join(img_dir, base + ".png")
        if not os.path.exists(img):
            continue
        has_dirt = any(line.strip().startswith("0 ") for line in open(lf))
        items.append((base, img, lf, has_dirt))

    def split_group(group):
        g = sorted(group, key=lambda x: x[0]); random.Random(SEED).shuffle(g)
        n = len(g); a = int(n*RATIO[0]); b = int(n*RATIO[1])
        return g[:a], g[a:a+b], g[a+b:]

    dirt   = [it for it in items if it[3]]
    nodirt = [it for it in items if not it[3]]
    d = split_group(dirt); nd = split_group(nodirt)
    splits = {"train": d[0]+nd[0], "val": d[1]+nd[1], "test": d[2]+nd[2]}

    if os.path.exists(OUT): shutil.rmtree(OUT)
    for sp, rows in splits.items():
        os.makedirs(os.path.join(OUT, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT, sp, "labels"), exist_ok=True)
        for base, img, lbl, _ in rows:
            shutil.copy2(img, os.path.join(OUT, sp, "images", base+".png"))
            shutil.copy2(lbl, os.path.join(OUT, sp, "labels", base+".txt"))

    yaml_path = os.path.abspath(os.path.join(OUT, "data.yaml"))
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n")
        f.write("nc: 2\nnames:\n  0: dirt\n  1: damage\n")
    for sp in ("train","val","test"):
        rows = splits[sp]; di = sum(1 for r in rows if r[3])
        print(f"  {sp:5} {len(rows):4}장 (Dirt포함 {di})")
    return yaml_path


def main():
    print("[1/2] 데이터 split 생성...")
    data_yaml = prepare_split()
    print("      data.yaml:", data_yaml)

    print(f"[2/2] 학습 시작 — 모델: {MODEL}")
    # rtdetr 만 클래스가 다름, 나머지(yolo계열)는 전부 YOLO
    if MODEL.lower().startswith("rtdetr"):
        from ultralytics import RTDETR as Net
    else:
        from ultralytics import YOLO as Net
    model = Net(MODEL)
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                device=DEVICE, seed=SEED, project="runs", name=MODEL.split(".")[0],
                exist_ok=True)

    # 최종: 한 번도 안 본 test 셋으로 정직하게 평가 (클래스별 AP 포함)
    print("[+] test 셋 최종 평가...")
    metrics = model.val(data=data_yaml, split="test")
    print("    test mAP50   :", round(float(metrics.box.map50), 4))
    print("    test mAP50-95:", round(float(metrics.box.map), 4))


if __name__ == "__main__":
    main()
