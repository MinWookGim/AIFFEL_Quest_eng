from ultralytics import YOLO

# 코랩에서 받아온 YOLO 4종 best.pt 를 한 번도 안 본 test 셋으로 평가.
# 로컬 ROCm 에서 YOLO 계열은 네이티브로 잘 도니 우회(HSA_OVERRIDE) 불필요.
models = ["yolov8s", "yolo11s", "yolov10s", "yolo12s"]

rows = []
for name in models:
    m = YOLO(f"colab_results/{name}/weights/best.pt")
    # split="test" -> data.yaml 의 test/images 로 평가. 클래스별 AP 포함.
    r = m.val(data="data/split/data.yaml", split="test", verbose=False)
    names = m.names
    # 클래스별로 dict 에 담아두기 (인덱스 순서가 뒤섞일 수 있어 이름 기준)
    per = {}
    for i, c in enumerate(r.box.ap_class_index):
        per[names[c]] = {
            "AP50": float(r.box.ap50[i]),
            "AP50_95": float(r.box.ap[i]),
            "P": float(r.box.p[i]),
            "R": float(r.box.r[i]),
        }
    rows.append({
        "model": name,
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "dirt": per.get("dirt", {}),
        "damage": per.get("damage", {}),
    })

# 요약 출력
print("\n\n===== 5행 비교표 재료 (test split) =====")
hdr = f"{'model':10} {'mAP50':>7} {'mAP5095':>8} | {'dirtAP50':>8} {'dirtR':>6} | {'dmgAP50':>7} {'dmgR':>6}"
print(hdr)
print("-" * len(hdr))
for x in rows:
    d, g = x["dirt"], x["damage"]
    print(f"{x['model']:10} {x['mAP50']:7.4f} {x['mAP50_95']:8.4f} | "
          f"{d.get('AP50',0):8.4f} {d.get('R',0):6.3f} | "
          f"{g.get('AP50',0):7.4f} {g.get('R',0):6.3f}")

# 파일로도 저장 (표 작성용)
import json
with open("verify/colab_yolo_test_results.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print("\n저장: verify/colab_yolo_test_results.json")
