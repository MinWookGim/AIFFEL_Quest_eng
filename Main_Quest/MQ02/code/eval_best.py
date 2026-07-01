from ultralytics import RTDETR
m = RTDETR("runs/rtdetr-l/weights/best.pt")
# 한 번도 안 본 test 셋으로 클래스별 AP 포함 평가
r = m.val(data="data/split/data.yaml", split="test", verbose=True)
print("\n===HEADLINE===")
print("test mAP50   :", round(float(r.box.map50),4))
print("test mAP50-95:", round(float(r.box.map),4))
# 클래스별 AP (ap50: 각 클래스 mAP50, ap: 각 클래스 mAP50-95)
names = m.names
for i, c in enumerate(r.box.ap_class_index):
    print(f"  {names[c]:8} AP50={r.box.ap50[i]:.4f}  AP50-95={r.box.ap[i]:.4f}  P={r.box.p[i]:.4f}  R={r.box.r[i]:.4f}")
