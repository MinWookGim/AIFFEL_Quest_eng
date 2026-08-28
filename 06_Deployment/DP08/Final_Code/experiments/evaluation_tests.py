# -*- coding: utf-8 -*-
"""
Day 8 평가 기준 다섯 항목을 그대로 테스트로 만든 것.
  1 서버가 정상 실행되는가          2 추론이 동작하는가
  3 API Key 없이 요청하면 401 인가  4 잘못된 입력에 적절한 에러가 나오는가
  5 (UI 는 브라우저로 따로 확인)
"""
import io, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API, KEY = "http://127.0.0.1:8509", "test-key-001"
S = "../DP08/verify/samples"

def post(img="cats_remote.jpg", key=KEY, thr=0.5, name="cats_remote.jpg", mime="image/jpeg", raw=None):
    files = {"file": (name, raw if raw is not None else open(f"{S}/{img}", "rb").read(), mime)}
    h = {"X-API-Key": key} if key else {}
    return requests.post(f"{API}/predict", files=files, params={"threshold": thr}, headers=h, timeout=120)

print("="*64); print("  Day 8 평가 기준 점검"); print("="*64)

print("\n[1] 서버가 정상적으로 실행되는가")
h = requests.get(f"{API}/health").json()
print(f"  /health -> {h}")

print("\n[2] 추론이 동작하는가 (샘플 3장)")
for img in ("cats_remote.jpg", "bear.jpg", "street.jpg"):
    r = post(img)
    j = r.json()
    names = ", ".join(f'{d["label"]}({d["score"]:.2f})' for d in j["detections"][:6])
    print(f"  {img:<16} HTTP {r.status_code}  {j['count']}개  {j['elapsed_ms']}ms  이미지 {j['image_size']}")
    print(f"      {names}{' ...' if j['count'] > 6 else ''}")

print("\n[3] API Key 없이 / 틀린 키로 요청하면")
r = post(key=None);        print(f"  헤더 없음     -> HTTP {r.status_code}  {r.json().get('detail','')}")
r = post(key="wrong-key"); print(f"  틀린 키       -> HTTP {r.status_code}  {r.json().get('detail','')}")
r = post(key=KEY);         print(f"  올바른 키     -> HTTP {r.status_code}")

print("\n[4] 잘못된 입력")
r = post(raw=b"this is not an image at all", name="fake.txt", mime="text/plain")
print(f"  텍스트 파일          -> HTTP {r.status_code}  {r.json().get('detail','')[:60]}")
r = post(raw=b"MZ\x90\x00 not really a png", name="fake.png", mime="image/png")
print(f"  png 로 위장한 파일   -> HTTP {r.status_code}  {r.json().get('detail','')[:60]}")
r = post(raw=b"\xff\xd8\xff" + b"\x00" * (6*1024*1024), name="big.jpg")
print(f"  6MB 파일             -> HTTP {r.status_code}  {r.json().get('detail','')[:60]}")
r = post(thr=1.5)
print(f"  기준값 1.5 (범위밖)  -> HTTP {r.status_code}")
r = post(thr=-0.1)
print(f"  기준값 -0.1 (범위밖) -> HTTP {r.status_code}")

print("\n[추가] 같은 이미지를 기준값만 바꿔 보면")
for thr in (0.05, 0.3, 0.5, 0.7, 0.9, 0.95):
    j = post("street.jpg", thr=thr).json()
    labs = {}
    for d in j["detections"]: labs[d["label"]] = labs.get(d["label"], 0) + 1
    top = ", ".join(f"{k}x{v}" for k, v in sorted(labs.items(), key=lambda x: -x[1])[:5])
    print(f"  기준값 {thr:<5} -> {j['count']:>3}개   {top}")

print("\n[추가] 동시 요청 4개")
def send(i):
    t = time.time(); r = post("street.jpg")
    return i+1, round(time.time()-t, 2), r.status_code
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    res = [f.result() for f in as_completed([ex.submit(send, i) for i in range(4)])]
for i, e, s in sorted(res):
    print(f"  요청 #{i}: {e}초 (HTTP {s})")
print(f"  전체 {round(time.time()-t0,2)}초 (순차였다면 대략 {sum(e for _,e,_ in res):.2f}초)")

print("\n[추가] 같은 이미지를 두 번 연속 보내면 (처음 보는 크기 대 이미 본 크기)")
for img in ("cats_remote.jpg", "bear.jpg", "street.jpg"):
    ms = [post(img).json()["elapsed_ms"] for _ in range(3)]
    print(f"  {img:<16} 1회 {ms[0]:>7.1f}ms   2회 {ms[1]:>7.1f}ms   3회 {ms[2]:>7.1f}ms")
