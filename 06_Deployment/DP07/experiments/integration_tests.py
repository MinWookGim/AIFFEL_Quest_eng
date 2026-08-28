# -*- coding: utf-8 -*-
"""Day7 통합 테스트 1~4. 노트북 6.2절과 같은 내용이고 포트만 8508 이다."""
import requests, time
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://127.0.0.1:8508"
KEY = "test-key-001"
print("=" * 60); print("  통합 테스트 (포트 8508)"); print("=" * 60)
print(f"\n[헬스체크] {requests.get(f'{API}/health').json()}")

print("\n[테스트 1] 인증")
r = requests.post(f"{API}/chat", json={"messages":[{"role":"user","content":"안녕"}]})
print(f"  인증 없음    -> HTTP {r.status_code}  {r.json().get('detail','')}")
r = requests.post(f"{API}/chat", json={"messages":[{"role":"user","content":"안녕"}]},
                  headers={"X-API-Key":"wrong-key"})
print(f"  잘못된 키    -> HTTP {r.status_code}  {r.json().get('detail','')}")
t = time.time()
r = requests.post(f"{API}/chat", json={"messages":[{"role":"user","content":"안녕"}]},
                  headers={"X-API-Key":KEY})
print(f"  올바른 키    -> HTTP {r.status_code}  ({time.time()-t:.1f}초)")
if r.status_code == 200:
    j = r.json()
    print(f"  응답: {j['response']}")
    print(f"  user 필드: {j['user']} / model: {j['model_name']}")

print("\n[테스트 2] 멀티턴 대화")
messages = []
for user_msg in ["안녕하세요!", "오늘 뭐 하면 좋을까?", "맛있는 거 추천해줘"]:
    messages.append({"role":"user","content":user_msg})
    r = requests.post(f"{API}/chat", json={"messages":messages,"max_new_tokens":50},
                      headers={"X-API-Key":KEY})
    bot = r.json()["response"]
    messages.append({"role":"bot","content":bot})
    print(f"  사용자: {user_msg}")
    print(f"  봇:    {bot}")
print(f"  총 대화 턴: {len(messages)//2}  (보낸 메시지 수 {len(messages)})")

print("\n[테스트 3] 입력 검증")
for label, body in [
    ("빈 메시지 목록     ", {"messages":[]}),
    ("temperature 5.0    ", {"messages":[{"role":"user","content":"테스트"}], "temperature":5.0}),
    ("빈 문자열 content  ", {"messages":[{"role":"user","content":""}]}),
    ("max_new_tokens 9   ", {"messages":[{"role":"user","content":"테스트"}], "max_new_tokens":9}),
]:
    r = requests.post(f"{API}/chat", json=body, headers={"X-API-Key":KEY})
    print(f"  {label} -> HTTP {r.status_code}")

print("\n[테스트 4] 동시 요청 (4개)")
def send(i):
    t = time.time()
    r = requests.post(f"{API}/chat",
        json={"messages":[{"role":"user","content":f"질문 {i+1}번입니다"}],"max_new_tokens":30},
        headers={"X-API-Key":KEY}, timeout=120)
    return {"id":i+1, "elapsed":round(time.time()-t,1), "status":r.status_code}
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    res = [f.result() for f in as_completed([ex.submit(send,i) for i in range(4)])]
total = round(time.time()-t0, 1)
for r_ in sorted(res, key=lambda x: x["id"]):
    print(f"  요청 #{r_['id']}: {r_['elapsed']}초 (HTTP {r_['status']})")
print(f"  전체: {total}초   (순차였다면 대략 {sum(x['elapsed'] for x in res):.1f}초)")
