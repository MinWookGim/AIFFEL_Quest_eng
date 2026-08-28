# -*- coding: utf-8 -*-
"""UI 테스트 B 보강 — temperature 를 바꾸면 응답이 실제로 달라지나."""
import requests, itertools
API, KEY = "http://127.0.0.1:8508", "test-key-001"
Q, N = "주말에 뭐 하면 좋을까?", 5

def ask(temp):
    r = requests.post(f"{API}/chat",
        json={"messages":[{"role":"user","content":Q}], "max_new_tokens":40, "temperature":temp},
        headers={"X-API-Key":KEY}, timeout=120)
    return r.json()["response"].strip().replace("\n", " ")

def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

print(f'질문 "{Q}" 를 temperature 별로 {N} 번씩 물어본다.\n')
for t in (0.1, 0.8, 1.5):
    outs = [ask(t) for _ in range(N)]
    pairs = list(itertools.combinations(outs, 2))
    avg = sum(common_prefix(a, b) for a, b in pairs) / len(pairs)
    print(f"temperature {t}")
    print(f"  {N}개 중 서로 다른 답: {len(set(outs))}개 / 답끼리 같은 앞부분 평균 {avg:.1f}글자")
    for i, o in enumerate(outs, 1):
        print(f"    {i}. {o[:78]}")
    print()
