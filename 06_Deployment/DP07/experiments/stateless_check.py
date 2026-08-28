# -*- coding: utf-8 -*-
"""멀티턴을 기억하는 건 서버인가 클라이언트인가. 대화 기록을 빼고 같은 질문을 던져 본다."""
import requests
API, KEY = "http://127.0.0.1:8508", "test-key-001"
H = {"X-API-Key": KEY}

def chat(msgs, n=60):
    return requests.post(f"{API}/chat", json={"messages":msgs,"max_new_tokens":n,"temperature":0.3},
                         headers=H, timeout=120).json()["response"].strip().replace("\n"," ")

print("[1] 이름을 알려준다")
a = chat([{"role":"user","content":"내 이름은 민욱이야. 기억해줘"}])
print("   봇:", a[:90])
print("\n[2] 기록을 같이 보내고 다시 묻는다")
b = chat([{"role":"user","content":"내 이름은 민욱이야. 기억해줘"},
          {"role":"bot","content":a},
          {"role":"user","content":"내 이름이 뭐라고 했지?"}])
print("   봇:", b[:90]); print("   '민욱' 들어있나:", "민욱" in b)
print("\n[3] 같은 서버에 기록 없이 그 질문만 보낸다")
c = chat([{"role":"user","content":"내 이름이 뭐라고 했지?"}])
print("   봇:", c[:90]); print("   '민욱' 들어있나:", "민욱" in c)
