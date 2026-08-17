"""
Day 3 [내 실험 E] - 추론을 별도 서버(ollama)로 넘겼을 때 네 가지 방식 비교

여기까지 온 경위:
  MNIST 추론은 1.4ms 라 너무 가벼워서 run_in_executor 효과가 안 보였다.
  그래서 진짜 무거운 추론이 필요했는데, 마침 로컬에 ollama 가 있다.
  그런데 ollama 는 별도 프로세스라, 내 서버 입장에서 그 1.3초는
  "계산"이 아니라 "남한테 시켜놓고 기다리는 시간"이다. 즉 CPU 바운드가 아니라 I/O 바운드다.
  3장에서 쓴 time.sleep(3) 이 원래 흉내내려던 게 바로 이것이다.

ollama 는 요청을 하나씩 처리한다(3개 동시 = 3.80초, 실측). 그래서 내 서버를 어떻게 짜든
전체 처리 시간은 줄지 않는다. 대신 갈리는 것이 있다 — 추론이 도는 동안 /health 가 응답하는가.
느린 것과 죽은 것은 다르고, 로드밸런서는 그 둘을 헬스체크로 구분한다.
"""
import os
import time
import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import httpx
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


logger = setup_logger("ml_api")

app = FastAPI(
    title="LLM 호출 방식 비교",
    description="I/O 바운드 추론에서 async def / def / run_in_executor / 비동기 클라이언트 비교",
    version="1.0.0-llm",
)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

# 내 환경에서는 ollama 가 localhost 가 아닌 다른 주소에 떠 있어서, 주소는 환경변수로 뺐다
# (원래는 그 주소를 코드에 직접 적어두고 돌렸는데 개인 네트워크 주소라 이렇게 바꿨다.
# 주소만 다르고 재는 내용은 같다)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = "qwen3:4b"

# ③번 전용 스레드풀. 오늘 배운 그 패턴 그대로 4칸
inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")


class Ask(BaseModel):
    prompt: str = "바다에 대해 짧게 설명해줘"


def _payload(prompt: str) -> dict:
    return {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,                      # 생각 과정을 끄면 응답 시간이 일정해진다
        "options": {"num_predict": 80},      # 토큰 수를 고정해야 실험마다 무게가 같다
    }


def call_ollama_sync(prompt: str) -> dict:
    """동기 HTTP 호출. 이 함수 안에는 await 로 넘겨줄 자리가 없다"""
    r = requests.post(OLLAMA_URL, json=_payload(prompt), timeout=180)
    d = r.json()
    return {"text": d["response"][:60], "tokens": d.get("eval_count", 0)}


# ===== ① async def 안에서 동기 HTTP =====
@app.post("/llm/blocking", tags=["LLM"])
async def llm_blocking(req: Ask):
    """겉은 async 인데 안에서 동기 호출을 한다. 이벤트 루프가 통째로 멈춘다"""
    t0 = time.time()
    result = call_ollama_sync(req.prompt)
    return {"method": "① async def + 동기 HTTP", "elapsed": round(time.time() - t0, 2), **result}


# ===== ② 일반 def (FastAPI 가 알아서 스레드풀로 보낸다) =====
@app.post("/llm/threadpool", tags=["LLM"])
def llm_threadpool(req: Ask):
    """async 를 아예 안 쓴다. FastAPI 가 자동으로 별도 스레드에서 돌린다"""
    t0 = time.time()
    result = call_ollama_sync(req.prompt)
    return {"method": "② def (자동 스레드풀)", "elapsed": round(time.time() - t0, 2), **result}


# ===== ③ async def + run_in_executor (오늘 배운 패턴) =====
@app.post("/llm/executor", tags=["LLM"])
async def llm_executor(req: Ask):
    """동기 함수를 내가 만든 스레드풀에 명시적으로 위임한다"""
    t0 = time.time()
    loop = asyncio.get_running_loop()        # async 함수 안에서는 이쪽이 맞다
    result = await loop.run_in_executor(inference_executor, call_ollama_sync, req.prompt)
    return {"method": "③ run_in_executor", "elapsed": round(time.time() - t0, 2), **result}


# ===== ④ async def + 비동기 HTTP 클라이언트 =====
@app.post("/llm/async", tags=["LLM"])
async def llm_async(req: Ask):
    """스레드를 하나도 안 쓴다. 기다리는 동안 이벤트 루프가 다른 요청을 처리한다"""
    t0 = time.time()
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(OLLAMA_URL, json=_payload(req.prompt))   # 여기가 진짜 await
    d = r.json()
    return {
        "method": "④ 비동기 클라이언트",
        "elapsed": round(time.time() - t0, 2),
        "text": d["response"][:60],
        "tokens": d.get("eval_count", 0),
    }


# ===== 판정용: 이 요청이 즉시 돌아오는지가 오늘의 시험이다 =====
@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy"}


# ===== 스레드 사용량 측정 장치 =====
#
# ②③④ 는 응답 시간으로는 안 갈린다(3·4·5ms = 잡음). 갈리는 축은 시간이 아니라
# "요청 하나를 처리하는 데 스레드를 몇 개 쓰느냐"다.
#
# 그런데 서버에 "지금 스레드 몇 개냐"고 물어보는 것도 요청이라, ①번을 재는 동안엔
# 그 질문마저 이벤트 루프에 막혀서 못 물어본다.
# 그래서 이벤트 루프와 무관한 별도 스레드가 계속 세어서 쌓아두게 하고,
# 부하가 다 끝난 뒤에 "아까 그 구간의 최댓값"을 꺼내 보는 방식으로 짠다.

_samples = deque(maxlen=4000)      # (시각, 전체, inference풀, AnyIO풀)


def _sampler():
    while True:
        names = [t.name for t in threading.enumerate()]
        _samples.append((
            time.time(),
            len(names),
            sum(1 for n in names if n.startswith("inference")),   # ③ 이 쓰는 내 전용 풀
            sum(1 for n in names if "AnyIO" in n),                # ② 가 쓰는 FastAPI 기본 풀
        ))
        time.sleep(0.05)


threading.Thread(target=_sampler, daemon=True, name="sampler").start()


@app.get("/debug/threads", tags=["System"])
async def debug_threads(since: float):
    """since 시각 이후 구간의 스레드 수 최댓값을 돌려준다"""
    window = [s for s in _samples if s[0] >= since]
    if not window:
        return {"error": "표본 없음"}
    return {
        "표본수": len(window),
        "전체최대": max(s[1] for s in window),
        "inference풀최대": max(s[2] for s in window),
        "AnyIO풀최대": max(s[3] for s in window),
        "평상시전체": _samples[0][1] if _samples else None,
    }


# ===== 글로벌 에러 핸들러가 실제로 걸리는지 보기 위한 경로 =====
#
# main_final.py 는 모든 엔드포인트가 try/except 로 감싸여 있어서 400·422 로 먼저 잡힌다.
# 그래서 오늘 만든 글로벌 핸들러(500)는 만들어놓고 한 번도 걸린 적이 없었다.
# 안전망이 실제로 받아내는지 보려면 잡히지 않은 예외가 필요해서 일부러 하나 만들었다.
@app.get("/debug/boom", tags=["System"])
async def debug_boom():
    raise RuntimeError("일부러 낸 에러 - 글로벌 핸들러가 받아내는지 보려는 것")
