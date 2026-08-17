"""
Day 3 [내 실험 H] - CPU 바운드 결정 실험

오늘 원래 물었던 것: 파이토치 추론에서 스레드풀은 진짜로 겹쳐 도는가?
GIL 때문에 파이썬 코드는 한 번에 한 스레드만 돈다고 배웠다. 그렇다면 스레드를 4개 줘도
CPU 작업은 안 빨라져야 한다. 그런데 실제로 그런지는 아직 안 재봤다.

실험 C 는 MNIST 추론이 1.4ms 라 너무 가벼워서 못 갈랐고,
실험 E~G 는 ollama 로 갔는데 그건 별도 프로세스라 I/O 바운드였다. CPU 바운드는 미결로 남았다.

여기서 통제하는 것:
  1) torch.set_num_threads(1)
     파이토치는 기본으로 코어를 여러 개 쓴다. 그러면 요청 하나가 이미 CPU 를 다 먹어서
     스레드를 4개 줘도 남는 코어가 없다. 그건 GIL 때문이 아닌데 GIL 탓으로 오해하게 된다.
     한 요청 = 한 코어로 묶어야 스레드 효과만 따로 볼 수 있다. (이 기계는 32코어라 자리는 충분)
  2) repeat 파라미터로 작업 무게 조절
     순전파 1회 = 0.302ms 이고 repeat 1~400 에서 완벽히 선형임을 미리 확인했다.
     repeat=1000 이면 302ms 짜리 추론이 된다.
  3) 요청에 몸통이 없다 (GET + 쿼리)
     784개 실수를 JSON 으로 만드는 비용이 실험 A~D 에서 병목이었다. 그걸 없앤다.
"""
import os
import time
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI

from app.model_utils import load_model
from app.middleware import RequestLoggingMiddleware
from app.error_handlers import register_error_handlers


# [실험 I 추가] 두 값을 환경변수로 뺐다. 실험 H 에서는 전제로 고정해뒀던 것을
# 이제 변수로 풀어서, 칸 수와 torch 스레드 수가 곱해지는지 보려는 것이다
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "1"))
POOL_SIZE = int(os.environ.get("POOL_SIZE", "4"))

torch.set_num_threads(TORCH_THREADS)

app = FastAPI(title="CPU 바운드 결정 실험", version="1.0.0-cpu")

# [실험 I 추가] X-Process-Time 헤더가 대기 시간까지 포함하는지 보려고 미들웨어를 붙였다.
# 부하 중에 로그가 쏟아지면 결과를 못 보니 레벨은 올려둔다
logging.getLogger("ml_api").setLevel(logging.WARNING)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)     # [실험 K 추가] 글로벌 핸들러가 스레드까지 닿는지 보려면 등록돼 있어야 한다

MODEL = load_model("models/mnist_state_dict.pth")
SAMPLE = torch.randn(1, 1, 28, 28)
inference_executor = ThreadPoolExecutor(max_workers=POOL_SIZE, thread_name_prefix="inference")


def heavy_inference(repeat: int) -> dict:
    """순전파를 repeat 번 반복한다. 순수 CPU 작업이고 기다리는 구간이 없다"""
    t0 = time.time()
    out = None
    with torch.no_grad():
        for _ in range(repeat):          # repeat=0 이면 아무 일도 안 한다.
            out = MODEL(SAMPLE)          # 순수 왕복 비용만 재려고 넣어둔 값이다
    # argmax 는 반복문 밖에서 한 번만 한다. 안에 넣으면 순전파 1회 비용이 달라져서
    # 실험 H 에서 잰 0.302ms 와 비교가 안 된다
    label = -1 if out is None else int(out.argmax().item())
    return {"repeat": repeat, "elapsed": round(time.time() - t0, 4), "label": label}


# ===== ① async def 안에서 그냥 호출 =====
@app.get("/cpu/blocking")
async def cpu_blocking(repeat: int = 100):
    return {"method": "① async def + 직접 호출", **heavy_inference(repeat)}


# ===== ② 일반 def (FastAPI 가 자동으로 스레드풀에 보낸다) =====
@app.get("/cpu/threadpool")
def cpu_threadpool(repeat: int = 100):
    return {"method": "② def (자동 스레드풀)", **heavy_inference(repeat)}


# ===== ③ async def + run_in_executor (풀 4칸) =====
@app.get("/cpu/executor")
async def cpu_executor(repeat: int = 100):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(inference_executor, heavy_inference, repeat)
    return {"method": "③ run_in_executor", **result}


# ===== [실험 K] 예외가 어디서 터지느냐에 따라 달라지는지 =====
#
# /debug/boom 으로 확인한 것은 이벤트 루프에서 터진 예외 하나뿐이었다.
# 그런데 오늘 만든 구조에서 추론이 실제로 도는 자리는 스레드풀 안이다.
# 거기서 터지면 ①글로벌 핸들러까지 오는가 ②그 워커가 죽어서 풀이 줄어드는가.
# ②가 고약하다. 에러가 날 때마다 처리 능력이 깎이는데 겉으로는 500 만 보인다.

def _raise_in_worker():
    raise RuntimeError("스레드풀 워커 안에서 일부러 낸 에러")


@app.get("/debug/boom-in-thread")
async def boom_in_thread():
    """run_in_executor 로 넘긴 스레드 안에서 터진다"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(inference_executor, _raise_in_worker)


@app.get("/debug/boom-in-def")
def boom_in_def():
    """일반 def 라 FastAPI 기본 스레드풀에서 돈다. 거기서 터진다"""
    _raise_in_worker()


@app.get("/debug/boom-in-loop")
async def boom_in_loop():
    """비교용. 이벤트 루프에서 그냥 터진다"""
    _raise_in_worker()


@app.get("/debug/threads")
async def debug_threads():
    """지금 이 순간의 스레드 수. 예외 전후로 읽어서 풀이 줄었는지 본다"""
    names = [t.name for t in threading.enumerate()]
    return {
        "전체": len(names),
        "inference풀": sum(1 for n in names if n.startswith("inference")),
        "AnyIO풀": sum(1 for n in names if "AnyIO" in n),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "torch_threads": torch.get_num_threads(), "pool": POOL_SIZE}
