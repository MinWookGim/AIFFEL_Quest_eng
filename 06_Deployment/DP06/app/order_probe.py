"""
인증과 파일 업로드 중 무엇이 먼저 실행되는지 재보려고 만든 실험용 서버.
제출용 서버가 아니라 순서만 보려고 만든 것이다.
"""
import time
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header

app = FastAPI()

# 언제 무슨 일이 있었는지 순서대로 쌓아둔다
EVENTS = []

def mark(what):
    EVENTS.append((time.perf_counter(), what))


@app.middleware("http")
async def on_request(request, call_next):
    # 미들웨어는 헤더까지 읽힌 시점에 먼저 실행된다
    mark("요청 도착")
    resp = await call_next(request)
    mark("응답 반환")
    return resp


async def verify_key(x_api_key: str = Header(None)):
    # 이 함수가 언제 실행되는지가 오늘 보고 싶은 것이다
    mark("인증 함수 실행")
    if x_api_key != "test-key-001":
        raise HTTPException(status_code=401, detail="유효하지 않은 API Key입니다.")
    return "사용자A"


@app.post("/probe")
async def probe(
    file: UploadFile = File(...),
    user: str = Depends(verify_key),
):
    mark("엔드포인트 본문 시작")
    data = await file.read()
    return {"size": len(data)}
