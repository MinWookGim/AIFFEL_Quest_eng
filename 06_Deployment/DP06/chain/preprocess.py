"""
체인의 가운데 (8021). 자기 일을 하고 뒤(8022)로 넘긴다.
  /run_trusted : X-User 헤더를 믿고, 그대로 뒤로 전달
  /run_jwt     : JWT 를 자기도 검증하고, 토큰을 그대로 뒤로 전달
  ?slow=초     : 일부러 늦게 처리한다 (만료 실험용)
"""
import asyncio

import httpx
import jwt
from fastapi import FastAPI, Header, HTTPException

from chain.common import read_token

app = FastAPI(title="preprocess (8021)")
INFERENCE = "http://127.0.0.1:8022"


@app.post("/run_trusted")
async def run_trusted(slow: float = 0, x_user: str = Header(None)):
    if slow:
        await asyncio.sleep(slow)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{INFERENCE}/infer_trusted", headers={"X-User": x_user or ""})
    return {"홉": "전처리(8021)", "내가 본 사용자": x_user, "검증": "안 함 (헤더를 믿음)", "뒤에서 온 것": r.json()}


@app.post("/run_jwt")
async def run_jwt(slow: float = 0, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="전처리(8021): 토큰이 없다")
    token = authorization.split(" ", 1)[1]
    try:
        payload = read_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="전처리(8021): 토큰이 만료됐다")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"전처리(8021): 토큰이 유효하지 않다 ({type(e).__name__})")

    if slow:
        await asyncio.sleep(slow)   # 여기서 시간을 끄는 동안 토큰이 만료될 수 있다

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{INFERENCE}/infer_jwt", headers={"Authorization": authorization})
    if r.status_code != 200:
        # 뒤에서 막힌 것을 그대로 올려보낸다
        raise HTTPException(status_code=r.status_code, detail=r.json().get("detail"))
    return {"홉": "전처리(8021)", "내가 본 사용자": payload["sub"], "검증": "서명 직접 확인", "뒤에서 온 것": r.json()}
