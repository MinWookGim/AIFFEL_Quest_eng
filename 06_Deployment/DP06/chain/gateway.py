"""
체인의 맨 앞 (8020). 바깥에서 오는 요청은 여기만 받는다.
  /token         : API Key 를 확인하고 JWT 를 발급한다 (ttl 로 유효기간 조절)
  /predict/naive : API Key 확인 후, 뒤에는 X-User 헤더만 붙여서 넘긴다
  /predict/jwt   : JWT 를 확인하고, 토큰을 그대로 뒤로 넘긴다
"""
import httpx
import jwt
from fastapi import FastAPI, Header, HTTPException

from chain.common import VALID_API_KEYS, make_token, read_token

app = FastAPI(title="gateway (8020)")
PREPROCESS = "http://127.0.0.1:8021"


def check_api_key(x_api_key):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="게이트웨이(8020): API Key 가 유효하지 않다")
    return VALID_API_KEYS[x_api_key]


@app.post("/token")
async def issue_token(ttl: int = 300, x_api_key: str = Header(None)):
    user = check_api_key(x_api_key)
    return {"access_token": make_token(user, ttl), "token_type": "bearer", "expires_in": ttl}


@app.post("/predict/naive")
async def predict_naive(slow: float = 0, x_api_key: str = Header(None)):
    user = check_api_key(x_api_key)
    # 뒤에는 "이 사람이다" 라고 헤더로만 알려준다. 뒷단은 그걸 확인할 방법이 없다.
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{PREPROCESS}/run_trusted", params={"slow": slow}, headers={"X-User": user})
    return {"홉": "게이트웨이(8020)", "내가 본 사용자": user, "검증": "API Key 확인", "뒤에서 온 것": r.json()}


@app.post("/predict/jwt")
async def predict_jwt(slow: float = 0, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="게이트웨이(8020): 토큰이 없다")
    token = authorization.split(" ", 1)[1]
    try:
        payload = read_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="게이트웨이(8020): 토큰이 만료됐다")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"게이트웨이(8020): 토큰이 유효하지 않다 ({type(e).__name__})")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{PREPROCESS}/run_jwt", params={"slow": slow},
                              headers={"Authorization": authorization})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.json().get("detail"))
    return {"홉": "게이트웨이(8020)", "내가 본 사용자": payload["sub"], "검증": "서명 확인", "뒤에서 온 것": r.json()}
