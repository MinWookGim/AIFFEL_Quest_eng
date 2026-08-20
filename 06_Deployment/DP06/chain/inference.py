"""
체인의 맨 뒤 (8022). 실제 추론 대신 결과 흉내만 낸다 — 오늘 보고 싶은 건 인증 쪽이라서다.
같은 일을 두 방식으로 열어둔다.
  /infer_trusted : 앞 서버가 붙여준 X-User 헤더를 그냥 믿는다
  /infer_jwt     : Authorization 헤더의 JWT 를 자기가 직접 검증한다
"""
from fastapi import FastAPI, Header, HTTPException
import jwt

from chain.common import read_token

app = FastAPI(title="inference (8022)")


@app.post("/infer_trusted")
async def infer_trusted(x_user: str = Header(None)):
    # 검증 없이 헤더를 그대로 읽는다. 앞에서 이미 확인했겠거니 하는 구조다.
    return {"홉": "추론(8022)", "내가 본 사용자": x_user, "검증": "안 함 (헤더를 믿음)", "결과": "7"}


@app.post("/infer_jwt")
async def infer_jwt(authorization: str = Header(None)):
    # 앞에서 뭘 했든 상관없이 여기서 다시 서명을 검증한다.
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="추론(8022): 토큰이 없다")
    token = authorization.split(" ", 1)[1]
    try:
        payload = read_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="추론(8022): 토큰이 만료됐다")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"추론(8022): 토큰이 유효하지 않다 ({type(e).__name__})")
    return {"홉": "추론(8022)", "내가 본 사용자": payload["sub"], "검증": "서명 직접 확인", "결과": "7"}
