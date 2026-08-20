"""
직렬 3단 체인 실험에서 세 서버가 같이 쓰는 것들.
공부용이라 시크릿을 코드에 그냥 박아둔다. 실무면 환경변수나 시크릿 매니저에서 읽어야 한다.
"""
import time
import jwt   # pyjwt

# 세 서버가 나눠 갖는 비밀. 이걸 아는 쪽만 서명을 만들고 검증할 수 있다.
SECRET = "dev-secret-do-not-use-in-real-life"
ALGO = "HS256"

# 게이트웨이가 알고 있는 API Key 목록 (DP06 에서 만든 것과 같은 방식)
# 이름을 영문으로 둔 이유: HTTP 헤더는 latin-1 만 담을 수 있어서
# X-User 헤더에 한글 이름을 실었더니 요청이 500 으로 터졌다 (아래 A 실험에서 처음 밟음).
VALID_API_KEYS = {
    "test-key-001": "userA",
    "test-key-002": "userB",
}


def make_token(user: str, ttl_seconds: int = 300) -> str:
    """사용자 이름을 담은 JWT 를 만든다. ttl 이 지나면 만료된다."""
    now = int(time.time())
    payload = {
        "sub": user,        # 누구인지
        "iat": now,         # 언제 발급했는지
        "exp": now + ttl_seconds,   # 언제까지 유효한지
    }
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def read_token(token: str) -> dict:
    """
    JWT 를 검증하고 안에 든 내용을 돌려준다.
    서명이 안 맞거나 만료됐으면 예외가 난다. (검증은 pyjwt 가 한다)
    """
    return jwt.decode(token, SECRET, algorithms=[ALGO])
