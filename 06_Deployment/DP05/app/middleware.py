import time, logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ml_api")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = round(time.time() - start, 3)
        msg = f"{request.method} {request.url.path} -> {response.status_code} ({duration}s)"
        if response.status_code >= 500: logger.error(msg)
        elif response.status_code >= 400: logger.warning(msg)
        else: logger.info(msg)
        response.headers["X-Process-Time"] = str(duration)
        return response
