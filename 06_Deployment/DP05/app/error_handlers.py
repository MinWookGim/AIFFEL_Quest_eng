import traceback, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ml_api")

def register_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        logger.error(f"에러: {type(exc).__name__}: {exc}\n경로: {request.method} {request.url}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"success": False, "error": "서버 내부 오류가 발생했습니다."})
