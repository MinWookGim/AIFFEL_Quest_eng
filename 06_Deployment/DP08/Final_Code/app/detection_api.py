"""
Day 8 - 객체 탐지 FastAPI 서버 (자율 프로젝트)

Day 1~7 에서 하나씩 붙여온 것들을 그대로 모아서 만든다.
  인증        Day 6 의 app/auth.py 를 고치지 않고 그대로 가져다 쓴다
  로깅·에러   Day 3 의 logger_config, error_handlers, middleware
  파일 업로드 Day 6 의 검증 순서를 따르되 원본 컬러 이미지를 쓰도록 바꾼 detection_utils
  비동기 추론 Day 3 에서 배운 run_in_executor 로 이벤트 루프를 막지 않는다
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, File, HTTPException, Query, UploadFile

from app.auth import verify_api_key
from app.detection_model import DetectionModel
from app.detection_schemas import DetectionResponse
from app.detection_utils import validate_and_read_rgb
from app.error_handlers import register_error_handlers
from app.logger_config import setup_logger
from app.middleware import RequestLoggingMiddleware

logger = setup_logger("detection_api")

# 추론은 CPU/GPU 를 오래 붙잡는 일이라 별도 스레드에서 돌린다.
# 자리를 2개로 둔 건 Day 7 과 같다. 그보다 늘리면 서로 GPU 를 다투게 된다.
inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="detect")

detector: DetectionModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 뜰 때 모델을 딱 한 번만 올린다. 요청마다 올리면 매번 느려진다."""
    global detector
    logger.info("객체 탐지 모델 로드 중: facebook/detr-resnet-50")
    detector = DetectionModel("facebook/detr-resnet-50")
    logger.info(f"모델 로드 완료 (device={detector.device}, 라벨 {len(detector.id2label)}종)")
    yield
    logger.info("서버를 내린다")


app = FastAPI(
    title="Object Detection API",
    description="DETR 로 이미지에서 물체를 찾아 위치와 이름을 돌려주는 API (인증 필요)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)


def run_detection(image, threshold):
    """별도 스레드에서 도는 추론 함수."""
    if detector is None:
        raise RuntimeError("모델이 아직 로드되지 않았습니다")
    return detector.predict(image, threshold=threshold)


@app.get("/health", tags=["System"])
async def health_check():
    """서버와 모델이 준비됐는지 알려준다. 여기에는 인증을 걸지 않는다."""
    return {
        "status": "healthy" if detector else "loading",
        "model": detector.model_name if detector else None,
        "device": detector.device if detector else None,
        "num_labels": len(detector.id2label) if detector else None,
    }


@app.post("/predict", response_model=DetectionResponse, tags=["Detection"])
async def predict(
    file: UploadFile = File(..., description="PNG 또는 JPEG 이미지 (최대 5MB)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="이 확신도 이상만 돌려준다"),
    user: str = Depends(verify_api_key),
):
    """이미지 하나를 받아 그 안의 물체 목록을 돌려준다."""
    logger.info(f"탐지 요청 — 사용자: {user}, 파일: {file.filename}, 기준값: {threshold}")

    # 1. 업로드 파일 검증 (타입 -> 크기 -> 디코딩) 후 원본 RGB 이미지로 받는다.
    image = await validate_and_read_rgb(file)

    # 2. 추론은 스레드로 넘겨서 다른 요청이 그동안 처리될 수 있게 한다.
    try:
        loop = asyncio.get_running_loop()
        detections, elapsed = await loop.run_in_executor(
            inference_executor, run_detection, image, threshold
        )
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        raise HTTPException(status_code=500, detail=f"추론 실패: {type(e).__name__}")

    logger.info(f"탐지 완료 — {len(detections)}개, {elapsed}ms")
    return {
        "success": True,
        "count": len(detections),
        "detections": detections,
        "threshold": threshold,
        "image_size": [image.width, image.height],
        "elapsed_ms": elapsed,
        "model_name": detector.model_name,
        "user": user,
    }
