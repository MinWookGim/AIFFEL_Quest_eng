"""
Day 5 - 주택 가격 예측 FastAPI 서버
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException

from app.housing_schemas import HousingRequest, HousingResponse
from app.housing_model import HousingPredictor
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


# ===== 설정 =====
logger = setup_logger("housing_api")

app = FastAPI(
    title="California Housing Price API",
    description="캘리포니아 주택 가격을 예측하는 API",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

# 추론 전용 스레드풀 (Day 3에서 배운 패턴)
inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="housing")  # *your code* — 스레드풀 생성

# ===== 모델 로드 =====
MODEL_PATH = "models/housing_model.pth"
PREPROCESS_PATH = "models/housing_preprocessing.json"
predictor = None


@app.on_event("startup")
async def startup():
    global predictor
    logger.info("주택 가격 모델 로드 중...")
    predictor = HousingPredictor(MODEL_PATH, PREPROCESS_PATH)  # *your code* — HousingPredictor 인스턴스 생성
    logger.info("모델 로드 완료")


# ===== 엔드포인트 =====

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if predictor is not None else "loading",
        "model": "California Housing",
    }


@app.post("/predict", response_model=HousingResponse, tags=["Prediction"])
async def predict_housing(request: HousingRequest):
    """주택 정보를 받아 가격을 예측합니다."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    # 요청 데이터를 딕셔너리로 변환
    features = request.model_dump()  # *your code* — Pydantic 모델 → dict

    try:
        # 추론 (별도 스레드에서 실행 — Day 3 패턴)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(       # *your code* — run_in_executor 사용
            inference_executor,
            predictor.predict,
            features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    return HousingResponse(
        success=True,
        predicted_price=result["predicted_price"],
        predicted_price_usd=result["predicted_price_usd"],
        input_features=features,
    )
