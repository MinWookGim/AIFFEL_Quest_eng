"""
Day 3 [내 실험 C] - main_final.py 에서 run_in_executor 만 뺀 대조군

미션 4를 검증하려면 비교 대상이 있어야 하는데, 지금까지 잰 건 executor 버전 하나뿐이었다.
그래서 최종 서버를 그대로 복사하고 딱 한 군데만 바꿨다.

    before:  result = await loop.run_in_executor(inference_executor, run_inference, tensor)
    after :  result = run_inference(tensor)          # async def 안에서 그냥 부른다

나머지(로거·에러핸들러·미들웨어·전처리·응답)는 main_final.py 와 글자 단위로 같다.
달라지는 조건이 하나뿐이어야 결과 차이의 원인을 그 한 줄로 돌릴 수 있다.
"""
import io
import base64

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException

from app.schemas import PixelPredictRequest, ImagePredictRequest, PredictResponse
from app.model_utils import load_model, predict, preprocess
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


logger = setup_logger("ml_api")

app = FastAPI(
    title="MNIST Prediction API (blocking 대조군)",   # 어느 서버에 붙었는지 구분하려고 제목을 바꿨다
    description="run_in_executor 를 뺀 비교용 서버",
    version="3.0.0-blocking",
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

# 스레드풀 자체를 안 만든다. 쓸 데가 없다

MODEL_PATH = "models/mnist_state_dict.pth"
model = None


@app.on_event("startup")
async def startup():
    global model
    logger.info(f"모델 로드 중: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("모델 로드 완료 (blocking 대조군)")


def run_inference(image_tensor: torch.Tensor) -> dict:
    """원본에서는 별도 스레드에서 실행되던 함수. 여기서는 이벤트 루프 위에서 그대로 돈다"""
    if model is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    return predict(model, image_tensor)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    from app.model_utils import CLASS_NAMES
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": "SimpleClassifier",
        "model_path": MODEL_PATH,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "total_parameters": total_params,
    }


@app.post("/predict/pixels", response_model=PredictResponse, tags=["Inference"])
async def predict_from_pixels(request: PixelPredictRequest):
    try:
        pixel_array = np.array(request.pixels, dtype=np.float32)
        pixel_tensor = torch.from_numpy(pixel_array)
        pixel_tensor = (pixel_tensor - 0.1307) / 0.3081
        pixel_tensor = pixel_tensor.unsqueeze(0).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"전처리 실패: {str(e)}")

    try:
        # 여기가 유일한 차이. await 로 넘기지 않고 이벤트 루프 위에서 직접 부른다
        result = run_inference(pixel_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    return PredictResponse(
        success=True,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"] if request.return_probabilities else None,
    )


@app.post("/predict/image", response_model=PredictResponse, tags=["Inference"])
async def predict_from_image(request: ImagePredictRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = preprocess(image).unsqueeze(0)
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="유효하지 않은 Base64 문자열입니다.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")

    try:
        result = run_inference(image_tensor)   # 여기도 동일
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    return PredictResponse(
        success=True,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"] if request.return_probabilities else None,
    )
