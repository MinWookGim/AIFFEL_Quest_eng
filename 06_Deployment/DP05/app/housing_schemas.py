"""
Day 5 - 주택 가격 예측 API 스키마
"""
from pydantic import BaseModel, Field


class HousingRequest(BaseModel):
    """주택 가격 예측 요청"""
    MedInc: float = Field(..., gt=0, description="중위 소득")                        # *your code* — gt=0 설정
    HouseAge: float = Field(..., ge=0, le=100, description="주택 연식 (년)")          # *your code* — ge, le 범위
    AveRooms: float = Field(..., gt=0, description="평균 방 수")
    AveBedrms: float = Field(..., gt=0, description="평균 침실 수")
    Population: float = Field(..., gt=0, description="인구")
    AveOccup: float = Field(..., gt=0, description="평균 거주 인원")
    Latitude: float = Field(..., ge=32, le=42, description="위도 (캘리포니아 범위)")   # *your code* — 캘리포니아 위도 범위
    Longitude: float = Field(..., ge=-125, le=-114, description="경도 (캘리포니아 범위)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 3.5,
                    "HouseAge": 25.0,
                    "AveRooms": 5.0,
                    "AveBedrms": 1.0,
                    "Population": 1500.0,
                    "AveOccup": 3.0,
                    "Latitude": 37.5,
                    "Longitude": -122.0,
                }
            ]
        }
    }


class HousingResponse(BaseModel):
    """주택 가격 예측 응답"""
    success: bool = Field(description="요청 처리 성공 여부")
    predicted_price: float = Field(description="예측 가격 ($100,000 단위)")
    predicted_price_usd: int = Field(description="예측 가격 (USD)")
    input_features: dict = Field(description="입력된 피처 값")
