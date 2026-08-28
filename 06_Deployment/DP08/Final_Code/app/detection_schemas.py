"""
Day 8 - 객체 탐지 API 스키마

Day 6 은 라벨 하나와 확신도 하나만 돌려주면 됐는데,
객체 탐지는 물체가 몇 개 나올지 모르고 각각 위치까지 딸려 나온다.
그래서 응답이 한 겹 더 들어간 구조가 된다.
"""
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """물체를 감싸는 네모. 원본 이미지 픽셀 좌표계다."""
    x_min: float = Field(..., description="왼쪽 경계")
    y_min: float = Field(..., description="위쪽 경계")
    x_max: float = Field(..., description="오른쪽 경계")
    y_max: float = Field(..., description="아래쪽 경계")


class Detection(BaseModel):
    """찾아낸 물체 하나."""
    label: str = Field(..., description="물체 이름 (COCO 91종)")
    score: float = Field(..., ge=0.0, le=1.0, description="확신도")
    box: BoundingBox = Field(..., description="위치")


class DetectionResponse(BaseModel):
    """탐지 결과 전체."""
    success: bool = Field(..., description="성공 여부")
    count: int = Field(..., description="찾은 물체 개수")
    detections: list[Detection] = Field(..., description="찾은 물체 목록")
    threshold: float = Field(..., description="이번 요청에 쓴 기준값")
    image_size: list[int] = Field(..., description="원본 이미지 크기 [가로, 세로]")
    elapsed_ms: float = Field(..., description="추론에 걸린 시간 (밀리초)")
    model_name: str = Field(..., description="사용한 모델")
    user: str = Field(..., description="인증된 사용자")
