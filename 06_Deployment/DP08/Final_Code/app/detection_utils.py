"""
Day 8 - 업로드 이미지 검증

Day 6 의 app/image_utils.py 와 검증 순서(타입 -> 크기 -> 디코딩)는 같다.
다만 Day 6 은 마지막에 28x28 그레이스케일로 바꿔서 돌려주는데,
객체 탐지는 원본 크기 컬러 이미지가 그대로 필요해서 그 부분만 다르게 뒀다.
"""
import io

from fastapi import UploadFile, HTTPException
from PIL import Image

# 허용 형식과 최대 크기. Day 6 과 같은 값을 쓴다.
ALLOWED_TYPES = {"image/png", "image/jpeg"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


async def validate_and_read_rgb(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> Image.Image:
    """업로드 파일을 검증하고 원본 크기 RGB 이미지로 돌려준다."""

    # 1단계. 클라이언트가 붙여 보낸 형식 표시를 먼저 본다.
    #        이건 위조될 수 있어서 명백히 엉뚱한 것만 싸게 걸러내는 용도다.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {file.content_type}. 허용 형식: {sorted(ALLOWED_TYPES)}",
        )

    # 2단계. 크기를 잰다. 다 읽은 뒤에 재는 것이라 전송 자체를 막지는 못한다.
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"파일 크기가 {max_size // (1024*1024)}MB를 초과합니다. "
                   f"현재: {len(contents) / (1024*1024):.1f}MB",
        )

    # 3단계. 실제로 열리는 이미지인지 본다. 위장 파일은 여기서 걸린다.
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다. 손상되었거나 이미지가 아닙니다.")

    # 4단계. 모델이 RGB 세 채널을 기대하므로 흑백이나 투명도가 있는 것도 RGB 로 맞춘다.
    #        Day 6 과 달리 크기는 줄이지 않는다. 상자 좌표가 원본 기준이어야 하기 때문이다.
    return image.convert("RGB")
