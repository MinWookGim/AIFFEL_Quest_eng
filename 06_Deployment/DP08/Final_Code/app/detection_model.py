"""
Day 8 - 객체 탐지 모델 (DETR)

facebook/detr-resnet-50 을 불러 이미지 하나에서 물체를 찾는다.
학습은 하지 않고 이미 학습된 것을 가져다 쓴다.
"""
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class DetectionModel:
    """DETR 을 한 번 올려두고 요청마다 재사용한다."""

    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        # GPU 가 있으면 GPU, 없으면 CPU. ROCm 도 cuda 로 잡힌다.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 전처리기와 모델을 각각 불러온다. 전처리기가 크기 조정과 정규화를 맡는다.
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()   # 추론만 할 것이라 평가 모드로 고정한다.

        self.model_name = model_name
        # 숫자 라벨을 사람이 읽는 이름으로 바꿀 표. COCO 91종이 들어 있다.
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, image: Image.Image, threshold: float = 0.5) -> tuple[list[dict], float]:
        """이미지 하나에서 물체를 찾아 목록과 걸린 시간(밀리초)을 돌려준다."""
        t0 = time.time()

        # 이미지를 모델이 먹는 텐서로 바꾼다.
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # 모델 출력은 0~1 로 정규화된 좌표라, 원본 픽셀 좌표로 되돌린다.
        # target_sizes 는 (세로, 가로) 순서라 image.size(가로, 세로)를 뒤집어 넣는다.
        target = torch.tensor([image.size[::-1]]).to(self.device)
        result = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target
        )[0]

        detections = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            x_min, y_min, x_max, y_max = [round(v, 1) for v in box.tolist()]
            detections.append({
                "label": self.id2label[label.item()],
                "score": round(score.item(), 4),
                "box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            })

        # 확신도가 높은 것부터 보이게 정렬한다.
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections, round((time.time() - t0) * 1000, 1)
