import numpy as np
import torch
import torch.nn as nn
from PIL import ImageOps
from torchvision import transforms

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# DP04 에서 추가한 단계.
# 모델은 MNIST 로만 배웠는데 MNIST 는 전부 검은 배경(0)에 흰 획(255)이다.
# 흰 종이에 검은 펜으로 쓴 사진을 그대로 넣으면 모델이 배경을 획으로 읽어서 엉뚱하게 답한다.
# 그래서 배경이 밝은 그림만 골라 뒤집어, 모델이 배운 생김새로 맞춰준다.
#
# 기준을 128 로 잡은 근거: MNIST 테스트셋 1만 장의 이미지별 평균 밝기를 재보니
# 최소 7.2 / 중앙 33.0 / 최대 83.4 였다. 128 을 넘는 장이 한 장도 없어서,
# 이 기준이 원래 잘 되던 그림을 잘못 뒤집을 걱정은 안 해도 될 것 같았다.
반전기준 = 128.0


def 배경이_밝으면_뒤집기(img):
    """평균 밝기가 기준을 넘으면(= 배경이 흰 쪽이면) 밝기를 뒤집는다."""
    if np.array(img, dtype=np.float32).mean() > 반전기준:
        return ImageOps.invert(img)
    return img


preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(배경이_밝으면_뒤집기),   # <- DP04 에서 넣은 줄
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

CLASS_NAMES = [str(i) for i in range(10)]

def load_model(model_path, num_classes=10):
    model = SimpleClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
        idx = probs.argmax().item()
        return {
            "predicted_class": CLASS_NAMES[idx],
            "confidence": round(probs[idx].item(), 4),
            "probabilities": {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))},
        }
