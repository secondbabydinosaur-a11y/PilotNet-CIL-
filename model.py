"""PilotNet: YUV 이미지 한 장 -> 곡률(1/m) 하나.

논문 그림 4의 Conv/FC 크기를 따른다. ReLU와 정규화 수식은 이번 구현의
선택이다. 마지막 64x1x18 feature를 펼친 실제 길이는 1152다.
"""
import torch
from torch import nn


class PixelNormalization(nn.Module):
    def forward(self, x):
        # Dataset이 반환한 float32 0~255 값을 여기서 한 번만 정규화한다.
        return x / 127.5 - 1.0


class PilotNet(nn.Module):
    architecture = "pilotnet_yuv_66x200_relu_v1"

    def __init__(self):
        super().__init__()
        self.normalize = PixelNormalization()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),    # 24 x 31 x 98
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),   # 36 x 14 x 47
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),   # 48 x 5 x 22
            nn.Conv2d(48, 64, 3), nn.ReLU(),            # 64 x 3 x 20
            nn.Conv2d(64, 64, 3), nn.ReLU(),            # 64 x 1 x 18
        )
        self.regressor = nn.Sequential(
            nn.Linear(1152, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1),  # 부호 있는 곡률 회귀이므로 출력 활성화는 없다.
        )
        # 초기 예측을 직진 근처에 둔다. 나머지는 PyTorch 기본 초기화다.
        nn.init.normal_(self.regressor[-1].weight, std=1e-3)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 66, 200):
            raise ValueError("expected image batch [B, 3, 66, 200]")
        x = self.features(self.normalize(x))
        return self.regressor(torch.flatten(x, 1))
