import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def build_model(num_classes=9, pretrained=True):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),      # ✅ dropout 강화
        nn.Linear(in_features, 256),          # ✅ FC 추가
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)           # ✅ 최종 출력
    )

    return model
