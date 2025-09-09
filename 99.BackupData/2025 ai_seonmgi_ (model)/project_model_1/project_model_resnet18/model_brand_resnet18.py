import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes):
    # 최신 방식: ImageNet 사전학습 가중치 불러오기
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # feature extractor freeze
    for param in model.parameters():
        param.requires_grad = False

    # classifier head 교체
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model
