import torch.nn as nn
from torchvision import models

def get_resnet18_binary():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
