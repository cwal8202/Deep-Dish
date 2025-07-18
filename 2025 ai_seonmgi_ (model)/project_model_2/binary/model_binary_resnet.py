import torch.nn as nn
from torchvision import models

def get_resnet18_binary_model(pretrained=True):
    """
    ResNet18 기반 이진 분류 모델 생성 함수
    :param pretrained: ImageNet 사전학습 가중치 사용할지 여부
    :return: nn.Module (food vs not_food 분류기)
    nn :  PyTorch의 신경망 모듈 / nn.ReLU()활성화 함수 
    nn.CrossEntropyLoss() : 분류 문제에서 자주 쓰는 손실 함수
    """
    # 모델 불러오기(ImageNet 사전학습 가중치 포함)
    model = models.resnet18(pretrained=pretrained)
    
    #  기존 fc layer의 입력 feature 수 확인
    # ImageNet 1000개 클래스를 예측하므로 출력층이 nn.Linear(512,1000)으로 되어있어
    # 2-class로 바꿔야 하니까 입력 크기 512를 꺼내서 씀
    num_ftrs = model.fc.in_features
    
    # fc layer  교체 : 출력 클래스 수 = 2(food, not_food)
    # linear(512,1000) ->lInear(512,2)로 바꿔서 food,not_food 분류되도록 만듬
    model.fc = nn.Linear(num_ftrs,2)
    
    return model

# ====== 진행과정 확인하기 
if __name__ == "__main__":
    print("🔍 모델 테스트 시작...")
    
    # 모델 생성 테스트
    model = get_resnet18_binary_model(pretrained=True)
    
    print("✅ 모델 생성 성공!")
    print(f"   모델 타입: {type(model)}")
    print(f"   최종 레이어: {model.fc}")
    
    # 더미 입력으로 테스트
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)  # 배치1, RGB3, 224x224
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   출력 크기: {output.shape}")  # [1, 2] 이어야 함
        print("🎉 모델 테스트 성공!")
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")