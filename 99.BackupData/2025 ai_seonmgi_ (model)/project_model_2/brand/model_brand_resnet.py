import torch.nn as nn
from torchvision import models
import torch

def get_resnet18_multi_model(num_classes=3, pretrained=True):
    """
    ResNet18 기반 다중 분류 모델 생성 (음식 타입 분류)
    :param num_classes: 클래스 수 (burger, dessert, sandwich = 3)
    :param pretrained: ImageNet 사전학습 가중치 사용 여부
    :return: 음식 타입 분류 모델
    """
    # ResNet18 모델 불러오기
    model = models.resnet18(pretrained=pretrained)
    
    # 기존 fc layer의 입력 feature 수 확인
    num_ftrs = model.fc.in_features
    
    # fc layer 교체: num_classes개 클래스로 변경
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_resnet18_brand_model(num_classes=9, pretrained=True):
    """
    ResNet18 기반 브랜드 분류 모델 생성
    :param num_classes: 브랜드 수 (9개 브랜드)
    :param pretrained: ImageNet 사전학습 가중치 사용 여부
    :return: 브랜드 분류 모델
    """
    # ResNet18 모델 불러오기
    model = models.resnet18(pretrained=pretrained)
    
    # 기존 fc layer의 입력 feature 수 확인
    num_ftrs = model.fc.in_features
    
    # fc layer 교체: num_classes개 브랜드로 변경
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# 테스트 코드
if __name__ == "__main__":
    print("🔍 Multi Classification 모델 테스트 시작...")
    print("=" * 50)
    
    # 1. Multi 모델 테스트 (3개 클래스)
    print("\n📊 Multi Classification 모델 테스트:")
    try:
        multi_model = get_resnet18_multi_model(num_classes=3, pretrained=True)
        print("✅ Multi 모델 생성 성공!")
        print(f"   모델 타입: {type(multi_model)}")
        print(f"   기존 fc layer: Linear(in_features=512, out_features=1000) → 교체됨")
        print(f"   새로운 fc layer: {multi_model.fc}")
        print(f"   출력 클래스 수: {multi_model.fc.out_features}개")
        
        # 더미 입력으로 테스트
        dummy_input = torch.randn(1, 3, 224, 224)  # 배치1, RGB3, 224x224
        
        with torch.no_grad():
            multi_output = multi_model(dummy_input)
        
        print(f"   ✅ Forward Pass 성공!")
        print(f"   입력 크기: {dummy_input.shape}")
        print(f"   출력 크기: {multi_output.shape}")  # [1, 3] 이어야 함
        print(f"   출력 값: {multi_output}")
        
        # Softmax 적용해서 확률 확인
        probabilities = torch.softmax(multi_output, dim=1)
        print(f"   확률 분포: {probabilities}")
        print(f"   확률 합계: {probabilities.sum().item():.4f}")
        
    except Exception as e:
        print(f"❌ Multi 모델 테스트 실패: {e}")
    
    # 2. Brand 모델 테스트 (9개 클래스)
    print("\n🏢 Brand Classification 모델 테스트:")
    try:
        brand_model = get_resnet18_brand_model(num_classes=9, pretrained=True)
        print("✅ Brand 모델 생성 성공!")
        print(f"   모델 타입: {type(brand_model)}")
        print(f"   기존 fc layer: Linear(in_features=512, out_features=1000) → 교체됨")
        print(f"   새로운 fc layer: {brand_model.fc}")
        print(f"   출력 클래스 수: {brand_model.fc.out_features}개")
        
        # 더미 입력으로 테스트
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            brand_output = brand_model(dummy_input)
        
        print(f"   ✅ Forward Pass 성공!")
        print(f"   입력 크기: {dummy_input.shape}")
        print(f"   출력 크기: {brand_output.shape}")  # [1, 9] 이어야 함
        print(f"   출력 값: {brand_output}")
        
        # Top-3 예측 확인
        probabilities = torch.softmax(brand_output, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        print(f"   Top-3 브랜드 예측:")
        for i in range(3):
            print(f"     {i+1}위: 클래스 {top3_indices[0][i].item()} ({top3_probs[0][i].item()*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Brand 모델 테스트 실패: {e}")
    
    # 3. 배치 테스트
    print("\n📦 배치 처리 테스트:")
    try:
        # 배치 입력 (32개 이미지)
        batch_input = torch.randn(32, 3, 224, 224)
        
        with torch.no_grad():
            multi_batch_output = multi_model(batch_input)
            brand_batch_output = brand_model(batch_input)
        
        print("✅ 배치 처리 성공!")
        print(f"   배치 입력 크기: {batch_input.shape}")
        print(f"   Multi 배치 출력: {multi_batch_output.shape}")  # [32, 3]
        print(f"   Brand 배치 출력: {brand_batch_output.shape}")  # [32, 9]
        
        # 배치 예측 결과
        multi_predictions = multi_batch_output.argmax(dim=1)
        brand_predictions = brand_batch_output.argmax(dim=1)
        
        print(f"   Multi 예측 (처음 10개): {multi_predictions[:10]}")
        print(f"   Brand 예측 (처음 10개): {brand_predictions[:10]}")
        
    except Exception as e:
        print(f"❌ 배치 테스트 실패: {e}")
    
    # 4. 모델 파라미터 정보
    print("\n📈 모델 파라미터 정보:")
    try:
        multi_params = sum(p.numel() for p in multi_model.parameters())
        brand_params = sum(p.numel() for p in brand_model.parameters())
        
        multi_trainable = sum(p.numel() for p in multi_model.parameters() if p.requires_grad)
        brand_trainable = sum(p.numel() for p in brand_model.parameters() if p.requires_grad)
        
        print(f"   Multi 모델:")
        print(f"     전체 파라미터: {multi_params:,}개")
        print(f"     학습 가능한 파라미터: {multi_trainable:,}개")
        
        print(f"   Brand 모델:")
        print(f"     전체 파라미터: {brand_params:,}개")
        print(f"     학습 가능한 파라미터: {brand_trainable:,}개")
        
    except Exception as e:
        print(f"❌ 파라미터 정보 확인 실패: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Multi Classification 모델 테스트 완료!")
    print("✅ 이제 train_brand.py를 실행할 수 있습니다!")
    print("=" * 50)