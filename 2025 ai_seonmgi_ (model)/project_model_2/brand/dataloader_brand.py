import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

def get_multi_dataloaders(data_dir=None, batch_size=32, num_workers=2):
    """
    Multi Classification용 DataLoader 생성 (음식 타입 분류)
    :param data_dir: 데이터 디렉토리 경로
    :param batch_size: 배치 크기
    :param num_workers: 워커 수
    :return: train_loader, val_loader, test_loader
    """
    # 기본 경로 설정
    if data_dir is None:
        data_dir = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\multi_classification\food_type_brand"
    
    # 전처리 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 데이터셋 로딩
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Loaded classes:", train_dataset.classes)
    print("Number of classes:", len(train_dataset.classes))
    
    return train_loader, val_loader, test_loader

def get_brand_dataloaders(data_dir=None, batch_size=32, num_workers=2):
    """
    Brand Classification용 DataLoader 생성 (브랜드 분류)
    Multi Classification과 동일한 경로 사용
    """
    return get_multi_dataloaders(data_dir, batch_size, num_workers)

if __name__ == "__main__":
    print("🔍 Multi Classification DataLoader 테스트 시작...")
    
    # DataLoader 생성 테스트
    train_loader, val_loader, test_loader = get_multi_dataloaders()
    
    if train_loader is not None:
        print("🎉 Multi DataLoader 생성 성공!")
        print(f"클래스: {train_loader.dataset.classes}")
        print(f"클래스 개수: {len(train_loader.dataset.classes)}")
        
        # 첫 번째 배치 테스트
        try:
            for images, labels in train_loader:
                print(f"✅ 배치 테스트 성공!")
                print(f"   이미지 크기: {images.shape}")
                print(f"   라벨 크기: {labels.shape}")
                print(f"   라벨 내용: {labels[:5]}")
                break
        except Exception as e:
            print(f"❌ 배치 테스트 실패: {e}")
    else:
        print("❌ DataLoader 생성 실패!")