import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_binary_dataloaders(data_dir=None, batch_size=32, num_workers=2):
    """
    Binary Classification용 DataLoader 생성
    :param data_dir: 데이터 디렉토리 경로
    :param batch_size: 배치 크기
    :param num_workers: 워커 수
    :return: train_loader, val_loader, test_loader
    """
    # 기본 경로 설정
    if data_dir is None:
        # 실제 Binary Classification 경로
        data_dir = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\binary_classification\food_binary"
    
    print(f"데이터 경로: {data_dir}")
    print(f"경로 존재 확인: {os.path.exists(data_dir)}")
    
    # 경로가 존재하지 않으면 에러 메시지 출력
    if not os.path.exists(data_dir):
        print("❌ 데이터 경로가 존재하지 않습니다!")
        print("📁 확인할 경로들:")
        
        # 단계별로 경로 확인
        base_paths = [
            r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model",
            r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100",
            r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\binary_classification",
            r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\binary_classification\food_binary"
        ]
        
        for path in base_paths:
            exists = os.path.exists(path)
            print(f"   {path} → {'✅' if exists else '❌'}")
            if exists and os.path.isdir(path):
                try:
                    contents = os.listdir(path)
                    print(f"      내용: {contents[:5]}{'...' if len(contents) > 5 else ''}")
                except:
                    print("      내용 확인 불가")
        
        return None, None, None
    
    # 전처리 변환
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])
    
    # 하위 폴더 확인
    subfolders = ['train', 'val', 'test']
    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        print(f"{folder} 폴더: {os.path.exists(folder_path)}")
        if os.path.exists(folder_path):
            try:
                contents = os.listdir(folder_path)
                print(f"  {folder} 내용: {contents}")
                
                # 각 클래스 폴더의 이미지 개수 확인
                for class_folder in contents:
                    class_path = os.path.join(folder_path, class_folder)
                    if os.path.isdir(class_path):
                        # food 폴더 안의 세부 폴더들도 확인
                        if class_folder == 'food':
                            print(f"    {class_folder} 세부 내용:")
                            try:
                                food_contents = os.listdir(class_path)
                                total_food_images = 0
                                for food_type in food_contents:
                                    food_type_path = os.path.join(class_path, food_type)
                                    if os.path.isdir(food_type_path):
                                        # burger, dessert, sandwich 폴더 확인
                                        print(f"      {food_type}:")
                                        brand_folders = os.listdir(food_type_path)
                                        food_type_total = 0
                                        for brand in brand_folders:
                                            brand_path = os.path.join(food_type_path, brand)
                                            if os.path.isdir(brand_path):
                                                brand_count = len([f for f in os.listdir(brand_path) 
                                                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                                                print(f"        {brand}: {brand_count}개")
                                                food_type_total += brand_count
                                        print(f"      {food_type} 소계: {food_type_total}개")
                                        total_food_images += food_type_total
                                print(f"    {class_folder} 총계: {total_food_images}개")
                        else:
                            # Not_food 폴더
                            image_count = len([f for f in os.listdir(class_path) 
                                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                            print(f"    {class_folder}: {image_count}개 이미지")
            except:
                print(f"  {folder} 내용 확인 불가")
    
    try:
        # 데이터셋 로딩
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
        
        # 👇 train_dataset 이후에 추가
        targets = train_dataset.targets  # 라벨 목록
        class_counts = np.bincount(targets)  # 각 클래스별 개수
        class_weights = 1. / class_counts  # 클래스별 가중치
        sample_weights = class_weights[targets]  # 각 샘플별 weight
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        
        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        print("✅ 데이터 로딩 성공!")
        print("Loaded classes:", train_dataset.classes)
        print(f"Train 데이터: {len(train_dataset)}개")
        print(f"Val 데이터: {len(val_dataset)}개")
        print(f"Test 데이터: {len(test_dataset)}개")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None, None, None

# 테스트 실행
if __name__ == "__main__":
    print("🔍 Binary Classification DataLoader 테스트 시작...")
    
    # DataLoader 생성 테스트
    train_loader, val_loader, test_loader = get_binary_dataloaders()
    
    if train_loader is not None:
        print("🎉 Binary DataLoader 생성 성공!")
        
        # 첫 번째 배치 테스트
        try:
            for images, labels in train_loader:
                print(f"✅ 배치 테스트 성공!")
                print(f"   이미지 크기: {images.shape}")
                print(f"   라벨 크기: {labels.shape}")
                print(f"   라벨 내용 (처음 10개): {labels[:10]}")
                
                # 라벨 분포 확인
                unique_labels, counts = torch.unique(labels, return_counts=True)
                print(f"   배치 내 라벨 분포:")
                for label, count in zip(unique_labels, counts):
                    class_name = train_loader.dataset.classes[label]
                    print(f"     {class_name}({label}): {count}개")
                break
                
        except Exception as e:
            print(f"❌ 배치 테스트 실패: {e}")
    else:
        print("❌ Binary DataLoader 생성 실패!")