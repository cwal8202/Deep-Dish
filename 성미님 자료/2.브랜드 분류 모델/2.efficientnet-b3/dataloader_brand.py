import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_efficientnetb3_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    EfficientNet-B3 전용 DataLoader (300x300, CenterCrop 포함)
    :param data_dir: 데이터셋 루트 디렉토리 (train/val/test 포함)
    :return: train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.Resize(320),              # 약간 크게 resize
        transforms.CenterCrop(300),         # 정확히 300x300 crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# 테스트 실행 (독립 테스트 시 사용 가능)
if __name__ == '__main__':
    data_dir = r"C:\\Users\\baby3\\OneDrive\\바탕 화면\\0716_ 모델링\\project_model_EfficientNet-B3\\0720_split_dataset"
    train_loader, val_loader, test_loader = get_efficientnetb3_dataloaders(data_dir)

    print("클래스 목록:", train_loader.dataset.classes)
    print("Train 샘플 수:", len(train_loader.dataset))
    print("Val 샘플 수:", len(val_loader.dataset))
    print("Test 샘플 수:", len(test_loader.dataset))