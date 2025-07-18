from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

data_dir = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\binary_classification\완벽한 이진분류를 위한 재도전\split_data(binary_2)"
def get_loaders(data_dir=data_dir, batch_size=32, num_workers=4):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomPerspective(0.2, p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
