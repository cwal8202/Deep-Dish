# train_brand.py (수정됨: EarlyStopping 개선 + 저장 경로 반영 + 최종 성능 시각화 표시)
import os
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd

from model_brand_resnet18 import build_model
from label_smoothing_loss import LabelSmoothingLoss

# ===== 설정 =====
data_dir = os.path.join(r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_EfficientNet-B3\0720_split_dataset")
save_path = "0720_2_best_resnet18_brand.pt"
batch_size = 64
epochs = 20
learning_rate = 1e-4
num_workers = 2
early_stop_patience = 6
use_label_smoothing = True

# ===== 데이터 준비 =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class_names = train_dataset.classes

# ===== 모델 및 손실함수 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=len(class_names)).to(device)

criterion = LabelSmoothingLoss(classes=len(class_names)) if use_label_smoothing else nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ===== 학습 함수 =====
def train_model():
    best_acc = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    early_stop_counter = 0

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc.item())

        # ===== Validation =====
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        val_loss_list.append(val_epoch_loss)
        val_acc_list.append(val_epoch_acc.item())

        scheduler.step(val_epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # ===== EarlyStopping 수정 조건 =====
        if val_epoch_acc > best_acc + 1e-4:
            best_acc = val_epoch_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
            print("✅ Best model saved.")
        else:
            early_stop_counter += 1
            print(f"⏳ EarlyStopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

        # ===== 실시간 그래프 표시 =====
        ax1.clear()
        ax1.plot(train_loss_list, label='Train Loss')
        ax1.plot(val_loss_list, label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()

        ax2.clear()
        ax2.plot(train_acc_list, label='Train Acc')
        ax2.plot(val_acc_list, label='Val Acc')
        ax2.set_title('Accuracy')
        ax2.legend()

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    ax1.set_title(f'Loss (Best Val Acc: {best_acc:.4f})')
    ax2.set_title(f'Accuracy (Best Val Acc: {best_acc:.4f})')
    plt.savefig("0720_2_resnet18_training_plot.png")

    # ===== 로그 저장 =====
    df = pd.DataFrame({
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
    })
    df.to_csv("0720_2_resnet18_training_log.csv", index=False)

    # ===== 최종 모델 적용 =====
    model.load_state_dict(best_model_wts)
    print(f"🎉 Training completed. Best Epoch: {best_epoch} | Best Val Acc: {best_acc:.4f}")

    # ✅ 추가: 최종 수치 출력
    print("="*40)
    print("📊 Final Summary")
    print(f"Best Epoch       : {best_epoch}")
    print(f"Best Val Acc     : {best_acc:.4f}")
    print(f"Final Train Loss : {train_loss_list[-1]:.4f}")
    print(f"Final Val Loss   : {val_loss_list[-1]:.4f}")
    print(f"Final Train Acc  : {train_acc_list[-1]:.4f}")
    print(f"Final Val Acc    : {val_acc_list[-1]:.4f}")
    print("="*40)

if __name__ == '__main__':
    train_model()
