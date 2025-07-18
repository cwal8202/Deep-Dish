import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model_binary_resnet import get_resnet18_binary
from dataloader import get_loaders


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def main():
    data_dir = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\binary_classification\완벽한 이진분류를 위한 재도전\split_data(binary_2)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 15
    FREEZE_EPOCHS = 5

    # 모델 정의
    model = get_resnet18_binary().to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    # 데이터 로더
    train_loader, val_loader = get_loaders(data_dir)

    # 손실함수, 옵티마이저, 스케줄러
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    # 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        if epoch == FREEZE_EPOCHS:
            print(f"🔓 Unfreezing all layers at epoch {epoch}")
            for p in model.parameters():
                p.requires_grad = True

        total_loss, y_true, y_pred = 0.0, [], []

        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_acc = accuracy_score(y_true, y_pred)
        train_losses.append(total_loss)
        train_accs.append(train_acc)

        print(f"📘 Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Acc = {train_acc:.4f}")

        # 검증
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(preds.cpu().numpy())

        val_acc = accuracy_score(val_true, val_pred)
        val_accs.append(val_acc)
        scheduler.step(val_acc)

        print(f"📗 Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_binary_resnet18_2.pth")
            print("✅ Best model saved!")

    # 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.close()
    print("📈 그래프 저장 완료: training_plot.png")


if __name__ == "__main__":
    main()
