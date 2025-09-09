# train_efficientnet_b3_gradcam.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score)
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import cv2

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_brand_efficientnet import build_efficientnetb3_model

# ===== 사용자 경로 설정 =====
TRAIN_DIR = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_EfficientNet-B3\0720_split_dataset\train"
VAL_DIR = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_EfficientNet-B3\0720_split_dataset\val"
TEST_DIR = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_EfficientNet-B3\0720_split_dataset\test"
MODEL_SAVE_PATH = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_EfficientNet-B3\0721_모델링\0726-1_best_efficientnetb3_model.pth"

# ===== 하이퍼파라미터 =====
batch_size = 32
num_epochs = 25
learning_rate = 1e-4
early_stop_patience = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Grad-CAM 함수 =====
def generate_gradcam(model, image_tensor, class_idx, target_layer):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return_nodes = {target_layer: 'feat'}
    extractor = create_feature_extractor(model, return_nodes=return_nodes)

    gradients = []
    def hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    handle = extractor[target_layer].register_full_backward_hook(hook_fn)

    features = extractor(image_tensor)
    output = model(image_tensor)
    pred = output[:, class_idx]
    model.zero_grad()
    pred.backward()

    grads_val = gradients[0]
    fmap = features['feat']
    weights = grads_val.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    cam /= cam.max()
    return cam

# ===== 데이터 로더 =====
def get_loader(path):
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader, dataset.classes

# ===== 학습 함수 =====
def train():
    train_loader, _ = get_loader(TRAIN_DIR)
    val_loader, class_names = get_loader(VAL_DIR)
    model = build_efficientnetb3_model(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        correct, total, total_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_loss = total_loss / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n📢 Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print("✅ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break

    elapsed = time.time() - start_time
    print(f"⏱️ Total Training Time: {elapsed:.2f} seconds")

    pd.DataFrame(history).to_csv("0726_training_log.csv", index=False)

    # ===== 최종 평가 + 분석 =====
    test_loader, _ = get_loader(TEST_DIR)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    wrong_imgs, wrong_lbls, wrong_preds = [], [], []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="🧪 Test Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().tolist())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    wrong_imgs.append(images[i].cpu())
                    wrong_lbls.append(labels[i].item())
                    wrong_preds.append(preds[i].item())

    cls_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n📊 Classification Report:\n", cls_report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cls_report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("0726_confusion_matrix.png")
    plt.close()

    cls_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10,6))
    sns.barplot(x=class_names, y=cls_acc)
    plt.title("Per-Class Accuracy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("0726_per_class_accuracy.png")
    plt.close()

    all_conf = [max(p) for p in y_probs]
    plt.figure(figsize=(6,4))
    sns.histplot(all_conf, bins=20, kde=True)
    plt.title("Prediction Confidence Distribution")
    plt.tight_layout()
    plt.savefig("0726_confidence_histogram.png")
    plt.close()

    wrong_matrix = confusion_matrix(wrong_lbls, wrong_preds, labels=range(len(class_names)))
    plt.figure(figsize=(8,6))
    sns.heatmap(wrong_matrix, annot=True, fmt="d", cmap="Reds",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Heatmap of Misclassifications")
    plt.tight_layout()
    plt.savefig("0726_wrong_heatmap.png")
    plt.close()

    os.makedirs("wrong_images", exist_ok=True)
    os.makedirs("wrong_gradcam", exist_ok=True)
    if wrong_imgs:
        conf_scores = [max(y_probs[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        sorted_idx = np.argsort(conf_scores)[-9:]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, idx in enumerate(sorted_idx):
            img = wrong_imgs[idx]
            true_cls = class_names[wrong_lbls[idx]]
            pred_cls = class_names[wrong_preds[idx]]
            np_img = img.permute(1, 2, 0).numpy()
            np_img = np.clip(np_img * [0.229,0.224,0.225] + [0.485,0.456,0.406], 0, 1)
            ax = axes[i // 3, i % 3]
            ax.imshow(np_img)
            ax.axis("off")
            ax.set_title(f"T: {true_cls}\nP: {pred_cls}")
            bgr_img = (np_img * 255).astype(np.uint8)[..., ::-1]
            filename = f"{i+1:02d}_{true_cls}_pred_{pred_cls}.jpg"
            cv2.imwrite(f"wrong_images/{filename}", bgr_img)
            cam = generate_gradcam(model, img, wrong_preds[idx], target_layer="features.6.2.block.3")
            cam = cv2.resize(cam, (np_img.shape[1], np_img.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(bgr_img, 0.6, heatmap, 0.4, 0)
            cv2.imwrite(f"wrong_gradcam/gradcam_{filename}", overlay)
        plt.tight_layout()
        plt.savefig("0726_top9_misclassified.png")
        plt.close()

if __name__ == "__main__":
    train()
