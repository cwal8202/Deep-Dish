import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_brand_efficientnet import build_efficientnetb3_model
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ✅ 설정
DATA_DIR = r"C:\\Users\\baby3\\OneDrive\\바탕 화면\\0716_ 모델링\\project_model_EfficientNet-B3\\0720_split_dataset"
MODEL_PATH = "0720_best_efficientnetb3_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ 데이터 로딩
print("📦 Loading test data...")
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes
print(f"클래스 수: {len(class_names)} | 클래스 목록: {class_names}")

# ✅ 원본 파일 경로 추적용
orig_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"))
image_paths = [path for path, _ in orig_dataset.imgs]

# ✅ 모델 로드
model = build_efficientnetb3_model(num_classes=len(class_names), pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ 예측 수행
all_labels = []
all_preds = []
all_probs = []
pred_logs = []

image_counter = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        for i in range(images.size(0)):
            idx = image_counter + i
            true_idx = labels[i].item()
            pred_idx = preds[i].item()
            top3_probs, top3_idx = torch.topk(probs[i], 3)
            file_path = image_paths[idx]
            file_name = os.path.basename(file_path)
            log = {
                'file': file_name,
                'true': class_names[true_idx],
                'pred_top1': class_names[top3_idx[0]],
                'top1_conf': float(top3_probs[0]),
                'pred_top2': class_names[top3_idx[1]],
                'top2_conf': float(top3_probs[1]),
                'pred_top3': class_names[top3_idx[2]],
                'top3_conf': float(top3_probs[2]),
                'correct': class_names[top3_idx[0]] == class_names[true_idx]
            }
            pred_logs.append(log)
        image_counter += images.size(0)

# ✅ 로그 저장
log_df = pd.DataFrame(pred_logs)
log_df.to_csv("0720_prediction_log_effb3.csv", index=False)
print("📁 Saved detailed prediction log: prediction_log_effb3.csv")

# ✅ 잘못된 예측 분석
incorrect_df = log_df[log_df['correct'] == False]
print("\n🔍 Top 10 잘못 예측된 이미지:")
print(incorrect_df[['file', 'true', 'pred_top1', 'top1_conf']].head(10))

print("\n❗ 클래스별 오류 건수:")
print(incorrect_df['true'].value_counts())

# ✅ 평가 지표 계산
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\n📊 Classification Report:\n")
print(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix (EfficientNet-B3)")
plt.tight_layout()
plt.savefig("0720_confusion_matrix_effb3.png", dpi=300)
plt.show()

confidences = np.max(np.array(all_probs), axis=1)
plt.hist(confidences, bins=20, color='lightgreen', edgecolor='black')
plt.title('Prediction Confidence Histogram (EffB3)')
plt.xlabel('Confidence')
plt.ylabel('Number of Samples')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("0720_confidence_histogram_effb3.png", dpi=300)
plt.show()

print("\n========================================")
print("📊 Final Evaluation Summary (EffB3)")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("========================================")

summary_df = pd.DataFrame({
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'f1_score': [f1]
})
summary_df.to_csv("0720_eval_summary_metrics_effb3.csv", index=False)
print("📁 Saved evaluation summary: eval_summary_metrics_effb3.csv")
