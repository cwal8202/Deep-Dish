
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_binary_resnet import get_resnet18_binary_model
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# 경로 설정
data_dir = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\binary_classification\food_binary"
model_path = "best_binary_model.pth"

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 테스트용 transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],  # ImageNet 평균
        [0.229, 0.224, 0.225]   # ImageNet 표준편차
    )])
# RGB 각 채널의 ImageNet 평균/표준편차로 정규화  , ImageNet 기준 pretrained=True 모델 사용시 mean/std 반드시 사용해야함 
# 표준편차 (std) red:0.229 / green:0.224 / blue;0.225
# 평균(mean) red:0.485 / green:0.456 / blue;0.406

# 테스트 데이터 로딩
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)  # 수정: transform → test_transform
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 수정: DataLoder → DataLoader

# 클래스 이름 확인
class_names = test_dataset.classes  # 수정: class_naems → class_names
print("클래스 라벨:", class_names)

# 모델 불러오기
model = get_resnet18_binary_model(pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# 예측 수행
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())  # 수정: apll_preds → all_preds
        all_labels.extend(labels.cpu().numpy())

# 평가 지표 출력
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 혼동 행렬 시각화
cm = confusion_matrix(all_labels, all_preds)  # 수정: confusion_metrix → confusion_matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)  # 수정: confusion_metrix → confusion_matrix

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Food vs Not_Food)")
plt.savefig("confusion_matrix_binary.png", dpi=300, bbox_inches='tight')
plt.show()

# 정확도 계산
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\n테스트 정확도: {accuracy:.4f}")