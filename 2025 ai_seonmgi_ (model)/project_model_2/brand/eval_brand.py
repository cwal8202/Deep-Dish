import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_brand_resnet import get_resnet18_brand_model
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil

# 경로 설정
data_dir = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\multi_classification" # ← 사용자 경로에 맞게 수정
model_path = "best_brand_model.pth"

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 테스트 transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 테스트 데이터셋 로딩
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 클래스명 로드 - 여기서 정의됩니다!
class_names = test_dataset.classes
print("브랜드 클래스:", class_names)
print(f"총 {len(class_names)}개 브랜드")

# 모델 로드
model = get_resnet18_brand_model(num_classes=len(class_names), pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("모델 로드 완료!")

# 예측 수행
all_preds = []
all_probs = []
all_labels = []
wrong_preds = []
wrong_paths = []
wrong_names = []

softmax = torch.nn.Softmax(dim=1)

# 원본 이미지 경로 저장용
orig_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"))
image_paths = [path for path, _ in orig_dataset.imgs]

image_counter = 0

print("\n테스트 진행 중...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = softmax(outputs)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                idx = image_counter + i
                wrong_preds.append((images[i].cpu(), preds[i].item(), labels[i].item(), probs[i].cpu().numpy()))
                wrong_paths.append(image_paths[idx])
                wrong_names.append(f"{class_names[preds[i]]}_{class_names[labels[i]]}_{idx:03d}.jpg")
        image_counter += len(images)

print("테스트 완료!")

# classification report 출력
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 전체 혼동 행렬
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix (Brand Classification)", fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrix_brand.png", dpi=300, bbox_inches='tight')
plt.show()

# 오답 이미지 9장 시각화 + 예측 상위 3개 확률
if len(wrong_preds) > 0:
    plt.figure(figsize=(12, 12))
    num_display = min(9, len(wrong_preds))
    
    for idx in range(num_display):
        img_tensor, pred, label, prob = wrong_preds[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)

        top3_idx = prob.argsort()[::-1][:3]
        top3_info = "\n".join([f"{class_names[i]}: {prob[i]*100:.1f}%" for i in top3_idx])

        plt.subplot(3, 3, idx + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[label]}\nPred: {class_names[pred]}\n{top3_info}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("wrong_predictions.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("오답이 없습니다! 완벽한 성능!")

# 오답이 자주 발생한 클래스 추출해서 혼동 행렬 시각화
if len(wrong_preds) > 0:
    error_counts = {}
    for _, pred, label, _ in wrong_preds:
        key = (class_names[label], class_names[pred])
        error_counts[key] = error_counts.get(key, 0) + 1

    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_errors) > 0:
        top_classes = set()
        for (true_cls, pred_cls), _ in sorted_errors[:min(10, len(sorted_errors))]:
            top_classes.add(true_cls)
            top_classes.add(pred_cls)

        if len(top_classes) > 1:
            top_classes = list(top_classes)
            indices = [class_names.index(cls) for cls in top_classes]

            filtered_cm = cm[np.ix_(indices, indices)]
            filtered_labels = [class_names[i] for i in indices]

            plt.figure(figsize=(8, 6))
            ConfusionMatrixDisplay(confusion_matrix=filtered_cm, display_labels=filtered_labels).plot(cmap=plt.cm.Oranges, xticks_rotation=45)
            plt.title("Confusion Matrix (Top Confused Classes)", fontsize=14)
            plt.tight_layout()
            plt.savefig("confusion_matrix_top_confused.png", dpi=300, bbox_inches='tight')
            plt.show()

# 오답 이미지 복사 저장
if len(wrong_paths) > 0:
    wrong_dir = "wrong_images"
    os.makedirs(wrong_dir, exist_ok=True)

    for i, (wrong_path, wrong_name) in enumerate(zip(wrong_paths, wrong_names)):
        dest_path = os.path.join(wrong_dir, wrong_name)
        shutil.copy(wrong_path, dest_path)

    print(f"\n오답 이미지 {len(wrong_paths)}장을 '{wrong_dir}' 폴더에 저장 완료했습니다.")

# 테스트 정확도 출력
test_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\n=== 최종 테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%) ===")

# 가장 많이 헷갈리는 Top 5 출력
if len(wrong_preds) > 0 and 'sorted_errors' in locals() and len(sorted_errors) > 0:
    print("\n가장 많이 헷갈리는 브랜드 조합 Top 5:")
    for i in range(min(5, len(sorted_errors))):
        (true_cls, pred_cls), count = sorted_errors[i]
        print(f"{i+1}. {true_cls} → {pred_cls}: {count}번")

# 브랜드별 정확도 출력
print("\n브랜드별 정확도:")
for i, class_name in enumerate(class_names):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(all_preds)[class_mask] == i).mean()
        total_samples = class_mask.sum()
        print(f"   {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) - {total_samples}개 샘플")
    else:
        print(f"   {class_name}: 테스트 샘플 없음")

print("\n=== 분석 완료 ===")
print("생성된 파일:")
print("- confusion_matrix_brand.png: 전체 혼동 행렬")
print("- wrong_predictions.png: 오답 샘플 시각화")
print("- confusion_matrix_top_confused.png: 주요 혼동 클래스")
print("- wrong_images/: 오답 이미지 폴더")