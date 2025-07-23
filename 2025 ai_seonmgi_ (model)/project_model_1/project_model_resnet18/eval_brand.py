# eval_brand.py (Grad-CAM, 잘못된 예측 저장, 시각화 포함)
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
from model_brand_resnet18 import build_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# ===== 설정 =====
data_dir = os.path.join(r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_EfficientNet-B3\0720_split_dataset")
save_root = "2"
model_path = r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model_resnet18\3차\0720_2_best_resnet18_brand.pt"
os.makedirs(f"0720_{save_root}", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== 데이터 로딩 =====
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# ===== 모델 로딩 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== 예측 및 결과 저장 =====
y_true, y_pred, y_probs = [], [], []
all_fnames = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

        batch_paths = [test_dataset.samples[i][0] for i in range(len(all_fnames), len(all_fnames)+len(labels))]
        all_fnames.extend(batch_paths)

# ===== DataFrame 저장 =====
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

result_df = pd.DataFrame({
    'image_path': all_fnames,
    'true_label': [class_names[i] for i in y_true],
    'pred_label': [class_names[i] for i in y_pred],
    'confidence': np.max(y_probs, axis=1)
})
result_df.to_csv(f"0720_{save_root}/test_predictions.csv", index=False)

# ===== 잘못된 예측 이미지 저장 =====
misclassified = result_df[result_df['true_label'] != result_df['pred_label']]
for _, row in misclassified.iterrows():
    true_cls = row['true_label']
    pred_cls = row['pred_label']
    img_path = row['image_path']
    dest_dir = f"0720_{save_root}/misclassified/{true_cls}_as_{pred_cls}"
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(img_path)
    os.system(f'copy "{img_path}" "{os.path.join(dest_dir, fname)}"')

# ===== confusion matrix =====
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig(f"0720_{save_root}/confusion_matrix.png")
plt.close()

# ===== confidence histogram =====
plt.figure(figsize=(8, 6))
sns.histplot(result_df['confidence'], bins=20, kde=True)
plt.title("Prediction Confidence Histogram")
plt.savefig(f"0720_{save_root}/confidence_histogram.png")
plt.close()

# ===== classification report 저장 =====
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f"0720_{save_root}/classification_report.csv")

# ===== Grad-CAM 시각화 (상위 예측 정확한 이미지 1개씩 저장) =====
os.makedirs(f"0720_{save_root}/gradcam_examples", exist_ok=True)

for cls in class_names:
    sample = result_df[(result_df['true_label'] == cls) & (result_df['true_label'] == result_df['pred_label'])].sort_values(by='confidence', ascending=False).head(1)
    if not sample.empty:
        img_path = sample.iloc[0]['image_path']
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        tensor.requires_grad_()  # 🔹 이 줄 추가
        
        target_layers = [model.layer4[-1]]
        with GradCAM(model=model, target_layers=target_layers, reshape_transform=None) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_names.index(cls))])[0, :]
            

        img_np = np.array(img.resize((224, 224))) / 255.0
        cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        plt.imsave(f"0720_{save_root}/gradcam_examples/{cls}.png", cam_img)
