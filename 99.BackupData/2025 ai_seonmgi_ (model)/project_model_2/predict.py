import sys, os
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import torch.nn as nn
from torchvision import models

# ———— 클래스 정의 ————
binary_classes = ['food', 'not_food']
brand_classes = [
    'burgerking','eggdropp','hongruijian','knotted',
    'lottelia','mongchouchou','shakeshakeburger','subway','twosome'
]

# ———— 장치 & 전처리 ————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ———— 모델 정의 함수 ————
def get_resnet18_binary_model(pretrained: bool = False, num_classes: int = 2) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet18_brand_model(pretrained: bool = False, num_classes: int = 9) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ———— 모델 로딩 ————
def load_models():
    # Binary
    bm = get_resnet18_binary_model(pretrained=False).to(device)
    if os.path.exists("best_binary_model.pth"):
        bm.load_state_dict(torch.load("best_binary_model.pth", map_location=device))
        bm.eval(); print("✅ Binary 모델 로드")
    else:
        print("❌ best_binary_model.pth 없음"); return None, None

    # Brand
    brm = get_resnet18_brand_model(pretrained=False).to(device)
    if os.path.exists("best_brand_model.pth"):
        brm.load_state_dict(torch.load("best_brand_model.pth", map_location=device))
        brm.eval(); print("✅ Brand 모델 로드")
    else:
        print("⚠️ best_brand_model.pth 없음 — 브랜드 예측 스킵")
        brm = None

    return bm, brm

# ———— 이미지 로드 함수 ————
def load_image(path_or_url: str):
    try:
        if path_or_url.startswith("http"):
            r = requests.get(path_or_url, timeout=5)
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")
        return img
    except Exception as e:
        print(f"[Error] 이미지 로드 실패: {e}")
        return None

# ———— 예측 파이프라인 ————
def predict_pipeline(input_src, bm, brm):
    img = load_image(input_src)
    if img is None:
        return

    x = transform(img).unsqueeze(0).to(device)

    # 1단계
    with torch.no_grad():
        o1 = bm(x); p1 = F.softmax(o1,1)[0]
    idx1 = p1.argmax().item(); conf1 = p1[idx1].item()*100
    lb1 = binary_classes[idx1]
    print(f"[1] Binary: {lb1} ({conf1:.2f}%)")

    if lb1 == 'not_food':
        plt.imshow(img); plt.title(f"NOT FOOD ({conf1:.2f}%)"); plt.axis("off"); plt.show()
        return

    # 2단계
    if brm:
        with torch.no_grad():
            o2 = brm(x); p2 = F.softmax(o2,1)[0]
            top3 = p2.topk(3)
        print("Top-3 Brands:")
        for i,(p,idx) in enumerate(zip(top3.values, top3.indices),1):
            print(f"  {i}. {brand_classes[idx]}: {p*100:.2f}%")
        best_idx = top3.indices[0].item(); best_p = top3.values[0].item()*100
        plt.imshow(img)
        plt.title(f"{brand_classes[best_idx]} ({best_p:.2f}%)")
        plt.axis("off"); plt.show()
    else:
        print("브랜드 모델 없음 — 스킵")

# ———— 메인 ————
def main():
    if len(sys.argv) < 2:
        print("사용법: python predict.py [이미지경로 또는 URL 또는 폴더경로]")
        return

    # 모델 로드
    bm, brm = load_models()
    if bm is None:
        print("❌ Binary 모델 로드 실패 — 종료")
        return

    input_src = sys.argv[1]

    # 1) 인자가 폴더면, 그 안의 이미지 파일(.jpg/.png 등)만 순회
    if os.path.isdir(input_src):
        for fname in os.listdir(input_src):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png','.jfif')):
                fp = os.path.join(input_src, fname)
                print(f"\n▶ 처리 중: {fp}")
                predict_pipeline(fp, bm, brm)
    # 2) 아니면 기존처럼 한 장 처리
    else:
        predict_pipeline(input_src, bm, brm)

if __name__ == "__main__":
    main()