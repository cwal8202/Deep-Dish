import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from typing import List, Dict, Any
import label_eng_to_kor

# 이미지 크기
IMG_SIZE = (260, 260)

# 모델 정의 (Input 레이어 명시)
def build_transfer_model_with_gap_output(base_cls, num_classes, dropout_rate=0.2):
    # weights=None으로 먼저 구조만 만듭니다.
    base = base_cls(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D(name='gap')(base.output)
    x_fc = Dense(256, activation='relu')(x)
    x_drop = Dropout(dropout_rate)(x_fc)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x_drop)
    model = Model(inputs=base.input, outputs=[x, out])
    return model

# 모델 및 label map 로드
def load_models(model_dir: str, timestamp: str):
    label_map = joblib.load(os.path.join(model_dir, f"label_to_index_{timestamp}.joblib"))
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    eff_model = build_transfer_model_with_gap_output(EfficientNetB2, num_classes)
    # fine-tuning된 가중치를 로드합니다.
    eff_model.load_weights(os.path.join(model_dir, f"effnetB2_model_finetuned.h5"))

    return eff_model, index_to_label

# --- 💡 1. 앱 시작 시 모델을 전역 변수로 미리 로드 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TIMESTAMP = '20250716_144252'

EFF_MODEL, IDX2LABEL = load_models(MODEL_DIR, TIMESTAMP)
print("✅ EfficientNetB2 모델 로딩 완료.")
# ----------------------------------------------------

# 이미지 예측 (최적화 적용)
# --- 💡 2. 함수가 더 이상 모델을 인자로 받지 않음 ---
def predict_image(path: str):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    inp = eff_pre(img)[None, ...]
    # 미리 로드된 전역 모델(EFF_MODEL)을 사용
    _, prob = EFF_MODEL(inp, training=False)

    prob = prob.numpy().flatten()
    idx = int(np.argmax(prob))
    # 미리 로드된 전역 딕셔너리(IDX2LABEL)를 사용
    label = IDX2LABEL[idx]
    confidence = float(prob[idx])
    probs = {IDX2LABEL[i]: float(prob[i]) for i in range(len(prob))}

    return label, confidence, probs

# 예측 실행 함수
def run_prediction(saved_info_list: List[tuple[int, str]]) -> List[Dict[str, Any]]:
    # --- 💡 3. 함수 내에서 모델을 로드하는 코드를 삭제 ---
    results = []
    for image_id, img_path in saved_info_list:
        full_path = os.path.join(BASE_DIR, '..', img_path)
        if not os.path.isfile(full_path):
            print(f"❌ 파일 없음: {full_path}")
            continue

        # 수정된 predict_image 함수 호출 (인자 변경)
        label, conf, probs = predict_image(full_path)
        kor_label = label_eng_to_kor.ENG_TO_KOR_MENU_MAP.get(label, label)

        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        top5_probs_eng = dict(sorted_probs[:5])

        top_5_probs_kor = {
            label_eng_to_kor.ENG_TO_KOR_MENU_MAP.get(k, k): round(v, 2)
            for k, v in top5_probs_eng.items()
        }

        results.append({
            'image_id': image_id,
            'file_path': os.path.basename(full_path),
            'predicted_label': kor_label,
            'confidence': round(conf, 2),
            'probabilities': top_5_probs_kor
        })

    return results