import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res_pre
from typing import List, Dict, Any
import label_eng_to_kor

# 이미지 크기
IMG_SIZE = (224, 224)

# 💡 1. 모델 정의 변경: 더 이상 GAP 출력이 필요 없으므로 단순화
def build_transfer_model(base_cls, num_classes, dropout_rate=0.2):
    """
    XGBoost를 사용하지 않으므로 GAP 특징 벡터를 반환할 필요가 없습니다.
    Softmax 확률만 출력하도록 모델 구조를 수정합니다.
    """
    base = base_cls(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D(name='gap')(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    # 최종 Softmax 출력만 반환
    out = Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = Model(inputs=base.input, outputs=out) # outputs가 리스트가 아님
    return model

# 💡 2. 모델 로더 변경: XGBoost 관련 코드 삭제
def load_models(model_dir: str, timestamp: str):
    """XGBoost 모델 로딩 부분을 삭제합니다."""
    label_map = joblib.load(os.path.join(model_dir, f"label_to_index_{timestamp}.joblib"))
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    eff_model = build_transfer_model(EfficientNetB0, num_classes)
    eff_model.load_weights(os.path.join(model_dir, f"effnet_model_best_{timestamp}.h5"))

    res_model = build_transfer_model(ResNet50V2, num_classes)
    res_model.load_weights(os.path.join(model_dir, f"resnet_model_best_{timestamp}.h5"))

    # xgb_model 로딩 삭제
    return eff_model, res_model, index_to_label # 반환 값에서 xgb_model 제외

# --- 앱 시작 시 모델을 전역 변수로 미리 로드 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TIMESTAMP = '20250716_144252'

# 💡 3. 전역 변수 변경: XGB_MODEL 제외
EFF_MODEL, RES_MODEL, IDX2LABEL = load_models(MODEL_DIR, TIMESTAMP)
print("✅ CNN 모델 (EffNet, ResNet) 로딩 완료.")
# ----------------------------------------------------

# 이미지 예측 (최적화 적용)
def predict_image(path: str):
    """
    XGBoost 관련 로직을 모두 제거하고,
    EfficientNet과 ResNet의 예측 확률을 단순 평균합니다.
    """
    raw = tf.io.read_file(path)
    try:
        img = tf.image.decode_jpeg(raw, channels=3)
    except:
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    inp_eff = eff_pre(img)[None, ...]
    inp_res = res_pre(img)[None, ...]

    # 💡 4. 예측 로직 변경
    # 미리 로드된 전역 모델들을 사용 (출력이 softmax 확률만 나옴)
    p_eff = EFF_MODEL(inp_eff, training=False)
    p_res = RES_MODEL(inp_res, training=False)

    # 두 CNN 모델의 예측 확률을 단순 평균하여 앙상블
    ensemble_probs = (p_eff + p_res) / 2.0
    ensemble_probs = ensemble_probs.numpy().flatten() # numpy 배열로 변환

    idx = int(np.argmax(ensemble_probs))
    label = IDX2LABEL[idx]
    confidence = float(ensemble_probs[idx])
    probs = {IDX2LABEL[i]: float(ensemble_probs[i]) for i in range(len(ensemble_probs))}

    return label, confidence, probs

# 예측 실행 함수 (이 부분은 변경할 필요 없음)
def run_prediction(saved_info_list: List[tuple[int, str]]) -> List[Dict[str, Any]]:
    results = []
    for image_id, img_path in saved_info_list:
        full_path = os.path.join(BASE_DIR, '..', img_path)
        if not os.path.isfile(full_path):
            print(f"❌ 파일 없음: {full_path}")
            continue

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