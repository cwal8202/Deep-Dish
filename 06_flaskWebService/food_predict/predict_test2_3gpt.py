# predict.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import joblib
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res_pre
from typing import List, Dict, Any


IMG_SIZE = (224, 224)

def build_transfer_model(base_cls, num_classes, dropout_rate=0.2):
    base = base_cls(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D(name='gap')(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return Model(inputs=base.input, outputs=out)

def load_models(model_dir: str, timestamp: str):
    label_map = joblib.load(os.path.join(model_dir, f"label_to_index_{timestamp}.joblib"))
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    eff = build_transfer_model(EfficientNetB0, num_classes, dropout_rate=0.2)
    eff.load_weights(os.path.join(model_dir, f"effnet_model_best_{timestamp}.h5"))

    res = build_transfer_model(ResNet50V2, num_classes, dropout_rate=0.2)
    res.load_weights(os.path.join(model_dir, f"resnet_model_best_{timestamp}.h5"))

    xgb = joblib.load(os.path.join(model_dir, f"xgb_model_{timestamp}.joblib"))

    return eff, res, xgb, index_to_label

def predict_image(path: str,
                  eff_model,
                  res_model,
                  xgb_model,
                  index_to_label: dict):
    raw = tf.io.read_file(path)
    try:
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError:
        img = tf.image.decode_webp(raw, channels=3)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    inp_eff = eff_pre(img)[None, ...]
    inp_res = res_pre(img)[None, ...]

    p_eff = eff_model.predict(inp_eff, verbose=0)
    p_res = res_model.predict(inp_res, verbose=0)
    p_cnn = (p_eff + p_res) / 2.0

    feat_eff = Model(eff_model.input, eff_model.get_layer('gap').output).predict(inp_eff, verbose=0)
    feat_res = Model(res_model.input, res_model.get_layer('gap').output).predict(inp_res, verbose=0)
    feat = np.hstack([feat_eff, feat_res])
    p_xgb = xgb_model.predict_proba(feat)

    ensemble = p_cnn * 0.6 + p_xgb * 0.4
    ensemble = ensemble.flatten()

    idx = int(np.argmax(ensemble))
    label = index_to_label[idx]
    confidence = float(ensemble[idx])

    probs = { index_to_label[i]: float(ensemble[i]) for i in range(len(ensemble)) }

    return label, confidence, probs

def run_prediction(raw_paths: List[str]) -> List[Dict[str, Any]]:
    # ─── 변수 정의 ───────────────────────────────────────────
    MODEL_DIR = 'models' # ******** 모델 경로 ***********
    TIMESTAMP = '20250716_144252' # ******** 모델 명 ***********
    
    # ─── 이미지 경로 정리 ─────────────────────────────────────
    IMAGE_PATHS = []
    
    # predict.py 파일이 있는 디렉터리의 절대 경로를 얻습니다.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for p in raw_paths:
        full_path = os.path.join(base_dir, '..', p)

        if os.path.isdir(full_path):
            for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.webp'):
                IMAGE_PATHS += glob.glob(os.path.join(full_path, ext))
        elif os.path.isfile(full_path):
            IMAGE_PATHS.append(full_path)
        else:
            print(f"경로 없음: {full_path}")

    if not IMAGE_PATHS:
        print("예측할 이미지가 없습니다.")
        return []

    # ─── 모델 로드 & 예측 루프 ─────────────────────────────────
    eff_model, res_model, xgb_model, idx2label = load_models(MODEL_DIR, TIMESTAMP)
    
    results = [] # 예측 결과를 담을 리스트
    for img_path in IMAGE_PATHS:
        label, conf, probs = predict_image(img_path, eff_model, res_model, xgb_model, idx2label)
        
        # 예측 결과를 딕셔너리 형태로 저장
        results.append({
            'file_name': os.path.basename(img_path),
            'predicted_label': label,
            'confidence': conf,
            'probabilities': probs
        })
        print(label, conf, probs, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    for result in results:
        print(result, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    return results # 최종 결과 리스트 반환