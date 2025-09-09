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
import label_eng_to_kor


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

def run_prediction(saved_info_list: list[tuple[int, str]]) -> List[Dict[str, Any]]:
    # predict.py 파일이 있는 디렉터리의 절대 경로를 얻습니다.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ─── 변수 정의 ───────────────────────────────────────────
    MODEL_DIR = os.path.join(base_dir, 'models') # ******** 모델 경로 ***********
    TIMESTAMP = '20250716_144252' # ******** 모델 명 ***********

    # ─── 모델 로드 & 예측 루프 ─────────────────────────────────
    eff_model, res_model, xgb_model, idx2label = load_models(MODEL_DIR, TIMESTAMP)

    results = [] # 예측 결과를 담을 리스트

    # saved_info_list를 순회하며 image_id와 filepath를 각각 가져옵니다.
    for image_id, img_path in saved_info_list:
        full_path = os.path.join(base_dir, '..', img_path)

        # 파일이 존재하는지 확인하는 로직
        if not os.path.isfile(full_path):
            print(f"경로 없음: {full_path}")
            continue # 다음 파일로 넘어감

        # 예측 실행
        label, conf, probs = predict_image(full_path, eff_model, res_model, xgb_model, idx2label)
        kor_label = label_eng_to_kor.ENG_TO_KOR_MENU_MAP[label]

        # 상위 5개 항목 추출 로직
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        top5_probs_eng = dict(sorted_probs[:5])
        # 딕셔너리 컴프리헨션을 사용해 키를 한글로 변경하고 값은 반올림
        top_5_probs_kor = {
            # get() 메서드로 안전하게 값 가져오기
            label_eng_to_kor.ENG_TO_KOR_MENU_MAP.get(k, k): round(v, 2)
            for k, v in top5_probs_eng.items()
        }

        # 예측 결과를 딕셔너리 형태로 저장
        results.append({
            'image_id': image_id, # image_id 추가
            'file_path': os.path.basename(full_path),
            'predicted_label': kor_label,
            'confidence': round(float(conf), 2),
            'probabilities': top_5_probs_kor
        })
    
    return results # 최종 결과 리스트 반환