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

def build_transfer_model_with_gap_output(base_cls, num_classes, dropout_rate=0.2):
    base = base_cls(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D(name='gap')(base.output)
    x_fc = Dense(256, activation='relu')(x)
    x_drop = Dropout(dropout_rate)(x_fc)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x_drop)
    model = Model(inputs=base.input, outputs=[x, out])  # GAP 출력 + Softmax 출력 반환
    return model

def load_models(model_dir: str, timestamp: str):
    label_map = joblib.load(os.path.join(model_dir, f"label_to_index_{timestamp}.joblib"))
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    eff_model = build_transfer_model_with_gap_output(EfficientNetB0, num_classes, dropout_rate=0.2)
    eff_model.load_weights(os.path.join(model_dir, f"effnet_model_best_{timestamp}.h5"))

    res_model = build_transfer_model_with_gap_output(ResNet50V2, num_classes, dropout_rate=0.2)
    res_model.load_weights(os.path.join(model_dir, f"resnet_model_best_{timestamp}.h5"))

    xgb_model = joblib.load(os.path.join(model_dir, f"xgb_model_{timestamp}.joblib"))

    return eff_model, res_model, xgb_model, index_to_label

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

    # GAP + Softmax 동시에 추출
    feat_eff, p_eff = eff_model.predict(inp_eff, verbose=0)
    feat_res, p_res = res_model.predict(inp_res, verbose=0)

    p_cnn = (p_eff + p_res) / 2.0

    feat = np.hstack([feat_eff, feat_res])
    p_xgb = xgb_model.predict_proba(feat)

    ensemble = p_cnn * 0.6 + p_xgb * 0.4
    ensemble = ensemble.flatten()

    idx = int(np.argmax(ensemble))
    label = index_to_label[idx]
    confidence = float(ensemble[idx])
    probs = {index_to_label[i]: float(ensemble[i]) for i in range(len(ensemble))}

    return label, confidence, probs

def run_prediction(saved_info_list: list[tuple[int, str]]) -> List[Dict[str, Any]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(base_dir, 'models')
    TIMESTAMP = '20250716_144252'

    eff_model, res_model, xgb_model, idx2label = load_models(MODEL_DIR, TIMESTAMP)

    results = []
    for image_id, img_path in saved_info_list:
        full_path = os.path.join(base_dir, '..', img_path)

        if not os.path.isfile(full_path):
            print(f"경로 없음: {full_path}")
            continue

        label, conf, probs = predict_image(full_path, eff_model, res_model, xgb_model, idx2label)
        kor_label = label_eng_to_kor.ENG_TO_KOR_MENU_MAP[label]

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
            'confidence': round(float(conf), 2),
            'probabilities': top_5_probs_kor
        })
    
    return results
