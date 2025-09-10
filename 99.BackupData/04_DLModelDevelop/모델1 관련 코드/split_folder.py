import os
import shutil
import random
import argparse

def split_data(src_dir, dest_dir, train_ratio, val_ratio, test_ratio, seed=42):
    """
    src_dir: 원본 데이터 루트 (각 클래스 폴더 안에 'png' 서브폴더가 있어야 함)
    dest_dir: 분할된 데이터를 저장할 상위 디렉터리
    train_ratio, val_ratio, test_ratio: 학습/검증/테스트 비율 (합이 1.0)
    seed: 랜덤 시드
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    random.seed(seed) 
    for cls in os.listdir(src_dir):
        png_dir = os.path.join(src_dir, cls, 'png')
        if not os.path.isdir(png_dir):
            print(f"[경고] '{cls}'에 'png' 서브폴더가 없습니다. 건너뜁니다.")
            continue

        images = [f for f in os.listdir(png_dir) if f.lower().endswith('.png')]
        print(f"[{cls}] 총 {len(images)}개 PNG 이미지 발견")
        if not images:
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val   = int(val_ratio   * n_total)

        splits = {
            'train': images[:n_train],
            'valid': images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:]
        }

        for split_name, files in splits.items():
            out_dir = os.path.join(dest_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for fname in files:
                shutil.copy2(
                    os.path.join(png_dir, fname),
                    os.path.join(out_dir, fname)
                )
        print(f" → {cls}: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

    print("Dataset 분할 완료.")


src = "E:\★★★★★AI★★★★★\음식 ai data\selectStart음식DATA\Computer Vision Lab"
dest = "E:\★★★★★AI★★★★★\음식 ai data\selectStart음식DATA\dataset_splits"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

split_data(src, dest, train_ratio, val_ratio, test_ratio, seed=42)