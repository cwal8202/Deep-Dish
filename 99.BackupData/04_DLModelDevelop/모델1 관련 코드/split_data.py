import os
import shutil
import random
import math

# 1. 기본 경로 설정
# Windows 경로 사용 시, 역슬래시(\) 문제를 피하기 위해 문자열 앞에 r을 붙입니다.
base_path = r"E:\★★★★★AI★★★★★\음식 ai data\selectStart음식DATA\Computer Vision Lab"

print(f"기준 폴더: {base_path}")
print("데이터셋 구성을 시작합니다...")

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15

try:
    # 2. 'Computer Vision Lab' 안의 모든 항목 목록을 가져옵니다.
    all_items = os.listdir(base_path)

    # 3. 폴더만 필터링합니다. (zip, DS_STORE 같은 파일 제외)
    food_folders = [item for item in all_items if os.path.isdir(os.path.join(base_path, item))]
    
    # 4. 스크립트가 생성할 train, valid, test 폴더는 처리 대상에서 제외합니다.
    #    (스크립트를 여러 번 실행해도 안전하도록)
    output_dirs = {'train', 'valid', 'test'}
    food_folders = [folder for folder in food_folders if folder not in output_dirs]

    print(f"\n처리 대상 음식 폴더: {food_folders}\n")

    # 5. 각 음식 폴더를 순회하며 작업 수행
    for food_name in food_folders:
        print(f"--- '{food_name}' 폴더 처리 중 ---")
        
        source_dir = os.path.join(base_path, food_name, 'png')

        # 'png' 폴더가 있는지 확인
        if not os.path.isdir(source_dir):
            print(f"경고: '{source_dir}' 경로에 png 폴더가 없습니다. 건너뜁니다.")
            continue

       # .png 확장자를 가진 파일만 리스트에 포함 (대소문자 구분 없음)
        images = [f for f in os.listdir(source_dir) 
                if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith('.png')]
                
        if not images:
            print(f"경고: '{source_dir}' 폴더에 이미지가 없습니다. 건너뜁니다.")
            continue

        # 6. 이미지 목록을 무작위로 섞음
        random.shuffle(images)
        print("==========================",images)
        
        # 7. 7:1.5:1.5 비율로 분할 지점 계산
        total_count = len(images)
        train_count = math.floor(total_count * TRAIN_RATIO)
        valid_count = math.floor(total_count * VALID_RATIO)
        # test_count는 나머지 전체로 할당하여 소수점 오차로 인한 파일 누락 방지

        train_files = images[:train_count]
        valid_files = images[train_count : train_count + valid_count]
        test_files = images[train_count + valid_count :]
        
        print(f"총 {total_count}개 이미지 -> Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

        # 8. 파일을 이동시킬 목적지 폴더 경로 설정 및 생성
        dest_train_dir = os.path.join(base_path, 'train', food_name)
        dest_valid_dir = os.path.join(base_path, 'valid', food_name)
        dest_test_dir = os.path.join(base_path, 'test', food_name)
        
        os.makedirs(dest_train_dir, exist_ok=True)
        os.makedirs(dest_valid_dir, exist_ok=True)
        os.makedirs(dest_test_dir, exist_ok=True)

        # 9. 파일 이동 함수
        def move_files(files, destination_dir):
            for f in files:
                source_file = os.path.join(source_dir, f)
                dest_file = os.path.join(destination_dir, f)
                shutil.copy2(source_file, dest_file)

        # 파일 이동 실행
        move_files(train_files, dest_train_dir)
        move_files(valid_files, dest_valid_dir)
        move_files(test_files, dest_test_dir)
        
        print(f"'{food_name}' 폴더의 이미지 이동 완료.\n")

    print("모든 작업이 성공적으로 완료되었습니다!")

except FileNotFoundError:
    print(f"오류: 기본 경로 '{base_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")