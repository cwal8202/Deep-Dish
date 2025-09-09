import os
from datetime import datetime
from werkzeug.utils import secure_filename
import mimetypes # mimetypes 모듈 추가
import mysql
from models import Image
from database.repository import images_repository
import uuid # uuid 모듈 추가

# 허용할 파일 확장자 및 MIME 타입 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    """
    파일 확장자가 허용된 확장자인지 확인하는 함수
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 업로드 파일 저장, db에 저장
def save_uploaded_files(app_config, uploaded_files):
    """
    이미지 파일을 지정폴더(uploads), db에 저장
    """
    print(uploaded_files, uploaded_files[0].filename, type(uploaded_files[0].filename))
    if not uploaded_files:
        print("파일이 없습니다.")
        return []

    month_folder = datetime.now().strftime("%Y%m")
    save_path = os.path.join(app_config['UPLOAD_FOLDER'], month_folder)
    os.makedirs(save_path, exist_ok=True)
    
    # 튜플 (new_image_id, filepath)를 담을 리스트
    saved_info_list = []
    
    for file in uploaded_files:
        if file and file.filename != '' and allowed_file(file.filename):
            try:
                # 1. 파일 이름을 고유하게 변경하고 저장 경로 설정
                original_filename, extension = os.path.splitext(file.filename)
                unique_filename = f"{uuid.uuid4()}{extension}" # uuid로 고유한 이름 생성
                full_filepath = os.path.join(save_path, unique_filename)
                
                # 2. 파일을 물리적으로 저장
                file.save(full_filepath)

                # 3. DB에 이미지 정보 저장
                image_info = Image(image_url=full_filepath, image_source='upload', created_at=datetime.now())
                new_image_id = images_repository.add_image(image_info)

                if new_image_id:
                    print(f"'{unique_filename}' 파일 저장 및 DB 기록 완료. ID: {new_image_id}")
                    saved_info_list.append((new_image_id, full_filepath))
                else:
                    # DB 저장 실패 시, 이미 저장된 파일 삭제 (롤백)
                    os.remove(full_filepath)
                    print(f"'{filename}' 파일 저장 성공, 그러나 DB 기록 실패! 파일 삭제됨.")
            
            except mysql.connector.Error as e:
                print(f"DB 오류 발생: {e}")
                # 파일은 이미 저장되었으므로, 이 상황에 대한 처리가 필요
                os.remove(full_filepath)
            except Exception as e:
                print(f"파일 처리 중 예상치 못한 오류 발생: {e}")

        else:
            if file.filename == '':
                print("빈 파일 객체는 무시합니다.")
            else:
                print(f"'{file.filename}' 파일은 허용되지 않는 형식입니다. 저장되지 않았습니다.")
    
    # 성공적으로 저장된 파일의 경로 리스트를 반환
    return saved_info_list