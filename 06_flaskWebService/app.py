# app.py
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import mysql.connector
import os # os 모듈 추가
from dotenv import load_dotenv # dotenv 모듈 추가
from database.repository.db_connection import get_db_connection
from database.repository import menus_repository
from datetime import datetime
import uploads

load_dotenv() # .env 파일에서 환경 변수를 불러옴

app = Flask(__name__)

# 1) 업로드 폴더 경로를 선언하고, 없으면 생성
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def dashboard():
    """트렌드 분석 대시보드 화면"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True) # 결과를 딕셔너리로 받기

    test_data = menus_repository.get_menus(cursor)

    # 예시: 월별 음식 트렌드 TOP 10 쿼리
    query = """
    SELECT m.menu_name, COUNT(ar.result_id) AS mention_count
    FROM images AS i
    JOIN analysis_results AS ar ON i.image_id = ar.image_id
    JOIN menus AS m ON ar.detected_id = m.menu_id
    WHERE
        -- ❗️이곳의 날짜를 바꿔서 원하는 기간의 트렌드를 볼 수 있습니다.
        i.created_at BETWEEN '2025-06-01 00:00:00' AND '2025-06-30 23:59:59'
        AND ar.result_type = 'FOOD'
    GROUP BY m.menu_name
    ORDER BY mention_count DESC LIMIT 10;
    """
    cursor.execute(query)
    food_trends = cursor.fetchall() # 쿼리 결과를 가져옴

    cursor.close()
    conn.close()

    # DB에서 가져온 데이터를 HTML 템플릿으로 전달
    return render_template('dashboard.html', food_trends=food_trends)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """파일 업로드 화면 및 처리"""
    if request.method == 'POST':
        uploaded_files = uploads.upload_file(request.files.getlist('image_files'))
        # 'image_files'라는 이름으로 전송된 파일들을 리스트로 받음
        uploaded_files = request.files.getlist('image_files')

        if not uploaded_files or uploaded_files[0].filename == '':
            print("파일이 선택되지 않았습니다.")
            return redirect(request.url)

        # 1) 월별 폴더명 생성 (YYYYMM)
        month_folder = datetime.now().strftime("%Y%m")
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], month_folder), exist_ok=True)

        for file in uploaded_files:
            if file:
                # 안전한 파일 이름으로 변경
                filename = secure_filename(file.filename)
                # 파일을 지정된 폴더에 저장
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], month_folder,filename))
                print(f"'{filename}' 파일 저장 완료.")
        
        # 업로드 완료 후, 결과 페이지나 대시보드로 리다이렉트
        # return redirect(url_for('dashboard'))
        return "파일 업로드 성공!" # 임시 메시지

    # GET 요청 시에는 업로드 페이지를 보여줌
    return render_template('upload.html')

# (검색, 결과 등 다른 라우트들도 위와 같이 추가)

if __name__ == '__main__':
    app.run(debug=True)