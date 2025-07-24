# app.py
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import mysql.connector
import os
from dotenv import load_dotenv
from database.repository.db_connection import get_db_connection
from database.repository import menus_repository
from datetime import datetime
import service.upload_service as upload_service
import service.predict_service as predict_service

load_dotenv() # .env 파일에서 환경 변수를 불러옴

app = Flask(__name__)

# 1) 업로드 폴더 경로를 선언하고, 없으면 생성
UPLOAD_FOLDER = os.path.join(app.root_path, 'upload_datas')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def dashboard():
    """트렌드 분석 대시보드 화면"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True) # 결과를 딕셔너리로 받기

    test_data = menus_repository.get_menus(cursor)

    # 월별 음식 트렌드 TOP 10 쿼리
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
    food_trends = cursor.fetchall() # 쿼리 결과

    cursor.close()
    conn.close()

    # DB에서 가져온 데이터를 HTML 템플릿으로 전달
    return render_template('dashboard.html', food_trends=food_trends)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """파일 업로드 화면 및 처리"""
    if request.method == 'POST':
        uploaded_files = request.files.getlist('image_files')
        
        # 수정된 함수를 호출하고, 저장된 파일 경로 리스트를 받습니다.
        saved_info_list = upload_service.save_uploaded_files(app.config, uploaded_files)

        if saved_info_list:
            # 예측 함수를 호출하고, 결과를 받아옵니다.
            prediction_results = predict_service.save_food_prediction(saved_info_list)
            # prediction_results = predictor.run_prediction(saved_filepaths)

            # 예측 결과를 담아 결과 페이지로 전달
            return render_template('prediction_results.html', results=prediction_results)
        else:
            print("파일이 없습니다.")
            return redirect(request.url)
            
    # GET 요청 시에는 업로드 페이지를 보여줌
    return render_template('upload.html')

# (검색, 결과 등 다른 라우트들도 위와 같이 추가)

if __name__ == '__main__':
    app.run(debug=True)