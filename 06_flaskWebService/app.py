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
from flask import send_from_directory
from flask import Flask, render_template, request, jsonify


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

    # 월별 음식 트렌드 TOP 10 쿼리
    food_query = """
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
    cursor.execute(food_query)
    food_trends = cursor.fetchall() # 쿼리 결과

    # 월별 식재료 트렌드 TOP 10 쿼리 추가
    ingredient_query = """
    SELECT 
        i.ingredient_name, 
        COUNT(mi.menu_id) AS usage_frequency
    FROM 
        analysis_results ar
    JOIN 
        menu_ingredients mi ON ar.detected_id = mi.menu_id
    JOIN 
        ingredients i ON mi.ingredient_id = i.ingredient_id
    WHERE 
        ar.result_type = 'FOOD'
        AND i.ingredient_category = 'main'
        AND ar.created_at BETWEEN '2025-06-01 00:00:00' AND '2025-06-30 23:59:59'
    GROUP BY 
        i.ingredient_name
    ORDER BY 
        usage_frequency DESC 
    LIMIT 10;
    """
    cursor.execute(ingredient_query)
    ingredient_trends = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('dashboard.html', food_trends=food_trends, ingredient_trends=ingredient_trends)

@app.route('/api/top5_chart_data')
def get_top5_chart_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    WITH top5_menus AS (
        SELECT m.menu_name
        FROM analysis_results ar
        JOIN images img ON ar.image_id = img.image_id
        JOIN menus m ON ar.detected_id = m.menu_id
        WHERE ar.result_type = 'FOOD'
          AND img.created_at >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        GROUP BY m.menu_name
        ORDER BY COUNT(*) DESC
        LIMIT 5
    )
    SELECT
        m.menu_name,
        DATE(img.created_at) AS upload_date,
        COUNT(*) AS daily_count
    FROM analysis_results ar
    JOIN images img ON ar.image_id = img.image_id
    JOIN menus m ON ar.detected_id = m.menu_id
    WHERE ar.result_type = 'FOOD'
      AND m.menu_name IN (SELECT menu_name FROM top5_menus)
      AND img.created_at >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY m.menu_name, upload_date
    ORDER BY m.menu_name, upload_date;
    """
    cursor.execute(query)
    data = cursor.fetchall()

    cursor.close()
    conn.close()
    return jsonify(data)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """파일 업로드 화면 및 처리"""
    if request.method == 'POST':
        uploaded_files = request.files.getlist('image_files')
        print()
        # 수정된 함수를 호출하고, 저장된 파일 경로 리스트를 받습니다.
        saved_info_list = upload_service.save_uploaded_files(app.config, uploaded_files)

        if saved_info_list:
            # 예측 함수를 호출하고, 결과를 받아옵니다.
            # 만약 menu_id 없으면 예측결과를 저장하지 않았습니다.
            prediction_results = predict_service.save_food_prediction(saved_info_list)

            return render_template('prediction_results.html', results=prediction_results)
        else:
            print("파일이 없습니다.")
            return redirect(request.url)
            
    # GET 요청 시에는 업로드 페이지를 보여줌
    return render_template('upload.html')

# (검색, 결과 등 다른 라우트들도 위와 같이 추가)
@app.route('/menu/<int:menu_id>')
def menu_detail(menu_id):
    menu_name = request.args.get('menu_name')  # 쿼리스트링에서 menu_name 받기

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # menu_id 기반 recipe 조회
    cursor.execute("""
        SELECT cooking_steps, cooking_info, flavor_characteristics,
               nutrition_info, serving_tips, cooking_time_minutes,
               difficulty_level, servings
        FROM menu_recipes
        WHERE menu_id = %s
    """, (menu_id,))
    recipe = cursor.fetchone()

      # 2. 가장 confidence 높은 이미지 조회
    cursor.execute("""
        SELECT img.image_url, ar.confidence_score
        FROM analysis_results AS ar
        JOIN images AS img ON ar.image_id = img.image_id
        WHERE ar.result_type = 'FOOD'
          AND ar.detected_id = %s
        ORDER BY ar.confidence_score DESC
        LIMIT 1;
    """, (menu_id,))
    row = cursor.fetchone()

    image_url = None
    if row:
        full_path = row["image_url"]
        # 상대 경로 추출: upload_datas/부터 시작하는 부분만 추출
        idx = full_path.find("upload_datas")
        if idx != -1:
            image_url = full_path[idx:]  # 예: upload_datas/202507/파일명.jpg

    cursor.close()
    conn.close()
    return render_template(
        'menu_detail.html',
        menu_name=menu_name,
        recipe=recipe,
        image_url= image_url if image_url else None
    )

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('.', filename)  # 현재 디렉토리 기준 상대경로

if __name__ == '__main__':
    app.run(debug=True)