from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from datetime import datetime

from database.repository.db_connection import get_db_connection
import service.upload_service as upload_service
import service.predict_service as predict_service


load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'upload_datas')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 개선된 음식 트렌드 TOP 10 쿼리
    food_query = """
    SELECT
      ar.detected_id,
      m.menu_name,
      COUNT(*) AS mention_count
    FROM analysis_results ar
    JOIN menus m ON ar.detected_id = m.menu_id
    WHERE ar.result_type = 'FOOD'
      AND EXISTS (
        SELECT 1 FROM images i
        WHERE i.image_id = ar.image_id
          AND i.created_at BETWEEN '2025-06-01' AND '2025-06-30'
      )
    GROUP BY ar.detected_id, m.menu_name
    ORDER BY mention_count DESC
    LIMIT 10;
    """
    cursor.execute(food_query)
    food_trends = cursor.fetchall()

    # 개선된 식재료 트렌드 TOP 10 쿼리
    ingredient_query = """
    SELECT
      ing.ingredient_name,
      COUNT(*) AS usage_count
    FROM menu_ingredients mi
    JOIN ingredients ing ON mi.ingredient_id = ing.ingredient_id
    WHERE ing.ingredient_category = 'main'
      AND EXISTS (
        SELECT 1
        FROM analysis_results ar
        JOIN images img ON ar.image_id = img.image_id
        WHERE ar.detected_id = mi.menu_id
          AND ar.result_type = 'FOOD'
          AND img.created_at BETWEEN '2025-06-01' AND '2025-06-30'
      )
    GROUP BY ing.ingredient_name
    ORDER BY usage_count DESC
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
          AND img.created_at >= DATE_SUB(CURDATE(), INTERVAL 25 MONTH)
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
      AND img.created_at >= DATE_SUB(CURDATE(), INTERVAL 25 MONTH)
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
    if request.method == 'POST':
        uploaded_files = request.files.getlist('image_files')
        saved_info_list = upload_service.save_uploaded_files(app.config, uploaded_files)

        if saved_info_list:
            prediction_results = predict_service.save_food_prediction(saved_info_list)
            return render_template('prediction_results.html', results=prediction_results)
        else:
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/menu/<int:menu_id>')
def menu_detail(menu_id):
    menu_name = request.args.get('menu_name')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT cooking_steps, cooking_info, flavor_characteristics,
               nutrition_info, serving_tips, cooking_time_minutes,
               difficulty_level, servings
        FROM menu_recipes
        WHERE menu_id = %s
    """, (menu_id,))
    recipe = cursor.fetchone()

    cursor.execute("""
        SELECT img.image_url, ar.confidence_score
        FROM analysis_results ar
        JOIN images img ON ar.image_id = img.image_id
        WHERE ar.result_type = 'FOOD'
          AND ar.detected_id = %s
        ORDER BY ar.confidence_score DESC
        LIMIT 1;
    """, (menu_id,))
    row = cursor.fetchone()

    image_url = None
    is_external = False  # 🔹 외부 이미지 여부 추가

    if row:
        full_path = row["image_url"]
        if full_path.startswith("http://") or full_path.startswith("https://"):
            image_url = full_path
            is_external = True
        else:
            idx = full_path.find("upload_datas")
            if idx != -1:
                image_url = full_path[idx:]

    cursor.close()
    conn.close()

    return render_template(
        'menu_detail.html',
        menu_name=menu_name,
        recipe=recipe,
        image_url=image_url,
        is_external=is_external
    )

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
