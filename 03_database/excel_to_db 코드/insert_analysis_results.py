# pip install pandas mysql-connector-python openpyxl
import os
import glob
import pandas as pd
import mysql.connector
from mysql.connector import Error
import random

# --- DB 연결 정보 ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '0000',
    'database': 'deep_dish'
}

# --- 처리할 파일 정보 ---
DIRECTORY_PATH = r'C:\ai_x\Deep-Dish\N시기별음식URL수집\N월별조회메뉴수집'
SHEET_NAME = 'URL목록'

def populate_analysis_results():
    db_conn = None
    cursor = None
    try:
        db_conn = mysql.connector.connect(**DB_CONFIG)
        cursor = db_conn.cursor()
        print("MySQL DB에 성공적으로 연결되었습니다.")

        # 1. DB에서 image와 menu 정보를 미리 모두 가져와 딕셔너리로 만듭니다. (성능 최적화)
        cursor.execute("SELECT image_id, image_url FROM images")
        url_to_image_id_map = {url: id for id, url in cursor.fetchall()}
        
        cursor.execute("SELECT menu_id, menu_name FROM menus")
        menu_name_to_id_map = {name: id for id, name in cursor.fetchall()}
        print("이미지 URL 및 메뉴 ID 맵을 생성했습니다.")

        # 2. 지정된 폴더의 모든 .xlsx 파일을 찾습니다.
        excel_files = glob.glob(os.path.join(DIRECTORY_PATH, '*.xlsx'))
        
        if not excel_files:
            print(f"'{DIRECTORY_PATH}' 폴더에서 엑셀 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(excel_files)}개의 엑셀 파일을 찾았습니다.")

        # 3. 각 엑셀 파일을 순회하며 처리합니다.
        for file_path in excel_files:
            file_name = os.path.basename(file_path)
            print(f"\n--- '{file_name}' 파일 처리 시작 ---")

            try:
                # 4. 엑셀 파일의 'URL목록' 시트를 읽습니다.
                df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
                
                # 첫 번째와 세 번째 컬럼의 이름을 가져옵니다.
                menu_col_name = df.columns[0]
                url_col_name = 'URL' # 컬럼명이 'URL'이라고 가정

                if url_col_name not in df.columns:
                    print(f"[경고] '{SHEET_NAME}' 시트에서 '{url_col_name}' 컬럼을 찾을 수 없습니다.")
                    continue

                # 5. DB에 저장할 데이터 리스트를 생성합니다.
                data_to_insert = []
                for index, row in df.iterrows():
                    menu_name = row[menu_col_name]
                    image_url = row[url_col_name]

                    # 맵(딕셔너리)에서 ID 조회
                    image_id = url_to_image_id_map.get(image_url)
                    menu_id = menu_name_to_id_map.get(menu_name)

                    # ID를 모두 찾은 경우에만 데이터 추가
                    if image_id and menu_id:
                        confidence = round(random.uniform(0.5, 0.8), 4) # 0.5~0.8 사이 랜덤 소수
                        data_to_insert.append((image_id, 'FOOD', menu_id, confidence))
                    else:
                        if not image_id: print(f"[경고] DB에 없는 URL: {image_url}")
                        if not menu_id: print(f"[경고] DB에 없는 메뉴: {menu_name}")

                # 6. executemany를 사용해 DB에 데이터를 한 번에 INSERT 합니다.
                if data_to_insert:
                    sql_insert = """
                        INSERT INTO analysis_results (image_id, result_type, detected_id, confidence_score) 
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.executemany(sql_insert, data_to_insert)
                    db_conn.commit()
                    print(f"총 {cursor.rowcount}개의 분석 결과를 DB에 저장했습니다.")
                else:
                    print("저장할 분석 결과 데이터가 없습니다.")

            except Exception as e:
                print(f"'{file_name}' 파일 처리 중 오류 발생: {e}")

    except Error as e:
        print(f"DB 작업 중 에러 발생: {e}")
    finally:
        if db_conn and db_conn.is_connected():
            cursor.close()
            db_conn.close()
            print("\nMySQL 연결이 해제되었습니다.")

# --- 스크립트 실행 ---
if __name__ == '__main__':
    populate_analysis_results()