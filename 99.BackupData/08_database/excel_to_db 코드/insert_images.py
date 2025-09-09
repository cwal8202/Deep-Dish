# pip install pandas mysql-connector-python openpyxl
import os
import glob
import pandas as pd
import mysql.connector
from mysql.connector import Error
import re

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

def import_image_urls():
    db_conn = None
    cursor = None
    try:
        db_conn = mysql.connector.connect(**DB_CONFIG)
        cursor = db_conn.cursor()
        print("MySQL DB에 성공적으로 연결되었습니다.")

        excel_files = glob.glob(os.path.join(DIRECTORY_PATH, '*.xlsx'))
        
        if not excel_files:
            print(f"'{DIRECTORY_PATH}' 폴더에서 엑셀 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(excel_files)}개의 엑셀 파일을 찾았습니다.")

        for file_path in excel_files:
            file_name = os.path.basename(file_path)
            print(f"\n--- '{file_name}' 파일 처리 시작 ---")

            match = re.search(r'_(\d{6})_', file_name)
            if not match:
                print(f"[경고] 파일명에서 날짜를 찾을 수 없어 건너<binary data, 2 bytes>니다: {file_name}")
                continue
            
            year_month = match.group(1)
            year = year_month[:4]
            month = year_month[4:]
            created_at_str = f"{year}-{month}-01 00:00:00"

            try:
                # === 수정된 부분 ===
                # header=None 옵션을 제거하여 첫 행을 컬럼명으로 인식하게 하고,
                # 'URL' 컬럼을 이름으로 직접 선택합니다.
                df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
                
                # 'URL'이라는 컬럼이 있는지 확인
                if 'URL' not in df.columns:
                    print(f"[경고] '{SHEET_NAME}' 시트에서 'URL' 컬럼을 찾을 수 없습니다.")
                    continue

                urls = df['URL'].dropna().tolist()
                
                if not urls:
                    print(f"'{SHEET_NAME}' 시트의 'URL' 컬럼에서 URL을 찾을 수 없습니다.")
                    continue

                data_to_insert = [(url, 'naver', created_at_str) for url in urls]

                sql_insert = "INSERT INTO images (image_url, image_source, created_at) VALUES (%s, %s, %s)"
                cursor.executemany(sql_insert, data_to_insert)
                db_conn.commit()
                print(f"총 {cursor.rowcount}개의 URL을 DB에 저장했습니다. (날짜: {created_at_str})")

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
    import_image_urls()