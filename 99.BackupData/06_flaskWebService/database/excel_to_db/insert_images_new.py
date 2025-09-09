# pip install pandas mysql-connector-python openpyxl
import os
import glob
import pandas as pd
import mysql.connector
from mysql.connector import Error
import re

# --- DB 연결 정보 ---
DB_CONFIG = {
    'host': '111.118.39.151',
    'user': 'deepdish',
    'password': 'deepdish',
    'database': 'deep_dish'
}

# --- 처리할 파일 정보 ---
DIRECTORY_PATH = r'C:\ai_x\Deep-Dish\N월별조회메뉴수집_비중반영_예측가능메뉴한정요약_결과'
SHEET_NAME = 'url목록'

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
                print(f"[경고] 파일명에서 날짜를 찾을 수 없어 건너뜁니다: {file_name}")
                continue
            
            year_month = match.group(1)
            year = year_month[:4]
            month = year_month[4:]
            created_at_str = f"{year}-{month}-01 00:00:00"

            try:
                df = pd.read_excel(file_path, sheet_name=SHEET_NAME)

                # 컬럼 존재 여부 확인
                required_columns = {'URL', '메뉴', '예측_음식명'}
                if not required_columns.issubset(df.columns):
                    print(f"[경고] 필수 컬럼({required_columns})이 부족합니다.")
                    continue

                # '메뉴'와 '예측_음식명'이 같은 행만 필터링
                matched_df = df[df['메뉴'] == df['예측_음식명']].copy()

                if matched_df.empty:
                    print(f"[안내] '{file_name}'에서 조건에 맞는 데이터가 없습니다.")
                    continue

                # 유효한 URL만 추출
                urls = matched_df['URL'].dropna().tolist()
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
