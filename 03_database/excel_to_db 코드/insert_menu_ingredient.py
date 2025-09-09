# pip install pandas mysql-connector-python openpyxl
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

# --- 파일 및 시트 정보 ---
EXCEL_FILE_PATH = '병합_레시피_항목그대로병합.xlsx'
SHEET_NAMES = ['주요식재료', '양념소스'] # 처리할 시트 목록

def parse_quantity_simple(value):
    """주어진 규칙에 따라 문자열을 파싱하고, unit의 띄어쓰기를 정리합니다."""
    if pd.isna(value) or value == '-':
        return None, None
    value_str = str(value).strip()
    amount, unit = None, None
    match_fraction = re.match(r'^(\d+\.?\d*)/(\d+\.?\d*)\s*(.*)', value_str)
    if match_fraction:
        numerator = float(match_fraction.group(1))
        denominator = float(match_fraction.group(2))
        if denominator != 0:
            amount = round(numerator / denominator, 2)
            unit = match_fraction.group(3).strip() if match_fraction.group(3) else None
    elif re.match(r'^\d', value_str):
        match_normal = re.match(r'^(\d+\.?\d*)\s*(.*)', value_str)
        if match_normal:
            amount = round(float(match_normal.group(1)), 2)
            unit = match_normal.group(2).strip() if match_normal.group(2) else None
    else:
        unit = value_str
    if unit and ' ' in unit:
        unit = unit.split(' ')[0]
    return amount, unit

def process_sheet(sheet_name, db_conn, cursor, menu_map, ingredient_map):
    """지정된 시트를 읽고 처리하여 DB에 저장하는 함수"""
    print(f"\n--- '{sheet_name}' 시트 처리 시작 ---")
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name)
        ingredient_col_name = df.columns[0]
        df.rename(columns={ingredient_col_name: 'ingredient_name'}, inplace=True)
        print(f"'{sheet_name}' 시트를 성공적으로 읽었습니다.")

        data_to_insert = []
        for index, row in df.iterrows():
            ingredient_name = row['ingredient_name']
            if ingredient_name not in ingredient_map:
                continue
            ingredient_id = ingredient_map[ingredient_name]

            for menu_name in df.columns[1:]:
                if menu_name not in menu_map:
                    continue
                
                menu_id = menu_map[menu_name]
                cell_value = row[menu_name]
                amount, unit = parse_quantity_simple(cell_value)
                
                if amount is not None or unit is not None:
                    data_to_insert.append((menu_id, ingredient_id, amount, unit))
        
        if data_to_insert:
            sql_insert = """
                INSERT INTO menu_ingredients (menu_id, ingredient_id, amount, unit) 
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE amount=VALUES(amount), unit=VALUES(unit)
            """
            cursor.executemany(sql_insert, data_to_insert)
            db_conn.commit()
            print(f"총 {cursor.rowcount}개의 데이터를 '{sheet_name}' 시트로부터 저장(또는 업데이트)했습니다.")
        else:
            print(f"'{sheet_name}' 시트에서 저장할 데이터를 찾지 못했습니다.")

    except Exception as e:
        print(f"'{sheet_name}' 시트 처리 중 오류 발생: {e}")
        db_conn.rollback()


def run_importer():
    """전체 임포트 프로세스를 실행하는 메인 함수"""
    db_conn = None
    cursor = None
    try:
        db_conn = mysql.connector.connect(**DB_CONFIG)
        cursor = db_conn.cursor()
        print("MySQL DB에 성공적으로 연결되었습니다.")

        cursor.execute("SELECT menu_id, menu_name FROM menus")
        menu_map = {name: id for id, name in cursor.fetchall()}
        
        cursor.execute("SELECT ingredient_id, ingredient_name FROM ingredients")
        ingredient_map = {name: id for id, name in cursor.fetchall()}
        print("메뉴 및 식재료 ID 맵을 생성했습니다.")
        
        # 지정된 모든 시트에 대해 처리 함수를 순차적으로 호출
        for sheet in SHEET_NAMES:
            process_sheet(sheet, db_conn, cursor, menu_map, ingredient_map)

    except Error as e:
        print(f"DB 작업 중 에러 발생: {e}")
    finally:
        if db_conn and db_conn.is_connected():
            cursor.close()
            db_conn.close()
            print("\nMySQL 연결이 해제되었습니다.")

if __name__ == '__main__':
    run_importer()