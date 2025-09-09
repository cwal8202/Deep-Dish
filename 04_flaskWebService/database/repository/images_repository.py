from database.repository.db_connection import get_db_connection  # DB 연결 함수 임포트
from typing import List, Dict
import mysql.connector
from models import Image

def add_image(upload_image: Image) -> int:
    """
    이미지를 DB에 저장하고, 생성된 ID를 반환합니다.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO images (image_url, image_source, created_at) VALUES (%(image_url)s, %(image_source)s, %(created_at)s)"
        cursor.execute(sql, upload_image.model_dump())
        conn.commit()
        
        # INSERT 쿼리로 생성된 행의 ID를 가져옵니다.
        # MySQL Connector에서 cursor.lastrowid를 사용합니다.
        new_image_id = cursor.lastrowid
        
        return new_image_id

    except mysql.connector.Error as e:
        print(f"이미지 DB 저장 중 오류 발생: {e}")
        if conn:
            conn.rollback()
        return 0 # 실패 시 0 반환
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()