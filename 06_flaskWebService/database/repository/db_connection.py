import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

# --- DB 연결 설정 ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

def get_db_connection():
    """요청이 있을 때마다 새로운 DB 연결을 생성하여 반환하는 함수"""
    conn = mysql.connector.connect(**DB_CONFIG)
    return conn