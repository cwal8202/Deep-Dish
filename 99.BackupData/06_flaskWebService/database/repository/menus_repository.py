import mysql.connector
from models import Menu
from database.repository.db_connection import get_db_connection

def get_menu_id_by_menu_name(menu_name: str) -> int:
    """메뉴 이름으로 메뉴 ID 조회"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "SELECT menu_id FROM menus WHERE menu_name = %(menu_name)s"
        cursor.execute(sql, {'menu_name': menu_name})
        result  = cursor.fetchone()
        if result :
            return result[0]
        else:
            return None

    except mysql.connector.Error as e:
        print(f"메뉴 DB 오류 발생: {e}")
        if conn:
            conn.rollback()
        return None # 실패 시 None 반환
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()