from database.repository.db_connection import get_db_connection
import mysql.connector
from typing import List, Dict, Any

def get_main_ingredients_top5_by_menu_id(menu_id: int) -> List[Dict[str, Any]] | None:
    """
    메뉴 ID로 해당 메뉴의 메인 재료를 조회합니다.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        sql = """
            SELECT 
                i.ingredient_name, 
                mi.amount, 
                mi.unit 
            FROM 
                menu_ingredients mi
            JOIN 
                ingredients i ON mi.ingredient_id = i.ingredient_id
            WHERE 
                mi.menu_id = %(menu_id)s AND i.ingredient_category = 'main'
            ORDER BY 
                mi.amount ASC
            LIMIT 5
        """
        
        cursor.execute(sql, {'menu_id': menu_id})
        ingredients = cursor.fetchall()
        
        return ingredients if ingredients else None

    except mysql.connector.Error as e:
        print(f"메인 재료 DB 조회 중 오류 발생: {e}")
        return None
        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def get_seasoning_ingredients_top5_by_menu_id(menu_id: int) -> List[Dict[str, Any]] | None:
    """
    메뉴 ID로 해당 메뉴의 양념 재료를 조회합니다.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        sql = """
            SELECT 
                i.ingredient_name, 
                mi.amount, 
                mi.unit 
            FROM 
                menu_ingredients mi
            JOIN 
                ingredients i ON mi.ingredient_id = i.ingredient_id
            WHERE 
                mi.menu_id = %(menu_id)s AND i.ingredient_category = 'seasoning'
            ORDER BY 
                mi.amount ASC
            LIMIT 5
        """
        
        cursor.execute(sql, {'menu_id': menu_id})
        ingredients = cursor.fetchall()
        
        return ingredients if ingredients else None

    except mysql.connector.Error as e:
        print(f"양념 재료 DB 조회 중 오류 발생: {e}")
        return None
        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()