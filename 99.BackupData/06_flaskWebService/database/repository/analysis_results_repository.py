from database.repository.db_connection import get_db_connection
import mysql.connector
from typing import Dict, Any
from models import AnalysisResult # AnalysisResult Pydantic 모델을 import

def save_analysis_results(analysis_result: AnalysisResult) -> int:
    """
    분석 결과를 DB의 analysis_results 테이블에 저장하고, 생성된 ID를 반환합니다.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO analysis_results 
            (image_id, result_type, detected_id, confidence_score, created_at) 
            VALUES (%(image_id)s, %(result_type)s, %(detected_id)s, %(confidence_score)s, %(created_at)s)
        """
        
        cursor.execute(sql, analysis_result.model_dump())
        conn.commit()
                
        new_result_id = cursor.lastrowid
        return new_result_id
    
    except mysql.connector.Error as e:
        print(f"분석 결과 DB 저장 중 오류 발생: {e}")
        if conn:
            conn.rollback()
        return 0
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()