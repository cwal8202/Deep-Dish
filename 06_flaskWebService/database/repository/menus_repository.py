def get_menus(cursor):
    """메뉴 목록 조회"""    
    query = """
        select * from menus;
    """
    cursor.execute(query)
    return cursor.fetchall()

def get_menu(menu_id):
    """메뉴 상세 조회"""
    pass
    