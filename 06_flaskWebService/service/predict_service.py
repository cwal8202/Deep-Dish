from food_predict import predict_final as predictor
from typing import List, Dict, Any
from database.repository import analysis_results_repository
from database.repository import menus_repository
from database.repository import menu_ingredients_repository
from models import AnalysisResult
from datetime import datetime

def save_food_prediction(saved_info_list: list[tuple[int, str]]) -> List[Dict[str, Any]]:
    ## imageid, url 받아서 predict 후 return 받을때 imageid도 같이 감싸서 받아야함.
    
    # 추출한 파일 경로 리스트를 예측 함수에 전달합니다.
    predict_results = predictor.run_prediction(saved_info_list)
    result_type = 'FOOD'

    # 예측 결과를 딕셔너리 형태로 변환
    predict_result_dict_list = []

    # 예측 결과의 label로 meuns 테이블에서 menu_id 찾기
    for predict_result in predict_results:
        menu_id = menus_repository.get_menu_id_by_menu_name(predict_result['predicted_label'])
        status_message = 0
        predict_result_dict = {'image_id': predict_result['image_id'], 'result_type' : result_type,
                               'detected_id': menu_id, 'confidence_score': predict_result['confidence']}

        # analyse_results db 저장
        if menu_id:
            analysis_data = AnalysisResult(
                image_id=predict_result['image_id'],
                result_type=result_type,
                detected_id=menu_id,
                confidence_score=predict_result['confidence'],
                created_at=datetime.now()
            )
            status_message = analysis_results_repository.save_analysis_results(analysis_data)

        # detected_id 로 menu_ingredients 조회하여 재료 top5를 가져와서 predict_result_dict에 저장
        main_ingredients = menu_ingredients_repository.get_main_ingredients_top5_by_menu_id(menu_id)
        seasoning_ingredients = menu_ingredients_repository.get_seasoning_ingredients_top5_by_menu_id(menu_id)

        predict_result_dict['main_ingredients'] = main_ingredients
        predict_result_dict['seasoning_ingredients'] = seasoning_ingredients
        predict_result_dict['predicted_label'] = predict_result['predicted_label']
        predict_result_dict['probabilities'] = predict_result['probabilities']
        predict_result_dict['status_message'] = status_message
        predict_result_dict_list.append(predict_result_dict)

    return predict_result_dict_list

