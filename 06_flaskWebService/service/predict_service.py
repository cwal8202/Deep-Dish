from food_predict import predict_test2_3gpt as predictor
from typing import List, Dict, Any
from database.repository import analysis_results_repository

def save_food_prediction(saved_info_list: list[tuple[int, str]]):
    filepaths = [info[1] for info in saved_info_list]
    
    # 추출한 파일 경로 리스트를 예측 함수에 전달합니다.
    predict_results = predictor.run_prediction(filepaths)
    result_type = 'FOOD'
    
    # 예측 결과를 딕셔너리 형태로 변환
    predict_results_dict = [{'image_id': info[0], 'result_type' : result_type,'predicted_label': result['predicted_label'], 'confidence': result['confidence'], 'probabilities': result['probabilities']} for info, result in zip(saved_info_list, predict_results)]

    # analyse_results db 저장
    analysis_results_repository.save_analysis_results(saved_info_list, predict_results)