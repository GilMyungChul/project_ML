# prediction/views.py
from django.shortcuts import render
import requests
import json
import pandas as pd

def predict_map_view(request):
    # 1. API에서 최신 대여소 위치 정보 가져오기
    # YOUR_API_KEY를 실제 키로 교체하세요.
    api_key = "414c656f76767776313030616d4a6662"
    bike_data_list = []
    
    ranges = [(1, 1000), (1001, 2000), (2001, 3000)]
    for start, end in ranges:
        api_url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start}/{end}/"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get('rentBikeStatus') and data['rentBikeStatus'].get('row'):
                bike_data_list.extend(data['rentBikeStatus']['row'])
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"API 호출 중 오류 발생: {e}")
            break
            
    # 2. CSV 파일에서 예측 데이터 불러오기
    try:
        # csv_file_path = "data/test_24.csv"
        csv_file_path = "media/pred_2024.csv"
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return render(request, 'prediction/predict_map.html', {'predict_data': []})

    # 3. 두 데이터를 합치기
    combined_data = []
    df['location_name'] = df['location_name'].str.strip()
    df_dict = df.set_index('location_name').T.to_dict('list')
    # df['대여소명'] = df['대여소명'].str.strip()
    # df_dict = df.set_index('대여소명').T.to_dict('list')

    station_to_focus = request.GET.get('station', '').strip()
    for station in bike_data_list:
        # API의 대여소명에서 번호와 공백을 제거하고 CSV와 매칭시킵니다.
        # 예: "102. 망원역 1번출구 앞" -> "망원역 1번출구 앞"
        api_station_name = station.get('stationName').split('.', 1)[-1].strip()

        if api_station_name in df_dict:
            combined_station = station
            # CSV 컬럼명을 기준으로 예측 값들을 추가합니다.
            combined_station['net_change'] = df_dict[api_station_name][3]
            # combined_station['순변화량'] = df_dict[api_station_name][3]
            combined_station['rental_count'] = df_dict[api_station_name][1]
            # combined_station['대여건수'] = df_dict[api_station_name][1]
            combined_station['return_count'] = df_dict[api_station_name][2]
            # combined_station['반납건수'] = df_dict[api_station_name][2]
            combined_data.append(combined_station)
            
    context = {
        'predict_data': combined_data,
        'station_to_focus': station_to_focus,
    }
    
    return render(request, 'prediction/predict_map.html', context)