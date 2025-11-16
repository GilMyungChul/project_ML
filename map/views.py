# map/views.py
import requests
from django.shortcuts import render
import json

def bike_map_view(request):
    api_key = "414c656f76767776313030616d4a6662" # 발급받은 키로 교체
    
    bike_data_list = []
    
    # 총 3번의 API 호출을 통해 데이터를 가져옵니다.
    # 1-1000, 1001-2000, 2001-3000 (최대치를 고려)
    ranges = [(1, 1000), (1001, 2000), (2001, 3000)]

    for start, end in ranges:
        api_url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start}/{end}/"
        try:
            response = requests.get(api_url)
            # 상태 코드를 확인하고 200 OK가 아니면 예외를 발생시킵니다.
            response.raise_for_status()
            data = response.json()
            
            # 응답 데이터에 'row' 키가 있는지 확인하고 데이터를 추가합니다.
            if data.get('rentBikeStatus') and data['rentBikeStatus'].get('row'):
                bike_data_list.extend(data['rentBikeStatus']['row'])
            else:
                # 데이터가 없으면 반복문을 종료합니다.
                break
                
        except requests.exceptions.RequestException as e:
            print(f"API 호출 중 오류 발생: {e}")
            break
            
    # 모든 데이터를 담을 딕셔너리 생성
    all_data = {
        'rentBikeStatus': {
            'row': bike_data_list
        }
    }
    
    context = {
        'bike_data': all_data,
    }
    return render(request, 'map/bike_map.html', context)