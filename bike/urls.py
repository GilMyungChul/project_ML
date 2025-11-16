from django.urls import path
from bike import views

app_name = "bike"
urlpatterns = [
    path("ml/", views.ml_home, name="ml_home"),
    # path("ml/run/", views.ml_train_predict, name="ml_train_predict"),  # 학습/예측 실행
    path("ml/select/", views.ml_select, name="ml_select"),                 # 모델 선택 페이지 (신규)
    path("ml/train_predict/", views.ml_train_predict, name="ml_train_predict"),  # 학습/예측 실행 (기존 이름 재사용)

    path("forecast/select/", views.forecast_select, name="forecast_select"),  # 모델 선택(미래 예측)
    path("forecast/",         views.forecast_home,   name="forecast_home"),   # 업로드 & 예측 실행

]