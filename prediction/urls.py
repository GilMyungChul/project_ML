# prediction/urls.py
from django.urls import path
from . import views

app_name = 'prediction'
urlpatterns = [
    path('', views.predict_map_view, name='predict_map'),
]