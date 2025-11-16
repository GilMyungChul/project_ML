# map/urls.py
from django.urls import path
from . import views

app_name = 'map'
urlpatterns = [
    path('', views.bike_map_view, name='bike_map'),
]