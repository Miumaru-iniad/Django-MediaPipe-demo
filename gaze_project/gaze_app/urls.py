# gaze_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video, name='upload_video'),  # メインページ用のURL
    path('download_video/', views.download_video, name='download_video'),  # ダウンロードページ用のURL
]