from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('get_prediction/', views.get_prediction,name='get_prediction'),
]
