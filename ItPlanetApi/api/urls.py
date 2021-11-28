from django.urls import include, path
from . import views

urlpatterns = [
  path('welcome', views.welcome),
  path('similar', views.find_similar)
]