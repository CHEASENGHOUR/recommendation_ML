from django.urls import path
from . import views

urlpatterns = [
    # Main API
    path('recommend/', views.recommend, name='recommend'),
    path('health/', views.health, name='health'),
    path('laptop/<int:laptop_id>/', views.laptop_detail, name='laptop_detail'),
    
    # Admin
    path('admin/reload/', views.admin_reload, name='admin_reload'),
]