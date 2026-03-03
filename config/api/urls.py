from django.urls import path
from . import views

urlpatterns = [
    # ── Main recommendation
    # POST  — supports text_search / similar / preferences / personalized
    path("recommend/", views.recommend, name="recommend"),

    # ── Health check
    # GET
    path("health/", views.health, name="health"),

    # ── Single laptop detail
    # GET
    path("laptop/<int:laptop_id>/", views.laptop_detail, name="laptop_detail"),

    # ── Admin: hot-reload model without restarting Django
    # POST
    path("admin/reload/", views.admin_reload, name="admin_reload"),
]