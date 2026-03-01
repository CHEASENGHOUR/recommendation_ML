"""
api/urls.py
-----------
URL routing for the Laptop Recommendation API.

Include in config/urls.py:
    from django.urls import path, include
    urlpatterns = [
        path("api/", include("api.urls")),
    ]

Then access via:
    POST http://127.0.0.1:8000/api/recommend/
    GET  http://127.0.0.1:8000/api/health/
    GET  http://127.0.0.1:8000/api/laptop/42/
    POST http://127.0.0.1:8000/api/admin/reload/
"""

from django.urls import path
from . import views

urlpatterns = [
    # ── Main recommendation endpoint ───────────────────────────────────
    # POST  — supports text_search / similar / preferences / personalized
    path("recommend/", views.recommend, name="recommend"),

    # ── Health check ───────────────────────────────────────────────────
    # GET
    path("health/", views.health, name="health"),

    # ── Single laptop detail ───────────────────────────────────────────
    # GET
    path("laptop/<int:laptop_id>/", views.laptop_detail, name="laptop_detail"),

    # ── Admin: hot-reload model without restarting Django ──────────────
    # POST
    path("admin/reload/", views.admin_reload, name="admin_reload"),
]