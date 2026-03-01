"""
api/views.py
------------
Django API views for the Laptop Recommendation System.

Project layout
--------------
rec_system/
  api/views.py          ← this file
  config/settings.py    ← BASE_DIR = rec_system/
  models/               ← trained model artifacts (production_version.json etc.)
  src/                  ← ML source code

Model loading
-------------
Reads rec_system/models/production_version.json which is written
automatically by pipelines/training_pipeline.py after training.
No manual configuration needed — just run `python main.py` to train,
then `python manage.py runserver` to serve.

Endpoints
---------
POST /api/recommend/            — text search, item-to-item, or preferences
GET  /api/health/               — model status + version
GET  /api/laptop/<id>/          — single laptop detail
POST /api/admin/reload/         — hot-reload model after retraining
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

# ---------------------------------------------------------------------------
# Make sure rec_system/ (project root) is on sys.path so `src.*` imports work.
# BASE_DIR in settings.py is rec_system/ (where manage.py lives).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(settings.BASE_DIR).parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.recommendation_engine import LaptopRecommendationEngine   # noqa: E402


# =============================================================================
# MODEL MANAGER  — thread-safe singleton
# =============================================================================

class ModelManager:
    """
    Thread-safe singleton that loads the trained model once and keeps it
    in memory.  Supports hot-reload via a trigger file.

    Model location
    --------------
    The training pipeline writes:
        rec_system/models/production_version.json
            {
                "version":    "20260301_161819_2e2a7e22",
                "model_path": "models/laptop_recommender_20260301_161819_2e2a7e22",
                ...
            }

    model_path is relative to PROJECT_ROOT (rec_system/).
    """

    _engine:  LaptopRecommendationEngine | None = None
    _version: str | None = None
    _lock:    threading.Lock = threading.Lock()

    # Paths — all relative to rec_system/ (settings.BASE_DIR)
    _PRODUCTION_JSON  = "models/production_version.json"
    _RELOAD_TRIGGER   = "models/.reload_trigger"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_engine(cls) -> LaptopRecommendationEngine | None:
        """Return the active engine, loading or reloading as needed."""
        trigger = os.path.join(_PROJECT_ROOT, cls._RELOAD_TRIGGER)

        if os.path.exists(trigger):
            print("[ModelManager] 🔄 Reload trigger detected — reloading …")
            with cls._lock:
                cls._engine = None
            try:
                os.remove(trigger)
            except OSError:
                pass

        if cls._engine is None:
            with cls._lock:
                if cls._engine is None:      # double-checked locking
                    cls._load()

        return cls._engine

    @classmethod
    def force_reload(cls) -> LaptopRecommendationEngine | None:
        """Force an immediate reload from disk."""
        with cls._lock:
            cls._engine = None
        return cls.get_engine()

    @classmethod
    def get_version(cls) -> str | None:
        return cls._version

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    @classmethod
    def _load(cls) -> None:
        """
        Read production_version.json and load the model artifacts.
        model_path in the JSON is relative to PROJECT_ROOT.
        """
        version_file = os.path.join(_PROJECT_ROOT, cls._PRODUCTION_JSON)
        print("DEBUG _PROJECT_ROOT:", _PROJECT_ROOT)
        
        print("DEBUG version_file:", version_file)
        print("DEBUG exists?:", os.path.exists(version_file))

        if not os.path.exists(version_file):
            print(f"[ModelManager] ⚠  {cls._PRODUCTION_JSON} not found.")
            print("[ModelManager]    Run `python main.py` to train first.")
            return

        with open(version_file) as f:
            info = json.load(f)

        # model_path may be stored as relative ("models/laptop_recommender_…")
        # or absolute.  Normalise to absolute.
        raw_path = info.get("model_path", "")
        if os.path.isabs(raw_path):
            model_path = raw_path
        else:
            model_path = os.path.join(_PROJECT_ROOT, raw_path)

        faiss_file = f"{model_path}_index.faiss"
        if not os.path.exists(faiss_file):
            print(f"[ModelManager] ❌ FAISS file not found: {faiss_file}")
            return

        version = info.get("version", "unknown")
        print(f"[ModelManager] 📦 Loading model version: {version} …")

        engine = LaptopRecommendationEngine()
        engine.load(model_path)

        cls._engine  = engine
        cls._version = version
        n = engine.index.index.ntotal if engine.index else 0
        print(f"[ModelManager] ✅ Model ready — {n} laptops | version={version}")


# =============================================================================
# HELPERS
# =============================================================================

def _ok(data: dict, status: int = 200) -> JsonResponse:
    return JsonResponse({"success": True, **data}, status=status)


def _err(msg: str, detail: str = "", status: int = 400) -> JsonResponse:
    body = {"success": False, "error": msg}
    if detail:
        body["detail"] = detail
    return JsonResponse(body, status=status)


def _no_model() -> JsonResponse:
    return _err(
        "Model not loaded",
        "Run `python main.py` to train, then restart Django.",
        status=503,
    )


# =============================================================================
# VIEWS
# =============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class RecommendView(View):
    """
    POST /api/recommend/

    Single unified endpoint — selects recommendation strategy via `type`.

    Request body (JSON)
    -------------------
    {
        "type": "text_search",          // required
        "query": "gaming laptop RTX",   // for text_search
        "laptop_id": 42,                // for similar
        "preferences": {                // for preferences
            "usage_type": "gaming",
            "max_price": 80000,
            "min_price": 30000,
            "preferred_brand": "ASUS",
            "min_ram": 16
        },
        "filters": {                    // optional for any type
            "min_price": 20000,
            "max_price": 100000,
            "usage": "gaming"
        },
        "n_recommendations": 5
    }

    Supported types
    ---------------
    text_search   Free-text semantic query
    similar       Item-to-item similarity by laptop_id
    preferences   Structured preference filters
    personalized  History-based (pass "history": [id1, id2, …])
    """

    def post(self, request):
        start = time.time()

        try:
            data = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return _err("Invalid JSON body.")

        engine = ModelManager.get_engine()
        if engine is None:
            return _no_model()

        rec_type = data.get("type", "text_search")
        n        = max(1, min(int(data.get("n_recommendations", 5)), 50))

        # Optional filters shared across types
        filters   = data.get("filters", {})
        min_p     = filters.get("min_price") or data.get("min_price")
        max_p     = filters.get("max_price") or data.get("max_price")
        usage     = filters.get("usage")     or data.get("usage_filter")
        price_rng = (float(min_p or 0), float(max_p)) if max_p else None

        try:
            # ── text search ────────────────────────────────────────────
            if rec_type == "text_search":
                query = data.get("query", "").strip()
                if not query:
                    return _err("'query' is required for text_search.")
                results = engine.search_by_text(
                    query,
                    n=n,
                    price_range=price_rng,
                    usage_filter=usage,
                )

            # ── item-to-item ───────────────────────────────────────────
            elif rec_type == "similar":
                laptop_id = data.get("laptop_id")
                if laptop_id is None:
                    return _err("'laptop_id' is required for type=similar.")
                results = engine.get_similar_laptops(
                    int(laptop_id),
                    n=n,
                    price_range=price_rng,
                )

            # ── preference-based ───────────────────────────────────────
            elif rec_type == "preferences":
                prefs = data.get("preferences", {})
                results = engine.get_recommendations_by_preferences(
                    usage_type=prefs.get("usage_type") or usage,
                    max_price=float(prefs["max_price"]) if prefs.get("max_price") else (float(max_p) if max_p else None),
                    min_price=float(prefs["min_price"]) if prefs.get("min_price") else (float(min_p) if min_p else None),
                    preferred_brand=prefs.get("preferred_brand"),
                    min_ram=int(prefs["min_ram"]) if prefs.get("min_ram") else None,
                    n=n,
                )

            # ── personalised ───────────────────────────────────────────
            elif rec_type == "personalized":
                history = data.get("history", [])
                if not isinstance(history, list):
                    return _err("'history' must be a list of laptop IDs.")
                results = engine.get_personalized_recommendations(
                    user_history=[int(i) for i in history],
                    n=n,
                )

            else:
                return _err(
                    f"Unknown type '{rec_type}'.",
                    "Valid types: text_search, similar, preferences, personalized",
                )

        except Exception as exc:
            return _err("Recommendation failed.", detail=str(exc), status=500)

        elapsed_ms = round((time.time() - start) * 1000, 2)

        return _ok({
            "type":            rec_type,
            "count":           len(results),
            "recommendations": results,
            "model_version":   ModelManager.get_version(),
            "response_time_ms":elapsed_ms,
        })


# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class HealthView(View):
    """GET /api/health/"""

    def get(self, request):
        engine = ModelManager.get_engine()
        n      = engine.index.index.ntotal if (engine and engine.index) else 0

        body = {
            "status":         "healthy" if engine else "unhealthy",
            "model_loaded":   engine is not None,
            "model_version":  ModelManager.get_version(),
            "total_laptops":  n,
            "timestamp":      time.time(),
        }
        return JsonResponse(body, status=200 if engine else 503)


# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class LaptopDetailView(View):
    """GET /api/laptop/<laptop_id>/"""

    def get(self, request, laptop_id: int):
        engine = ModelManager.get_engine()
        if engine is None:
            return _no_model()

        mask = engine.df["laptop_id"] == int(laptop_id)
        if not mask.any():
            return _err(f"Laptop {laptop_id} not found.", status=404)

        row = engine.df[mask].iloc[0]

        return _ok({
            "laptop": {
                "laptop_id":    int(row["laptop_id"]),
                "name":         str(row["name"]),
                "brand":        str(row["brand"]),
                "price":        float(row["price"]),
                "cpu":          str(row["cpu"]),
                "gpu":          str(row["gpu"]),
                "ram_capacity": int(row["ram_capacity"]),
                "ssd":          int(row["ssd"]),
                "screen_size":  row["screen_size"],
                "user_rating":  float(row["user_rating"]),
                "usage_type":   str(row.get("usage_type", "unknown")),
                "gpu_type":     str(row.get("gpu_type",   "unknown")),
                "cpu_tier":     str(row.get("cpu_tier",   "unknown")),
            }
        })


# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class AdminReloadView(View):
    """
    POST /api/admin/reload/
    Immediately reloads the model from disk.
    Call this after running `python main.py` to retrain without
    restarting the Django server.
    """

    def post(self, request):
        try:
            engine = ModelManager.force_reload()
            if engine is None:
                return _err("Reload failed — no trained model found.", status=503)

            return _ok({
                "message":       "Model reloaded successfully.",
                "model_version": ModelManager.get_version(),
                "total_laptops": engine.index.index.ntotal,
            })
        except Exception as exc:
            return _err("Reload error.", detail=str(exc), status=500)


# =============================================================================
# URL HANDLER ALIASES  (used in urls.py)
# =============================================================================

recommend    = RecommendView.as_view()
health       = HealthView.as_view()
laptop_detail = LaptopDetailView.as_view()
admin_reload = AdminReloadView.as_view()