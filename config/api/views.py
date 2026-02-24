import os
import json
import sys
import time
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings

# Add project root to path for src imports
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, PROJECT_ROOT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
    
# BASE_DIR = Path(__file__).resolve().parent.parent
# PROJECT_ROOT = BASE_DIR.parent  # Go up one more level to rec_system

# # Add src to Python path
# sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation_engine import LaptopRecommendationEngine


# =============================================================================
# SINGLETON MODEL INSTANCE
# =============================================================================

class ModelManager:
    """
    Manages the recommendation model singleton
    Loads from api/ml_models/ directory
    """
    _instance = None
    _version = None
    _last_check = 0
    
    @classmethod
    def get_engine(cls):
        """Get or reload model if trigger exists"""
        model_dir = os.path.join(settings.BASE_DIR, 'api', 'ml_models')
        
        # Check for reload trigger (created by deployment pipeline)
        trigger_file = os.path.join(model_dir, '.reload_trigger')
        if os.path.exists(trigger_file):
            print("🔄 Reload trigger detected, reloading model...")
            cls._instance = None
            os.remove(trigger_file)
        
        # Load if not exists
        if cls._instance is None:
            cls._load_model(model_dir)
        
        return cls._instance
    
    @classmethod
    def _load_model(cls, model_dir):
        """Load model from Django's ml_models directory"""
        version_file = os.path.join(model_dir, 'version.json')
        
        if not os.path.exists(version_file):
            print("⚠️ No version.json found")
            return None
        
        # Read version info
        with open(version_file) as f:
            version_info = json.load(f)
        
        model_path = version_info.get('model_path', os.path.join(model_dir, 'recommender_latest'))
        
        # Check if model files exist
        if not os.path.exists(f"{model_path}_index.faiss"):
            print(f"❌ Model files not found at: {model_path}")
            return None
        
        # Load engine
        print(f"📦 Loading model: {version_info.get('version', 'unknown')}")
        cls._instance = LaptopRecommendationEngine()
        cls._instance.load(model_path)
        cls._version = version_info.get('version')
        
        print(f"✅ Model loaded: {cls._version} ({len(cls._instance.df)} laptops)")
        return cls._instance
    
    @classmethod
    def get_version(cls):
        return cls._version


# =============================================================================
# API ENDPOINTS
# =============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class RecommendView(View):
    """
    POST /api/recommend/
    
    Request:
    {
        "type": "text_search" | "similar" | "preferences",
        "query": "gaming laptop with RTX",           # for text_search
        "laptop_id": 5,                               # for similar
        "preferences": {                              # for preferences
            "usage_type": "gaming",
            "max_price": 80000,
            "min_ram": 16
        },
        "n_recommendations": 5
    }
    """
    
    def post(self, request):
        start_time = time.time()
        
        # Parse request
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        
        # Get model
        engine = ModelManager.get_engine()
        if engine is None:
            return JsonResponse({
                "error": "Model not loaded",
                "message": "Run: python -m pipelines.deployment_pipeline"
            }, status=503)
        
        # Process request
        rec_type = data.get('type', 'text_search')
        n = data.get('n_recommendations', 5)
        
        try:
            if rec_type == 'text_search':
                results = engine.search_by_text(
                    query=data.get('query', ''),
                    n=n
                )
                
            elif rec_type == 'similar':
                laptop_id = data.get('laptop_id')
                if laptop_id is None:
                    return JsonResponse({"error": "laptop_id required"}, status=400)
                results = engine.get_similar_laptops(laptop_id=laptop_id, n=n)
                
            elif rec_type == 'preferences':
                prefs = data.get('preferences', {})
                results = engine.get_recommendations_by_preferences(
                    usage_type=prefs.get('usage_type'),
                    max_price=prefs.get('max_price'),
                    min_ram=prefs.get('min_ram'),
                    preferred_brand=prefs.get('preferred_brand'),
                    n=n
                )
            else:
                return JsonResponse({"error": f"Unknown type: {rec_type}"}, status=400)
            
            response_time = (time.time() - start_time) * 1000
            
            return JsonResponse({
                "success": True,
                "type": rec_type,
                "count": len(results),
                "recommendations": results,
                "model_version": ModelManager.get_version(),
                "response_time_ms": round(response_time, 2)
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class HealthView(View):
    """
    GET /api/health/
    
    Returns model status and version
    """
    
    def get(self, request):
        engine = ModelManager.get_engine()
        
        # Get active version
        active_version = None
        active_file = os.path.join(settings.BASE_DIR, 'api', 'ml_models', 'active_version.txt')
        if os.path.exists(active_file):
            with open(active_file) as f:
                active_version = f.read().strip()
        
        health_data = {
            "status": "healthy" if engine else "unhealthy",
            "model_loaded": engine is not None,
            "model_version": ModelManager.get_version() or active_version,
            "total_laptops": len(engine.df) if engine else 0,
            "timestamp": time.time()
        }
        
        status_code = 200 if engine else 503
        return JsonResponse(health_data, status=status_code)


@method_decorator(csrf_exempt, name='dispatch')
class LaptopDetailView(View):
    """
    GET /api/laptop/<laptop_id>/
    
    Get details of a specific laptop
    """
    
    def get(self, request, laptop_id):
        engine = ModelManager.get_engine()
        if engine is None:
            return JsonResponse({"error": "Model not loaded"}, status=503)
        
        laptop = engine.df[engine.df['laptop_id'] == int(laptop_id)]
        if laptop.empty:
            return JsonResponse({"error": "Laptop not found"}, status=404)
        
        row = laptop.iloc[0]
        return JsonResponse({
            "laptop_id": int(row['laptop_id']),
            "name": row['name'],
            "brand": row['brand'],
            "price": float(row['price']),
            "cpu": row['cpu'],
            "gpu": row['gpu'],
            "ram_capacity": int(row['ram_capacity']),
            "ssd": int(row['ssd']),
            "screen_size": row['screen_size'],
            "user_rating": float(row['user_rating']),
            "usage_type": row.get('usage_type', 'unknown')
        })


@method_decorator(csrf_exempt, name='dispatch')
class AdminReloadView(View):
    """
    POST /api/admin/reload/
    
    Force model reload (called by deployment pipeline)
    """
    
    def post(self, request):
        # Create reload trigger
        trigger_file = os.path.join(settings.BASE_DIR, 'api', 'ml_models', '.reload_trigger')
        with open(trigger_file, 'w') as f:
            f.write(str(time.time()))
        
        return JsonResponse({
            "success": True,
            "message": "Model reload triggered"
        })


# =============================================================================
# URL HANDLERS
# =============================================================================

recommend = RecommendView.as_view()
health = HealthView.as_view()
laptop_detail = LaptopDetailView.as_view()
admin_reload = AdminReloadView.as_view()