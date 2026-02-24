#!/usr/bin/env python3
"""
Main training script for Laptop Recommendation System
Run: python train.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_ingest import load_laptop_data
from src.text_encoder import LaptopTextEncoder
from src.knn_index import LaptopKNNIndex
from src.recommendation_engine import LaptopRecommendationEngine
from src.feature_store import LaptopFeatureStore
from src.mlflow_tracker import RecommendationExperimentTracker

def train_model(data_path: str = "data/laptop_data.csv"):
    """Complete training pipeline"""
    
    print("=" * 60)
    print("LAPTOP RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n📊 Step 1: Loading data...")
    df = load_laptop_data(data_path)
    print(f"   Loaded {len(df)} laptops")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Feature Store
    print("\n💾 Step 2: Saving to feature store...")
    store = LaptopFeatureStore()
    version, _ = store.save_raw_data(df, source="laptop_csv")
    
    # Step 3: MLflow Tracking
    print("\n📈 Step 3: Starting MLflow tracking...")
    tracker = RecommendationExperimentTracker("laptop_recommender_v2")
    run = tracker.start_run(tags={"version": version, "stage": "training"})
    
    tracker.log_params({
        "data_version": version,
        "n_laptops": len(df),
        "encoder_model": "all-MiniLM-L6-v2"
    })
    
    # Step 4: Build Engine
    print("\n🔧 Step 4: Building recommendation engine...")
    engine = LaptopRecommendationEngine()
    engine.fit(df)
    
    # Step 5: Test
    print("\n🧪 Step 5: Testing recommendations...")
    test_queries = [
        "gaming laptop with RTX",
        "budget laptop for students",
        "professional laptop with 16GB RAM"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = engine.search_by_text(query, n=2)
        for r in results:
            print(f"   → {r['name'][:50]}... (₹{r['price']:,.0f}, score: {r['similarity_score']})")
    
    # Step 6: Save Model
    print("\n💾 Step 6: Saving model...")
    model_path = f"models/laptop_recommender_{version}"
    os.makedirs("models", exist_ok=True)
    engine.save(model_path)
    
    # Save production marker
    with open("models/production_version.json", 'w') as f:
        json.dump({
            "version": version,
            "model_path": model_path,
            "n_laptops": len(df),
            "trained_at": datetime.now().isoformat()
        }, f, indent=2)
    
    # Log to MLflow
    tracker.log_metrics({
        "model_size_mb": os.path.getsize(f"{model_path}_index.faiss") / 1024**2
    })
    tracker.end_run()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Version: {version}")
    print(f"   Model: {model_path}")
    print(f"   MLflow Run: {run.info.run_id}")
    print("=" * 60)
    
    return version, model_path

if __name__ == "__main__":
    from datetime import datetime
    
    # Check if data exists
    if not os.path.exists("data/laptop_data.csv"):
        print("❌ Error: data/laptop_data.csv not found!")
        print("   Please copy your laptop_data.csv to the data/ folder")
        sys.exit(1)
    
    # Run training
    version, model_path = train_model()
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Test API: python test_api.py")
    print(f"   2. Start server: python manage.py runserver")