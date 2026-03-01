# from zenml import pipeline, step
# from zenml.config import DockerSettings
# import mlflow
# import pandas as pd
# import numpy as np
# import os
# import json

# # Import components
# from src.data_ingest import load_laptop_data
# from src.text_encoder import LaptopTextEncoder
# from src.knn_index import LaptopKNNIndex
# from src.recommendation_engine import LaptopRecommendationEngine
# from src.mlflow_tracker import RecommendationExperimentTracker
# from src.feature_store import LaptopFeatureStore
# from src.explainer import RecommendationExplainer
# from src.model_evaluator import evaluate_recommendation_quality

# docker_settings = DockerSettings(
#     requirements=["sentence-transformers", "faiss-cpu", "mlflow", "zenml"]
# )

# @step
# def data_ingestion(file_path: str) -> pd.DataFrame:
#     """Step 1: Ingest raw data"""
#     df = load_laptop_data(file_path)
#     return df

# @step
# def extract_and_version_features(df: pd.DataFrame) -> dict:
#     """Step 2: Extract features and version them"""
#     store = LaptopFeatureStore()
    
#     # Save raw data
#     version, meta = store.save_raw_data(df, source="laptop_data_csv")
    
#     # Create feature profile
#     profile = store.create_feature_profile(df)
    
#     return {
#         "df": df,
#         "version": version,
#         "profile": profile,
#         "store": store
#     }

# @step  
# def encode_with_mlflow(feature_package: dict) -> dict:
#     """Step 3: Encode text with MLflow tracking"""
#     df = feature_package["df"]
#     store = feature_package["store"]
    
#     # Start MLflow tracking
#     tracker = RecommendationExperimentTracker("laptop_recommendations_v2")
#     run = tracker.start_run(tags={"stage": "feature_engineering"})
    
#     # Log dataset info
#     tracker.log_dataset_info(df)
    
#     # Encode
#     encoder = LaptopTextEncoder('all-MiniLM-L6-v2')
#     tracker.log_params({
#         "encoder_model": "all-MiniLM-L6-v2",
#         "embedding_dim": encoder.embedding_dim,
#         "batch_size": 32
#     })
    
#     embeddings = encoder.encode_laptops(df)
    
#     # Save to feature store
#     feature_info = store.save_processed_features(
#         df, embeddings, feature_package["version"],
#         transformation_params={"encoder": "all-MiniLM-L6-v2", "normalized": True}
#     )
    
#     tracker.log_metrics({
#         "embedding_time_seconds": 0,  # Add actual timing
#         "n_embeddings": len(embeddings)
#     })
    
#     tracker.end_run()
    
#     return {
#         "df": df,
#         "embeddings": embeddings,
#         "encoder": encoder,
#         "version": feature_package["version"],
#         "mlflow_run_id": run.info.run_id
#     }

# @step
# def build_index_with_tracking(encoding_package: dict) -> dict:
#     """Step 4: Build KNN index"""
#     df = encoding_package["df"]
#     embeddings = encoding_package["embeddings"]
    
#     tracker = RecommendationExperimentTracker("laptop_recommendations_v2")
#     tracker.start_run(tags={"stage": "model_building"})
    
#     # Build index
#     index = LaptopKNNIndex(embedding_dim=embeddings.shape[1])
    
#     metadata = {}
#     for _, row in df.iterrows():
#         metadata[int(row['laptop_id'])] = {
#             'name': row['name'], 'brand': row['brand'],
#             'price': float(row['price']), 'cpu': row['cpu'],
#             'gpu': row['gpu'], 'ram': int(row['ram_capacity']),
#             'ssd': int(row['ssd']), 'rating': float(row['user_rating']),
#             'usage_type': row.get('usage_type', 'unknown')
#         }
    
#     index.build_index(embeddings, df['laptop_id'].tolist(), metadata)
    
#     tracker.log_params({
#         "index_type": "FAISS-FlatIP",
#         "n_vectors": index.index.ntotal,
#         "embedding_dim": embeddings.shape[1]
#     })
    
#     # Build engine
#     engine = LaptopRecommendationEngine()
#     engine.encoder = encoding_package["encoder"]
#     engine.index = index
#     engine.df = df
    
#     tracker.end_run()
    
#     return {
#         "engine": engine,
#         "index": index,
#         "df": df,
#         "version": encoding_package["version"]
#     }

# @step
# def evaluate_and_explain(model_package: dict) -> dict:
#     """Step 5: Evaluate and create explainer"""
#     engine = model_package["engine"]
#     df = model_package["df"]
    
#     tracker = RecommendationExperimentTracker("laptop_recommendations")
#     tracker.start_run(tags={"stage": "evaluation"})
    
#     # Evaluate
#     test_cases = [
#         {"laptop_id": df.iloc[i]['laptop_id'], "price": df.iloc[i]['price']}
#         for i in range(0, min(50, len(df)), 10)
#     ]
    
#     results = evaluate_recommendation_quality(engine, test_cases)
#     tracker.log_evaluation_results(results)
    
#     # Create explainer
#     explainer = RecommendationExplainer(
#         engine.encoder, engine.index, df
#     )
    
#     # Log sample explanations
#     sample_explanation = explainer.explain_similarity(
#         df.iloc[0]['laptop_id'], 
#         df.iloc[1]['laptop_id']
#     )
#     tracker.log_dict(sample_explanation, "sample_explanation.json")
    
#     # Global feature importance
#     global_imp = explainer.get_global_feature_importance()
#     tracker.log_dict(global_imp, "global_feature_importance.json")
    
#     tracker.end_run()
    
#     return {
#         "engine": engine,
#         "explainer": explainer,
#         # "evaluation_results": results,
#         "version": model_package["version"]
#     }

# @step
# def register_and_deploy(final_package: dict) -> str:
#     """Step 6: Register model and prepare for deployment"""
#     engine = final_package["engine"]
#     version = final_package["version"]
    
#     # Save model artifacts
#     model_path = f"models/laptop_recommender_{version}"
#     os.makedirs("models", exist_ok=True)
#     engine.save(model_path)
    
#     # Register with MLflow
#     tracker = RecommendationExperimentTracker("laptop_recommendations_v2")
#     tracker.start_run(tags={"stage": "deployment"})
#     tracker.register_model("laptop_recommender_v2")
    
#     # Log artifacts
#     tracker.log_model_artifact(f"{model_path}_index.faiss")
#     tracker.log_model_artifact(f"{model_path}_config.pkl")
    
#     tracker.end_run()
    
#     # Create deployment marker
#     deployment_info = {
#         "version": version,
#         "model_path": model_path,
#         "deployed_at": pd.Timestamp.now().isoformat(),
#         "status": "production"
#     }
    
#     with open(f"models/production_version.json", 'w') as f:
#         json.dump(deployment_info, f, indent=2)
    
#     return version

# @pipeline(settings={"docker": docker_settings})
# def complete_recommendation_pipeline(data_path: str):
#     """Complete MLOps pipeline with tracking and explanation"""
    
#     # 1. Data ingestion
#     raw_df = data_ingestion(file_path=data_path)
    
#     # 2. Feature store
#     feature_package = extract_and_version_features(df=raw_df)
    
#     # 3. Encoding with MLflow
#     encoding_package = encode_with_mlflow(feature_package=feature_package)
    
#     # 4. Model building
#     model_package = build_index_with_tracking(encoding_package=encoding_package)
    
#     # 5. Evaluation + Explanation
#     final_package = evaluate_and_explain(model_package=model_package)
    
#     # 6. Deployment
#     deployed_version = register_and_deploy(final_package=final_package)
    
#     return deployed_version

# if __name__ == "__main__":
#     version = complete_recommendation_pipeline(data_path="data/laptop_data.csv")
#     print(f"Pipeline complete. Deployed version: {version}")

"""
pipelines/training_pipeline.py
-------------------------------
Pure-Python training pipeline — no ZenML, no Docker settings.
Each stage is a plain function that passes native Python objects.
Run:  python pipelines/training_pipeline.py
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd

from src.data_ingest        import load_laptop_data
from src.explainer          import RecommendationExplainer
from src.feature_store      import LaptopFeatureStore
from src.knn_index          import LaptopKNNIndex
from src.mlflow_tracker     import RecommendationExperimentTracker
from src.model_evaluator    import evaluate_recommendation_quality
from src.recommendation_engine import LaptopRecommendationEngine
from src.text_encoder       import LaptopTextEncoder


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_ingest(data_path: str) -> pd.DataFrame:
    """Stage 1 — Load and clean raw CSV data."""
    print("\n" + "─" * 50)
    print("📥 STAGE 1 — Data Ingestion")
    print("─" * 50)
    df = load_laptop_data(data_path)
    print(f"   ✔ {len(df)} laptops loaded | columns: {list(df.columns)}")
    return df


def stage_feature_store(df: pd.DataFrame) -> tuple[LaptopFeatureStore, str]:
    """Stage 2 — Version and persist raw features."""
    print("\n" + "─" * 50)
    print("💾 STAGE 2 — Feature Store")
    print("─" * 50)
    store            = LaptopFeatureStore()
    version, meta    = store.save_raw_data(df, source="laptop_csv")
    profile          = store.create_feature_profile(df)
    print(f"   ✔ Version: {version}")
    print(f"   ✔ Missing values: {sum(v for v in meta.items() if isinstance(v, int) and v > 0)}")
    return store, version


def stage_encode(
    df:      pd.DataFrame,
    store:   LaptopFeatureStore,
    version: str,
    tracker: RecommendationExperimentTracker,
) -> tuple[LaptopTextEncoder, np.ndarray]:
    """Stage 3 — Encode laptop descriptions with Sentence Transformer."""
    print("\n" + "─" * 50)
    print("🧠 STAGE 3 — Text Encoding")
    print("─" * 50)

    encoder    = LaptopTextEncoder("all-MiniLM-L6-v2")
    t0         = time.time()
    embeddings = encoder.encode_laptops(df)
    elapsed    = round(time.time() - t0, 2)

    tracker.log_params({
        "encoder_model": "all-MiniLM-L6-v2",
        "embedding_dim": encoder.embedding_dim,
        "batch_size":    32,
    })
    tracker.log_metrics({
        "encoding_seconds": elapsed,
        "n_embeddings":     len(embeddings),
    })
    tracker.log_dataset_info(df)

    # Cache embeddings in feature store
    store.save_processed_features(
        df, embeddings, version,
        transformation_params={
            "encoder": "all-MiniLM-L6-v2",
            "normalized": True,
        },
    )

    print(f"   ✔ Encoded {len(embeddings)} laptops in {elapsed}s")
    print(f"   ✔ Embedding shape: {embeddings.shape}")
    return encoder, embeddings


def stage_build_index(
    df:         pd.DataFrame,
    encoder:    LaptopTextEncoder,
    embeddings: np.ndarray,
    tracker:    RecommendationExperimentTracker,
) -> LaptopRecommendationEngine:
    """Stage 4 — Build FAISS KNN index and wire up the engine."""
    print("\n" + "─" * 50)
    print("🔍 STAGE 4 — KNN Index")
    print("─" * 50)

    metadata = {
        int(row["laptop_id"]): {
            "name":       row["name"],
            "brand":      row["brand"],
            "price":      float(row["price"]),
            "cpu":        row["cpu"],
            "gpu":        row["gpu"],
            "ram":        int(row["ram_capacity"]),
            "ssd":        int(row["ssd"]),
            "rating":     float(row["user_rating"]),
            "usage_type": row.get("usage_type", "unknown"),
            "screen_size":row.get("screen_size", 15),
        }
        for _, row in df.iterrows()
    }

    index = LaptopKNNIndex(embedding_dim=encoder.embedding_dim)
    index.build_index(embeddings, df["laptop_id"].tolist(), metadata)

    tracker.log_params({
        "index_type":  "FAISS-FlatIP",
        "n_vectors":   index.index.ntotal,
        "embedding_dim": encoder.embedding_dim,
    })

    # Wire engine
    engine         = LaptopRecommendationEngine()
    engine.encoder = encoder
    engine.index   = index
    engine.df      = df
    engine._encoder_model = "all-MiniLM-L6-v2"

    print(f"   ✔ KNN index built with {index.index.ntotal} vectors")
    return engine


def stage_evaluate(
    engine:  LaptopRecommendationEngine,
    df:      pd.DataFrame,
    tracker: RecommendationExperimentTracker,
) -> dict:
    """Stage 5 — Evaluate quality + generate explainer output."""
    print("\n" + "─" * 50)
    print("📊 STAGE 5 — Evaluation & Explanation")
    print("─" * 50)

    # ── Quantitative evaluation ─────────────────────────────────────────
    test_cases = [
        {"laptop_id": df.iloc[i]["laptop_id"], "price": df.iloc[i]["price"]}
        for i in range(0, min(50, len(df)), max(1, len(df) // 10))
    ]
    results = evaluate_recommendation_quality(engine, test_cases, n=5)
    tracker.log_evaluation_results(results)

    # ── Explainer sample ────────────────────────────────────────────────
    explainer = RecommendationExplainer(engine.encoder, engine.index, df)

    sample_exp = explainer.explain_similarity(
        int(df.iloc[0]["laptop_id"]),
        int(df.iloc[1]["laptop_id"]),
    )
    tracker.log_dict(sample_exp, "sample_explanation.json")       # FIX: method now exists

    global_imp = explainer.get_global_feature_importance()
    tracker.log_dict(global_imp, "global_feature_importance.json")# FIX: method now exists

    print(f"   ✔ mean_similarity:     {results['mean_similarity']}")
    print(f"   ✔ mean_diversity:      {results['mean_diversity']}")
    print(f"   ✔ mean_price_coverage: {results['mean_price_coverage']}")
    return results


def stage_save(
    engine:  LaptopRecommendationEngine,
    version: str,
    tracker: RecommendationExperimentTracker,
) -> str:
    """Stage 6 — Save model artifacts and write production marker."""
    print("\n" + "─" * 50)
    print("💾 STAGE 6 — Save & Register")
    print("─" * 50)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/laptop_recommender_{version}"

    # Save FIRST — then measure size (fixes main.py bug)
    engine.save(model_path)

    faiss_size = os.path.getsize(f"{model_path}_index.faiss") / 1024 ** 2
    tracker.log_metrics({"model_size_mb": round(faiss_size, 3)})
    tracker.log_model_artifact(f"{model_path}_index.faiss")
    tracker.log_model_artifact(f"{model_path}_config.pkl")

    # Write production marker
    from datetime import datetime
    marker = {
        "version":    version,
        "model_path": model_path,
        "n_laptops":  len(engine.df),
        "trained_at": datetime.now().isoformat(),
        "status":     "production",
    }
    with open("models/production_version.json", "w") as f:
        json.dump(marker, f, indent=2)

    print(f"   ✔ Model saved  → {model_path}_index.faiss ({faiss_size:.2f} MB)")
    print(f"   ✔ Config saved → {model_path}_config.pkl")
    return model_path


# ---------------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(data_path: str = "data/laptop_data.csv") -> tuple[str, str]:
    """
    Run all six training stages end-to-end.

    Returns
    -------
    (version, model_path)
    """
    print("\n" + "=" * 60)
    print("   LAPTOP RECOMMENDATION SYSTEM — TRAINING PIPELINE")
    print("=" * 60)

    # ── Stage 1: Ingest ────────────────────────────────────────────────
    df = stage_ingest(data_path)

    # ── Stage 2: Feature store ─────────────────────────────────────────
    store, version = stage_feature_store(df)

    # ── MLflow run wraps stages 3-6 ────────────────────────────────────
    tracker = RecommendationExperimentTracker("laptop_recommender")
    run     = tracker.start_run(tags={"version": version, "stage": "full_pipeline"})
    tracker.log_params({"data_version": version, "n_laptops": len(df)})

    try:
        # ── Stage 3: Encode ────────────────────────────────────────────
        encoder, embeddings = stage_encode(df, store, version, tracker)

        # ── Stage 4: KNN index ─────────────────────────────────────────
        engine = stage_build_index(df, encoder, embeddings, tracker)

        # ── Stage 5: Evaluate ──────────────────────────────────────────
        _eval_results = stage_evaluate(engine, df, tracker)

        # ── Stage 6: Save ──────────────────────────────────────────────
        model_path = stage_save(engine, version, tracker)

    finally:
        tracker.end_run()

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Version:    {version}")
    print(f"   Model path: {model_path}")
    print(f"   MLflow run: {run.info.run_id}")
    print("=" * 60)

    # Quick smoke test
    print("\n🧪 Smoke test …")
    test_queries = [
        "gaming laptop with RTX 4060",
        "budget laptop for students under 40000",
        "professional laptop with 16GB RAM and good display",
    ]
    for q in test_queries:
        hits = engine.search_by_text(q, n=2)
        for h in hits:
            print(f"   [{q[:30]}…] → {h['name'][:45]} ₹{h['price']:,.0f}  sim={h['similarity_score']}")

    return version, model_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/laptop_data.csv"

    if not os.path.exists(data_path):
        print(f"❌  Data file not found: {data_path}")
        print("    Usage: python pipelines/training_pipeline.py [path/to/laptop_data.csv]")
        sys.exit(1)

    version, model_path = run_training_pipeline(data_path)
    print(f"\n🚀 Next: python manage.py runserver")