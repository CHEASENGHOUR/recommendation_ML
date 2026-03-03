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

def stage_ingest(data_path: str) -> pd.DataFrame:
    """Stage 1 — Load and clean raw CSV data."""
    print("\n" + "─" * 50)
    print("STAGE 1 — Data Ingestion")
    print("─" * 50)
    df = load_laptop_data(data_path)
    print(f"   ✔ {len(df)} laptops loaded | columns: {list(df.columns)}")
    return df


def stage_feature_store(df: pd.DataFrame) -> tuple[LaptopFeatureStore, str]:
    """Stage 2 — Version and persist raw features."""
    print("\n" + "─" * 50)
    print("STAGE 2 — Feature Store")
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
    print("STAGE 3 — Text Encoding")
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
    print("STAGE 4 — KNN Index")
    print("─" * 50)

    metadata = {
        int(row["laptop_id"]): {
            "name":       row["name"],
            "brand":      row["brand"],
            # "price":      float(row["price"]),
            "price": float(row["price_usd"]),
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

    print(f" ✔ KNN index built with {index.index.ntotal} vectors")
    return engine


def stage_evaluate(
    engine:  LaptopRecommendationEngine,
    df:      pd.DataFrame,
    tracker: RecommendationExperimentTracker,
) -> dict:
    """Stage 5 — Evaluate quality + generate explainer output."""
    print("\n" + "─" * 50)
    print("STAGE 5 — Evaluation & Explanation")
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
    print("STAGE 6 — Save & Register")
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

def run_training_pipeline(data_path: str = "data/laptop_data.csv") -> tuple[str, str]:

    print("\n" + "=" * 60)
    print("   LAPTOP RECOMMENDATION SYSTEM — TRAINING PIPELINE")
    print("=" * 60)

    # ── Stage 1: Ingest ────────────────────────────────────────────────
    df = stage_ingest(data_path)
    INR_TO_USD = 0.012  # Update periodically if needed

    if "price_usd" not in df.columns:
        df["price_usd"] = df["price"] * INR_TO_USD
    
    print("   ✔ Added price_usd column")

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
    print("TRAINING COMPLETE!")
    print(f"   Version:    {version}")
    print(f"   Model path: {model_path}")
    print(f"   MLflow run: {run.info.run_id}")
    print("=" * 60)

    # Quick smoke test
    print("\n Smoke test …")
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

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/laptop_data.csv"

    if not os.path.exists(data_path):
        print(f" Data file not found: {data_path}")
        print("  Usage: python pipelines/training_pipeline.py [path/to/laptop_data.csv]")
        sys.exit(1)

    version, model_path = run_training_pipeline(data_path)
    print(f"\n Next: python manage.py runserver")