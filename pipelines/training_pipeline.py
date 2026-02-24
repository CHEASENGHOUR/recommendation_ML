from zenml import pipeline, step
from zenml.config import DockerSettings
import mlflow
import pandas as pd
import numpy as np
import os
import json

# Import components
from src.data_ingest import load_laptop_data
from src.text_encoder import LaptopTextEncoder
from src.knn_index import LaptopKNNIndex
from src.recommendation_engine import LaptopRecommendationEngine
from src.mlflow_tracker import RecommendationExperimentTracker
from src.feature_store import LaptopFeatureStore
from src.explainer import RecommendationExplainer
from src.model_evaluator import evaluate_recommendation_quality

docker_settings = DockerSettings(
    requirements=["sentence-transformers", "faiss-cpu", "mlflow", "zenml"]
)

@step
def data_ingestion(file_path: str) -> pd.DataFrame:
    """Step 1: Ingest raw data"""
    df = load_laptop_data(file_path)
    return df

@step
def extract_and_version_features(df: pd.DataFrame) -> dict:
    """Step 2: Extract features and version them"""
    store = LaptopFeatureStore()
    
    # Save raw data
    version, meta = store.save_raw_data(df, source="laptop_data_csv")
    
    # Create feature profile
    profile = store.create_feature_profile(df)
    
    return {
        "df": df,
        "version": version,
        "profile": profile,
        "store": store
    }

@step  
def encode_with_mlflow(feature_package: dict) -> dict:
    """Step 3: Encode text with MLflow tracking"""
    df = feature_package["df"]
    store = feature_package["store"]
    
    # Start MLflow tracking
    tracker = RecommendationExperimentTracker("laptop_recommendations_v2")
    run = tracker.start_run(tags={"stage": "feature_engineering"})
    
    # Log dataset info
    tracker.log_dataset_info(df)
    
    # Encode
    encoder = LaptopTextEncoder('all-MiniLM-L6-v2')
    tracker.log_params({
        "encoder_model": "all-MiniLM-L6-v2",
        "embedding_dim": encoder.embedding_dim,
        "batch_size": 32
    })
    
    embeddings = encoder.encode_laptops(df)
    
    # Save to feature store
    feature_info = store.save_processed_features(
        df, embeddings, feature_package["version"],
        transformation_params={"encoder": "all-MiniLM-L6-v2", "normalized": True}
    )
    
    tracker.log_metrics({
        "embedding_time_seconds": 0,  # Add actual timing
        "n_embeddings": len(embeddings)
    })
    
    tracker.end_run()
    
    return {
        "df": df,
        "embeddings": embeddings,
        "encoder": encoder,
        "version": feature_package["version"],
        "mlflow_run_id": run.info.run_id
    }

@step
def build_index_with_tracking(encoding_package: dict) -> dict:
    """Step 4: Build KNN index"""
    df = encoding_package["df"]
    embeddings = encoding_package["embeddings"]
    
    tracker = RecommendationExperimentTracker("laptop_recommendations_v2")
    tracker.start_run(tags={"stage": "model_building"})
    
    # Build index
    index = LaptopKNNIndex(embedding_dim=embeddings.shape[1])
    
    metadata = {}
    for _, row in df.iterrows():
        metadata[int(row['laptop_id'])] = {
            'name': row['name'], 'brand': row['brand'],
            'price': float(row['price']), 'cpu': row['cpu'],
            'gpu': row['gpu'], 'ram': int(row['ram_capacity']),
            'ssd': int(row['ssd']), 'rating': float(row['user_rating']),
            'usage_type': row.get('usage_type', 'unknown')
        }
    
    index.build_index(embeddings, df['laptop_id'].tolist(), metadata)
    
    tracker.log_params({
        "index_type": "FAISS-FlatIP",
        "n_vectors": index.index.ntotal,
        "embedding_dim": embeddings.shape[1]
    })
    
    # Build engine
    engine = LaptopRecommendationEngine()
    engine.encoder = encoding_package["encoder"]
    engine.index = index
    engine.df = df
    
    tracker.end_run()
    
    return {
        "engine": engine,
        "index": index,
        "df": df,
        "version": encoding_package["version"]
    }

@step
def evaluate_and_explain(model_package: dict) -> dict:
    """Step 5: Evaluate and create explainer"""
    engine = model_package["engine"]
    df = model_package["df"]
    
    tracker = RecommendationExperimentTracker("laptop_recommendations")
    tracker.start_run(tags={"stage": "evaluation"})
    
    # Evaluate
    test_cases = [
        {"laptop_id": df.iloc[i]['laptop_id'], "price": df.iloc[i]['price']}
        for i in range(0, min(50, len(df)), 10)
    ]
    
    results = evaluate_recommendation_quality(engine, test_cases)
    tracker.log_evaluation_results(results)
    
    # Create explainer
    explainer = RecommendationExplainer(
        engine.encoder, engine.index, df
    )
    
    # Log sample explanations
    sample_explanation = explainer.explain_similarity(
        df.iloc[0]['laptop_id'], 
        df.iloc[1]['laptop_id']
    )
    tracker.log_dict(sample_explanation, "sample_explanation.json")
    
    # Global feature importance
    global_imp = explainer.get_global_feature_importance()
    tracker.log_dict(global_imp, "global_feature_importance.json")
    
    tracker.end_run()
    
    return {
        "engine": engine,
        "explainer": explainer,
        # "evaluation_results": results,
        "version": model_package["version"]
    }

@step
def register_and_deploy(final_package: dict) -> str:
    """Step 6: Register model and prepare for deployment"""
    engine = final_package["engine"]
    version = final_package["version"]
    
    # Save model artifacts
    model_path = f"models/laptop_recommender_{version}"
    os.makedirs("models", exist_ok=True)
    engine.save(model_path)
    
    # Register with MLflow
    tracker = RecommendationExperimentTracker("laptop_recommendations_v2")
    tracker.start_run(tags={"stage": "deployment"})
    tracker.register_model("laptop_recommender_v2")
    
    # Log artifacts
    tracker.log_model_artifact(f"{model_path}_index.faiss")
    tracker.log_model_artifact(f"{model_path}_config.pkl")
    
    tracker.end_run()
    
    # Create deployment marker
    deployment_info = {
        "version": version,
        "model_path": model_path,
        "deployed_at": pd.Timestamp.now().isoformat(),
        "status": "production"
    }
    
    with open(f"models/production_version.json", 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    return version

@pipeline(settings={"docker": docker_settings})
def complete_recommendation_pipeline(data_path: str):
    """Complete MLOps pipeline with tracking and explanation"""
    
    # 1. Data ingestion
    raw_df = data_ingestion(file_path=data_path)
    
    # 2. Feature store
    feature_package = extract_and_version_features(df=raw_df)
    
    # 3. Encoding with MLflow
    encoding_package = encode_with_mlflow(feature_package=feature_package)
    
    # 4. Model building
    model_package = build_index_with_tracking(encoding_package=encoding_package)
    
    # 5. Evaluation + Explanation
    final_package = evaluate_and_explain(model_package=model_package)
    
    # 6. Deployment
    deployed_version = register_and_deploy(final_package=final_package)
    
    return deployed_version

if __name__ == "__main__":
    version = complete_recommendation_pipeline(data_path="data/laptop_data.csv")
    print(f"Pipeline complete. Deployed version: {version}")