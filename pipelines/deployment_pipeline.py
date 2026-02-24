#!/usr/bin/env python3
"""
Deployment Pipeline - Deploy trained model to Django API using ZenML
Run: python -m pipelines.deployment_pipeline
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mlflow
from zenml import pipeline, step
from zenml.config import DockerSettings

# Import your existing components
from src.recommendation_engine import LaptopRecommendationEngine
from src.mlflow_tracker import RecommendationExperimentTracker


# =============================================================================
# STEPS
# =============================================================================

@step
def load_trained_model(model_path: Optional[str] = None) -> dict:
    """
    Load the trained model from MLflow or local path
    """
    print("=" * 60)
    print("🚀 DEPLOYMENT PIPELINE - Step 1: Load Model")
    print("=" * 60)
    
    # If no path specified, get from MLflow or production_version.json
    if model_path is None:
        # Try production_version.json first
        prod_file = "models/production_version.json"
        if os.path.exists(prod_file):
            with open(prod_file) as f:
                info = json.load(f)
                model_path = info["model_path"]
                version = info["version"]
                print(f"✓ Found production model: {version}")
        else:
            # Get best run from MLflow
            tracker = RecommendationExperimentTracker("laptop_recommender_v2")
            best_run = tracker.get_best_run(metric="model_size_mb")
            if best_run is not None:
                # Download artifacts
                run_id = best_run.run_id
                artifact_uri = f"runs:/{run_id}/model"
                local_path = mlflow.artifacts.download_artifacts(artifact_uri)
                model_path = os.path.join(local_path, "model")
                version = f"mlflow_{run_id[:8]}"
                print(f"✓ Downloaded from MLflow: {version}")
            else:
                raise ValueError("No model found. Run training first.")
    else:
        version = os.path.basename(model_path).replace("laptop_recommender_", "")
    
    # Validate files exist
    required = [f"{model_path}_index.faiss", f"{model_path}_config.pkl"]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing: {f}")
        print(f"  ✓ {os.path.basename(f)}")
    
    # Test load
    engine = LaptopRecommendationEngine()
    engine.load(model_path)
    print(f"  ✓ Model loaded: {len(engine.df)} laptops")
    
    return {
        "model_path": model_path,
        "version": version,
        "n_laptops": len(engine.df)
    }


@step
def validate_model(model_info: dict) -> dict:
    """
    Validate model before deployment
    """
    print("\n📋 Step 2: Validate Model")
    
    engine = LaptopRecommendationEngine()
    engine.load(model_info["model_path"])
    
    # Run test queries
    test_cases = [
        ("gaming laptop with RTX", "gaming"),
        ("budget laptop for students", "budget"),
        ("professional laptop 16GB RAM", "professional")
    ]
    
    for query, expected in test_cases:
        results = engine.search_by_text(query, n=1)
        if not results:
            raise ValueError(f"No results for query: {query}")
        print(f"  ✓ '{query}' → {results[0]['name'][:40]}...")
    
    # Log validation
    mlflow.log_metric("validation_passed", 1)
    
    return model_info


@step
def package_model(model_info: dict, deploy_dir: str = "deployment/packages") -> dict:
    """
    Package model for deployment
    """
    print("\n📦 Step 3: Package Model")
    
    os.makedirs(deploy_dir, exist_ok=True)
    
    version = model_info["version"]
    package_name = f"model_{version}"
    package_path = os.path.join(deploy_dir, package_name)
    
    # Copy files
    src_path = model_info["model_path"]
    shutil.copy(f"{src_path}_index.faiss", f"{package_path}_index.faiss")
    shutil.copy(f"{src_path}_config.pkl", f"{package_path}_config.pkl")
    
    # Create manifest
    manifest = {
        "version": version,
        "packaged_at": datetime.now().isoformat(),
        "n_laptops": model_info["n_laptops"],
        "files": [
            f"{package_name}_index.faiss",
            f"{package_name}_config.pkl"
        ]
    }
    
    with open(f"{package_path}_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ Packaged: {package_path}")
    
    # Log artifact
    mlflow.log_artifact(f"{package_path}_manifest.json", "deployment")
    
    return {
        **model_info,
        "package_path": package_path,
        "package_dir": deploy_dir
    }


@step
def deploy_to_django(package_info: dict, api_model_dir: str = "api/ml_models") -> dict:
    """
    Deploy packaged model to Django API directory
    """
    print("\n🎯 Step 4: Deploy to Django")
    
    # Create Django model directory
    os.makedirs(api_model_dir, exist_ok=True)
    
    # Copy to Django (as "recommender_latest")
    target_name = "recommender_latest"
    target_path = os.path.join(api_model_dir, target_name)
    
    pkg_path = package_info["package_path"]
    shutil.copy(f"{pkg_path}_index.faiss", f"{target_path}_index.faiss")
    shutil.copy(f"{pkg_path}_config.pkl", f"{target_path}_config.pkl")
    
    # Create version info
    version_info = {
        "version": package_info["version"],
        "deployed_at": datetime.now().isoformat(),
        "model_path": target_path,
        "n_laptops": package_info["n_laptops"]
    }
    
    version_file = os.path.join(api_model_dir, "version.json")
    with open(version_file, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    # Create reload trigger
    trigger_file = os.path.join(api_model_dir, ".reload_trigger")
    with open(trigger_file, 'w') as f:
        f.write(str(datetime.now().timestamp()))
    
    print(f"  ✓ Deployed to: {api_model_dir}/")
    print(f"  ✓ Version: {package_info['version']}")
    print(f"  ✓ Reload trigger created")
    
    # Log deployment
    mlflow.log_param("deploy_version", package_info["version"])
    mlflow.log_param("deploy_target", api_model_dir)
    
    return {
        **package_info,
        "deployed_path": target_path,
        "api_model_dir": api_model_dir
    }


@step
def verify_deployment(deploy_info: dict, api_url: str = "http://localhost:8000") -> dict:
    """
    Verify deployment by calling Django API health endpoint
    """
    print("\n🔍 Step 5: Verify Deployment")
    
    import requests
    
    max_retries = 5
    for i in range(max_retries):
        try:
            # Health check
            response = requests.get(
                f"{api_url}/api/health/",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("model_loaded"):
                    print(f"  ✓ API healthy!")
                    print(f"  ✓ Model version: {data.get('model_version', 'unknown')}")
                    
                    # Test recommendation
                    test_resp = requests.post(
                        f"{api_url}/api/recommend/",
                        json={"type": "text_search", "query": "gaming laptop", "n_recommendations": 1},
                        timeout=10
                    )
                    
                    if test_resp.status_code == 200:
                        print(f"  ✓ Recommendation endpoint working")
                        mlflow.log_metric("deployment_verified", 1)
                        
                        return {
                            **deploy_info,
                            "status": "deployed",
                            "api_url": api_url,
                            "verified_at": datetime.now().isoformat()
                        }
            
            print(f"  ⏳ Attempt {i+1}/{max_retries}...")
            
        except requests.exceptions.ConnectionError:
            print(f"  ⏳ API not reachable (attempt {i+1}/{max_retries})")
            if i == 0:
                print(f"     Tip: Start Django with: python manage.py runserver")
        
        time.sleep(2)
    
    # Verification failed but deployment succeeded
    print("  ⚠️ Verification failed, but model deployed")
    mlflow.log_metric("deployment_verified", 0)
    
    return {
        **deploy_info,
        "status": "deployed_not_verified",
        "api_url": api_url
    }


@step
def update_production_marker(deploy_info: dict) -> str:
    """
    Update production version marker and log to MLflow
    """
    print("\n🚦 Step 6: Update Production Marker")
    
    # Update active version
    api_model_dir = deploy_info["api_model_dir"]
    active_file = os.path.join(api_model_dir, "active_version.txt")
    
    with open(active_file, 'w') as f:
        f.write(deploy_info["version"])
    
    # Log to MLflow Model Registry
    try:
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/deployment",
            "laptop_recommender_production"
        )
    except Exception as e:
        print(f"  ⚠️ Could not register to Model Registry: {e}")
    
    # Create deployment record
    deployment_record = {
        "version": deploy_info["version"],
        "deployed_at": datetime.now().isoformat(),
        "model_path": deploy_info["deployed_path"],
        "api_url": deploy_info.get("api_url"),
        "status": deploy_info.get("status", "deployed")
    }
    
    record_file = f"models/deployment_{deploy_info['version']}.json"
    with open(record_file, 'w') as f:
        json.dump(deployment_record, f, indent=2)
    
    print(f"  ✓ Active version: {deploy_info['version']}")
    print(f"  ✓ Deployment record: {record_file}")
    
    return deploy_info["version"]


# =============================================================================
# PIPELINE
# =============================================================================

docker_settings = DockerSettings(
    requirements=[
        "sentence-transformers",
        "faiss-cpu",
        "mlflow",
        "zenml",
        "requests"
    ]
)

@pipeline(settings={"docker": docker_settings})
def deployment_pipeline(
    model_path: Optional[str] = None,
    api_url: str = "http://localhost:8000",
    api_model_dir: str = "api/ml_models"
):
    """
    ZenML pipeline for deploying laptop recommendation model to Django API
    """
    # Step 1: Load model
    model_info = load_trained_model(model_path)
    
    # Step 2: Validate
    validated = validate_model(model_info)
    
    # Step 3: Package
    packaged = package_model(validated)
    
    # Step 4: Deploy to Django
    deployed = deploy_to_django(packaged, api_model_dir=api_model_dir)
    
    # Step 5: Verify
    verified = verify_deployment(deployed, api_url=api_url)
    
    # Step 6: Update markers
    version = update_production_marker(verified)
    
    return version


# =============================================================================
# CLI RUNNER
# =============================================================================

def run_deployment(
    model_path: Optional[str] = None,
    api_url: str = "http://localhost:8000",
    api_model_dir: str = "api/ml_models"
):
    """Run the deployment pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 STARTING DEPLOYMENT PIPELINE")
    print("=" * 70)
    
    # Run pipeline
    deployed_version = deployment_pipeline(
        model_path=model_path,
        api_url=api_url,
        api_model_dir=api_model_dir
    )
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE!")
    print(f"   Version: {deployed_version}")
    print(f"   API: {api_url}/api/")
    print("=" * 70)
    
    print(f"\n📋 Test commands:")
    print(f"  curl {api_url}/api/health/")
    print(f"  curl -X POST {api_url}/api/recommend/ \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"type\": \"text_search\", \"query\": \"gaming laptop\"}}'")
    
    return deployed_version


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy model to Django API")
    parser.add_argument("--model", "-m", help="Path to trained model")
    parser.add_argument("--api-url", "-u", default="http://localhost:8000", 
                       help="Django API URL")
    parser.add_argument("--api-model-dir", "-d", default="api/ml_models",
                       help="Django model directory")
    
    args = parser.parse_args()
    
    run_deployment(
        model_path=args.model,
        api_url=args.api_url,
        api_model_dir=args.api_model_dir
    )