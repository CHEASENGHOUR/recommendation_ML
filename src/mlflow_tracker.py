import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional
import json

class RecommendationExperimentTracker:
    def __init__(self, experiment_name: str = "laptop_recommendations"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.active_run = None
        
    def start_run(self, run_name: Optional[str] = None, tags: Dict = None):
        """Start MLflow run"""
        if run_name is None:
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.active_run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
            
        # Log system info
        mlflow.set_tag("python_version", "3.9")
        mlflow.set_tag("framework", "sentence-transformers")
        
        return self.active_run
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model_artifact(self, model_path: str, artifact_path: str = "model"):
        """Log model artifact"""
        mlflow.log_artifact(model_path, artifact_path)
    
    def log_dataset_info(self, df, dataset_name: str = "training_data"):
        """Log dataset statistics"""
        mlflow.log_param(f"{dataset_name}_rows", len(df))
        mlflow.log_param(f"{dataset_name}_columns", len(df.columns))
        mlflow.log_param(f"{dataset_name}_size_mb", df.memory_usage().sum() / 1024**2)
        
        # Log distribution
        usage_dist = df['usage_type'].value_counts().to_dict()
        mlflow.log_dict(usage_dist, f"{dataset_name}_usage_distribution.json")
        
        price_stats = {
            "min": float(df['price'].min()),
            "max": float(df['price'].max()),
            "mean": float(df['price'].mean()),
            "median": float(df['price'].median())
        }
        mlflow.log_dict(price_stats, f"{dataset_name}_price_stats.json")
    
    def log_evaluation_results(self, results: Dict):
        """Log comprehensive evaluation"""
        # Main metrics
        metrics = {
            "mean_similarity": results.get('mean_similarity', 0),
            "diversity_score": results.get('mean_diversity', 0),
            "coverage_score": results.get('mean_price_coverage', 0),
            "ndcg@5": results.get('ndcg_at_5', 0),
            "precision@5": results.get('precision_at_5', 0)
        }
        self.log_metrics(metrics)
        
        # Detailed results as artifact
        mlflow.log_dict(results, "evaluation_details.json")
    
    def register_model(self, model_name: str = "laptop_recommender"):
        """Register model to MLflow Model Registry"""
        if self.active_run:
            model_uri = f"runs:/{self.active_run.info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
    
    def end_run(self):
        """End current run"""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def get_best_run(self, metric: str = "ndcg@5", mode: str = "DESC"):
        """Get best run from experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {mode}"]
        )
        return runs.iloc[0] if len(runs) > 0 else None