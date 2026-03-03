import json
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd

import mlflow
import mlflow.sklearn


class RecommendationExperimentTracker:
    """ Manage MLflow runs for the laptop recommendation system. """

    def __init__(self, experiment_name: str = "laptop_recommendations"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.active_run: Optional[mlflow.ActiveRun] = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags:     Optional[Dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_run = mlflow.start_run(run_name=run_name)

        mlflow.set_tag("framework", "sentence-transformers+faiss")
        if tags:
            mlflow.set_tags(tags)

        return self.active_run

    def end_run(self) -> None:
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step:    Optional[int] = None,
    ) -> None:
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_dict(self, data: dict, artifact_filename: str) -> None:
        """
        Log a Python dict as a JSON artifact.
        (FIX: this method was called in training_pipeline.py but did not exist)
        """
        mlflow.log_dict(data, artifact_filename)

    def log_model_artifact(
        self,
        model_path:    str,
        artifact_path: str = "model",
    ) -> None:
        mlflow.log_artifact(model_path, artifact_path)

    def log_dataset_info(
        self,
        df:           "pd.DataFrame",
        dataset_name: str = "training_data",
    ) -> None:
          # local to avoid circular at module load
        mlflow.log_param(f"{dataset_name}_rows",      len(df))
        mlflow.log_param(f"{dataset_name}_columns",   len(df.columns))
        mlflow.log_param(f"{dataset_name}_size_mb",
                         round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3))

        usage_dist  = df["usage_type"].value_counts().to_dict()
        price_stats = {
            "min":    float(df["price"].min()),
            "max":    float(df["price"].max()),
            "mean":   float(df["price"].mean()),
            "median": float(df["price"].median()),
        }
        mlflow.log_dict(usage_dist,  f"{dataset_name}_usage_dist.json")
        mlflow.log_dict(price_stats, f"{dataset_name}_price_stats.json")

    def log_evaluation_results(self, results: Dict) -> None:
        metrics = {
            "mean_similarity":  results.get("mean_similarity",    0.0),
            "diversity_score":  results.get("mean_diversity",     0.0),
            "coverage_score":   results.get("mean_price_coverage",0.0),
            "ndcg_at_5":           results.get("ndcg_at_5",          0.0),
            "precision_at_5":      results.get("precision_at_5",     0.0),
        }
        self.log_metrics(metrics)
        mlflow.log_dict(results, "evaluation_details.json")

    def register_model(self, model_name: str = "laptop_recommender") -> None:
        if self.active_run:
            model_uri = f"runs:/{self.active_run.info.run_id}/model"
            mlflow.register_model(model_uri, model_name)

    def get_best_run(
        self,
        metric: str = "mean_similarity",
        mode:   str = "DESC",
    ) -> Optional[Any]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {mode}"],
        )
        return runs.iloc[0] if len(runs) > 0 else None