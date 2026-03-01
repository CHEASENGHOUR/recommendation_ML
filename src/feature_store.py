import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class LaptopFeatureStore:
    """
    Lightweight feature store backed by local files.

    Directory layout
    ----------------
    extracted_data/
      raw/            ← raw parquet + json metadata
      processed/      ← cleaned parquet + json metadata
      embeddings/     ← .npy embedding arrays
    """

    def __init__(self, store_path: str = "extracted_data"):
        self.store_path       = store_path
        self.raw_path         = os.path.join(store_path, "raw")
        self.processed_path   = os.path.join(store_path, "processed")
        self.embeddings_path  = os.path.join(store_path, "embeddings")

        for path in (self.raw_path, self.processed_path, self.embeddings_path):
            os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def _compute_hash(self, df: pd.DataFrame) -> str:
        content = pd.util.hash_pandas_object(df).sum()
        return hashlib.md5(str(content).encode()).hexdigest()[:8]

    def _make_version(self, df: pd.DataFrame) -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        h    = self._compute_hash(df)
        return f"{ts}_{h}"

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    def save_raw_data(
        self,
        df:     pd.DataFrame,
        source: str = "csv",
    ) -> Tuple[str, dict]:
        """
        Persist raw DataFrame and return (version, metadata).
        """
        version  = self._make_version(df)
        filepath = os.path.join(self.raw_path, f"laptops_{version}.parquet")
        df.to_parquet(filepath, index=False)

        metadata = {
            "version":   version,
            "timestamp": datetime.now().isoformat(),
            "source":    source,
            "rows":      len(df),
            "columns":   list(df.columns),
        }
        meta_path = os.path.join(self.raw_path, f"laptops_{version}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[feature_store] Raw data saved → {filepath}")
        return version, metadata

    # ------------------------------------------------------------------
    # Processed features + embeddings
    # ------------------------------------------------------------------

    def save_processed_features(
        self,
        df:                   pd.DataFrame,
        embeddings:           np.ndarray,
        version:              str,
        transformation_params: Dict,
    ) -> dict:
        """Persist processed DataFrame and embeddings."""
        df_path  = os.path.join(self.processed_path, f"processed_{version}.parquet")
        emb_path = os.path.join(self.embeddings_path, f"embeddings_{version}.npy")

        df.to_parquet(df_path, index=False)
        np.save(emb_path, embeddings)

        meta = {
            "version":               version,
            "transformation_params": transformation_params,
            "embedding_shape":       list(embeddings.shape),
            "feature_columns":       list(df.columns),
            "created_at":            datetime.now().isoformat(),
        }
        meta_path = os.path.join(self.processed_path, f"meta_{version}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[feature_store] Processed features saved → {df_path}")
        return {"df_path": df_path, "emb_path": emb_path, "meta": meta}

    def load_processed_features(
        self, version: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, str]:
        """Load processed DataFrame + embeddings by version (or latest)."""
        if version is None:
            version = self.get_latest_version()
        if version is None:
            raise FileNotFoundError("No processed features found in store.")

        df_path  = os.path.join(self.processed_path, f"processed_{version}.parquet")
        emb_path = os.path.join(self.embeddings_path, f"embeddings_{version}.npy")

        df         = pd.read_parquet(df_path)
        embeddings = np.load(emb_path)
        return df, embeddings, version

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_latest_version(self, data_type: str = "processed") -> Optional[str]:
        """Return the most recent version string, or None."""
        base   = self.processed_path if data_type == "processed" else self.raw_path
        prefix = "processed_" if data_type == "processed" else "laptops_"
        files  = [
            f for f in os.listdir(base)
            if f.startswith(prefix) and f.endswith(".parquet")
        ]
        if not files:
            return None
        files.sort(reverse=True)
        return files[0].replace(prefix, "").replace(".parquet", "")

    def create_feature_profile(self, df: pd.DataFrame) -> Dict:
        """Basic data profile for monitoring / logging."""
        numeric_cols     = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        profile: Dict = {
            "numeric_stats":     {},
            "categorical_stats": {},
            "missing_values":    df.isnull().sum().to_dict(),
        }

        for col in numeric_cols:
            profile["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std":  float(df[col].std()),
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
            }
        for col in categorical_cols:
            profile["categorical_stats"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values":   df[col].value_counts().head(5).to_dict(),
            }

        return profile