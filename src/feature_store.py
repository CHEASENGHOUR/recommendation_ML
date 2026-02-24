import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import json
import os

class LaptopFeatureStore:
    """Feature store for laptop recommendation system"""
    
    def __init__(self, store_path: str = "extracted_data"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        
        # Subdirectories
        self.raw_path = f"{store_path}/raw"
        self.processed_path = f"{store_path}/processed"
        self.embeddings_path = f"{store_path}/embeddings"
        
        for path in [self.raw_path, self.processed_path, self.embeddings_path]:
            os.makedirs(path, exist_ok=True)
    
    def compute_feature_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for versioning"""
        content = pd.util.hash_pandas_object(df).sum()
        return hashlib.md5(str(content).encode()).hexdigest()[:8]
    
    def save_raw_data(self, df: pd.DataFrame, source: str = "csv"):
        """Save raw ingested data with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_hash = self.compute_feature_hash(df)
        version = f"{timestamp}_{feature_hash}"
        
        # FIX: Remove extra /raw - self.raw_path already ends with /raw
        filepath = f"{self.raw_path}/laptops_{version}.parquet"
        df.to_parquet(filepath)
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": timestamp,
            "source": source,
            "rows": len(df),
            "columns": list(df.columns),
            "hash": feature_hash
        }
        
        # FIX: Remove extra /raw here too
        with open(f"{self.raw_path}/laptops_{version}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved raw data: {filepath}")
        return version, metadata  # Return tuple to match main.py expectation
    
    def save_processed_features(
        self, 
        df: pd.DataFrame, 
        embeddings: np.ndarray,
        version: str,
        transformation_params: Dict
    ):
        """Save processed features and embeddings"""
        # Save processed dataframe
        df_path = f"{self.processed_path}/processed_{version}.parquet"
        df.to_parquet(df_path)
        
        # Save embeddings
        emb_path = f"{self.embeddings_path}/embeddings_{version}.npy"
        np.save(emb_path, embeddings)
        
        # Save transformation metadata
        meta = {
            "version": version,
            "transformation_params": transformation_params,
            "embedding_shape": embeddings.shape,
            "feature_columns": list(df.columns),
            "created_at": datetime.now().isoformat()
        }
        
        with open(f"{self.processed_path}/meta_{version}.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        return {
            "df_path": df_path,
            "emb_path": emb_path,
            "meta": meta
        }
    
    def get_latest_version(self, data_type: str = "processed") -> Optional[str]:
        """Get latest data version"""
        path = self.processed_path if data_type == "processed" else self.raw_path
        files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        
        if not files:
            return None
        
        # Sort by timestamp in filename
        files.sort(reverse=True)
        return files[0].replace('processed_', '').replace('.parquet', '').replace('laptops_', '')
    
    def load_processed_features(self, version: Optional[str] = None):
        """Load processed features by version"""
        if version is None:
            version = self.get_latest_version()
        
        df_path = f"{self.processed_path}/processed_{version}.parquet"
        emb_path = f"{self.embeddings_path}/embeddings_{version}.npy"
        
        df = pd.read_parquet(df_path)
        embeddings = np.load(emb_path)
        
        return df, embeddings, version
    
    def get_feature_lineage(self, version: str) -> Dict:
        """Get lineage information for features"""
        meta_path = f"{self.processed_path}/meta_{version}.json"
        
        if not os.path.exists(meta_path):
            return {}
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def create_feature_profile(self, df: pd.DataFrame) -> Dict:
        """Create data profile for monitoring"""
        profile = {
            "numeric_stats": {},
            "categorical_stats": {},
            "missing_values": {},
            "correlations": {}
        }
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "zeros": int((df[col] == 0).sum())
            }
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            profile["categorical_stats"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        
        # Missing values
        profile["missing_values"] = df.isnull().sum().to_dict()
        
        return profile