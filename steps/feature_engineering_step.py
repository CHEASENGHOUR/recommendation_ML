from zenml import step, log_artifact_metadata
from src.text_encoder import LaptopTextEncoder
import numpy as np

@step
def feature_engineering_step(df):
    """Encode laptops using Sentence Transformers"""
    encoder = LaptopTextEncoder('all-MiniLM-L6-v2')
    embeddings = encoder.encode_laptops(df)
    
    log_artifact_metadata(
        artifact_name="laptop_embeddings",
        metadata={
            "model": "all-MiniLM-L6-v2",
            "embedding_dim": encoder.embedding_dim,
            "n_laptops": len(df),
            "sample_description": encoder.create_laptop_description(df.iloc[0])
        }
    )
    
    return embeddings, encoder