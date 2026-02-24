from zenml import step, log_artifact_metadata
from src.knn_index import LaptopKNNIndex
from src.recommendation_engine import LaptopRecommendationEngine
import pickle

@step
def model_building_step(df, embeddings, encoder):
    """Build KNN index and recommendation engine"""
    # Build metadata
    metadata = {}
    for _, row in df.iterrows():
        metadata[int(row['laptop_id'])] = {
            'name': row['name'],
            'brand': row['brand'],
            'price': float(row['price']),
            'cpu': row['cpu'],
            'gpu': row['gpu'],
            'ram': int(row['ram_capacity']),
            'ssd': int(row['ssd']),
            'rating': float(row['user_rating']),
            'usage_type': row.get('usage_type', 'unknown')
        }
    
    # Build index
    index = LaptopKNNIndex(embedding_dim=encoder.embedding_dim)
    index.build_index(embeddings, df['laptop_id'].tolist(), metadata)
    
    # Test search
    test_emb = encoder.encode_query("gaming laptop with RTX")
    test_results = index.search(test_emb, k=3)
    
    log_artifact_metadata(
        artifact_name="knn_index",
        metadata={
            "index_type": "FAISS-FlatIP",
            "n_vectors": index.index.ntotal,
            "test_results": len(test_results),
            "embedding_dim": encoder.embedding_dim
        }
    )
    
    # Build full engine
    engine = LaptopRecommendationEngine()
    engine.encoder = encoder
    engine.index = index
    engine.df = df
    
    return engine