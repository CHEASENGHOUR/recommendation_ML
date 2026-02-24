import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional

class LaptopKNNIndex:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.laptop_ids = None
        self.metadata = None
        
    def build_index(self, embeddings: np.ndarray, laptop_ids: List[int], metadata: dict):
        """Build FAISS index for fast KNN search"""
        self.laptop_ids = np.array(laptop_ids)
        self.metadata = metadata
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine after norm
        
        # Add vectors
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built index with {self.index.ntotal} laptops")
        
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10,
        price_range: Optional[Tuple[float, float]] = None,
        usage_filter: Optional[str] = None
    ) -> List[dict]:
        """Search K nearest neighbors with optional filters"""
        # Normalize query
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k * 3  # Get more for filtering
        )
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
                
            laptop_id = int(self.laptop_ids[idx])
            meta = self.metadata[laptop_id]
            
            # Apply filters
            if price_range:
                if not (price_range[0] <= meta['price'] <= price_range[1]):
                    continue
            if usage_filter and meta.get('usage_type') != usage_filter:
                continue
                
            results.append({
                'laptop_id': laptop_id,
                'similarity_score': float(dist),
                'metadata': meta
            })
            
            if len(results) >= k:
                break
                
        return results
    
    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump({
                'laptop_ids': self.laptop_ids,
                'metadata': self.metadata
            }, f)
    
    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", 'rb') as f:
            data = pickle.load(f)
            self.laptop_ids = data['laptop_ids']
            self.metadata = data['metadata']