import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.text_encoder import LaptopTextEncoder
from src.knn_index import LaptopKNNIndex

class LaptopRecommendationEngine:
    def __init__(self):
        self.encoder = None
        self.index = None
        self.df = None
        
    def fit(self, df: pd.DataFrame, encoder_model: str = 'all-MiniLM-L6-v2'):
        """Build the recommendation engine"""
        self.df = df.copy()
        
        # Initialize encoder
        self.encoder = LaptopTextEncoder(encoder_model)
        
        # Create embeddings
        embeddings = self.encoder.encode_laptops(df)
        
        # Build metadata lookup
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
                'usage_type': row.get('usage_type', 'unknown'),
                'screen_size': row['screen_size']
            }
        
        # Build KNN index
        self.index = LaptopKNNIndex(embedding_dim=self.encoder.embedding_dim)
        self.index.build_index(
            embeddings, 
            df['laptop_id'].tolist(),
            metadata
        )
        
        return self
    
    def get_similar_laptops(
        self, 
        laptop_id: int, 
        n: int = 5,
        price_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """Find laptops similar to given laptop ID"""
        if laptop_id not in self.index.metadata:
            return []
        
        # Get laptop embedding
        laptop_meta = self.index.metadata[laptop_id]
        query_text = f"{laptop_meta['name']}. {laptop_meta['cpu']}. {laptop_meta['gpu']}"
        query_embedding = self.encoder.encode_query(query_text)
        
        # Search
        results = self.index.search(
            query_embedding, 
            k=n,
            price_range=price_range
        )
        
        return self._format_results(results, exclude_id=laptop_id)
    
    def search_by_text(
        self, 
        query: str, 
        n: int = 5,
        price_range: Optional[Tuple[float, float]] = None,
        usage_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search laptops by natural language query"""
        query_embedding = self.encoder.encode_query(query)
        
        results = self.index.search(
            query_embedding,
            k=n,
            price_range=price_range,
            usage_filter=usage_filter
        )
        
        return self._format_results(results)
    
    def get_recommendations_by_preferences(
        self,
        usage_type: Optional[str] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        preferred_brand: Optional[str] = None,
        min_ram: Optional[int] = None,
        n: int = 5
    ) -> List[Dict]:
        """Get recommendations based on preferences"""
        # Build preference query
        query_parts = ["laptop"]
        
        if usage_type:
            query_parts.append(f"for {usage_type}")
        if preferred_brand:
            query_parts.append(f"by {preferred_brand}")
        if min_ram:
            query_parts.append(f"with {min_ram}GB RAM")
            
        query = " ".join(query_parts)
        query_embedding = self.encoder.encode_query(query)
        
        # Set price range
        price_range = None
        if min_price or max_price:
            price_range = (min_price or 0, max_price or float('inf'))
        
        # Search with usage filter
        results = self.index.search(
            query_embedding,
            k=n,
            price_range=price_range,
            usage_filter=usage_type
        )
        
        return self._format_results(results)
    
    def get_personalized_recommendations(
        self,
        user_history: List[int],
        n: int = 5
    ) -> List[Dict]:
        """Get recommendations based on user's viewing history"""
        if not user_history:
            # Return popular items
            return self._get_popular_recommendations(n)
        
        # Average embeddings of viewed laptops
        embeddings = []
        for lid in user_history:
            if lid in self.index.metadata:
                meta = self.index.metadata[lid]
                text = f"{meta['name']}. {meta['cpu']}. {meta['gpu']}"
                emb = self.encoder.encode_query(text)
                embeddings.append(emb)
        
        if not embeddings:
            return self._get_popular_recommendations(n)
        
        # Weighted average (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights /= weights.sum()
        
        query_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # Exclude already seen
        results = self.index.search(query_embedding, k=n * 2)
        filtered = [r for r in results if r['laptop_id'] not in user_history]
        
        return self._format_results(filtered[:n])
    
    def _format_results(
        self, 
        results: List[dict], 
        exclude_id: Optional[int] = None
    ) -> List[Dict]:
        """Format results for API response"""
        formatted = []
        for r in results:
            if exclude_id and r['laptop_id'] == exclude_id:
                continue
                
            meta = r['metadata']
            formatted.append({
                'laptop_id': r['laptop_id'],
                'name': meta['name'],
                'brand': meta['brand'],
                'price': meta['price'],
                'cpu': meta['cpu'],
                'gpu': meta['gpu'],
                'ram_capacity': meta['ram'],
                'ssd': meta['ssd'],
                'user_rating': meta['rating'],
                'usage_type': meta['usage_type'],
                'similarity_score': round(r['similarity_score'], 4)
            })
        return formatted
    
    def _get_popular_recommendations(self, n: int) -> List[Dict]:
        """Fallback: return highest rated laptops"""
        top_laptops = self.df.nlargest(n, 'user_rating')
        results = []
        for _, row in top_laptops.iterrows():
            results.append({
                'laptop_id': int(row['laptop_id']),
                'name': row['name'],
                'brand': row['brand'],
                'price': float(row['price']),
                'cpu': row['cpu'],
                'gpu': row['gpu'],
                'ram_capacity': int(row['ram_capacity']),
                'ssd': int(row['ssd']),
                'user_rating': float(row['user_rating']),
                'usage_type': row.get('usage_type', 'unknown'),
                'similarity_score': 1.0
            })
        return results
    
    def save(self, path_prefix: str):
        """Save model components"""
        self.index.save(f"{path_prefix}_index")
        # Save encoder config and df
        import pickle
        with open(f"{path_prefix}_config.pkl", 'wb') as f:
            pickle.dump({
                'df': self.df,
                'encoder_model': 'all-MiniLM-L6-v2'
            }, f)
    
    def load(self, path_prefix: str):
        """Load model components"""
        import pickle
        self.index = LaptopKNNIndex()
        self.index.load(f"{path_prefix}_index")
        
        with open(f"{path_prefix}_config.pkl", 'rb') as f:
            config = pickle.load(f)
            self.df = config['df']
            
        self.encoder = LaptopTextEncoder(config['encoder_model'])