# from sentence_transformers import SentenceTransformer
# import numpy as np
# import pandas as pd
# from typing import List, Union

# class LaptopTextEncoder:
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         """Initialize sentence transformer model"""
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
#     def create_laptop_description(self, row: pd.Series) -> str:
#         """Create rich text description from laptop specs"""
#         parts = [
#             f"{row['brand']} laptop",
#             row['name'],
#             f"with {row['cpu']}",
#             f"and {row['gpu']}",
#             f"{int(row['ram_capacity'])}GB RAM",
#             f"{int(row['ssd'])}GB SSD",
#             f"{row['screen_size']} inch screen",
#             f"priced at {int(row['price'])} rupees",
#             f"rated {row['user_rating']} stars"
#         ]
        
#         # Add usage type hint
#         if row.get('gpu_vram', 0) > 0:
#             parts.append("gaming capable")
#         if row['price'] > 100000:
#             parts.append("premium professional")
#         elif row['price'] < 40000:
#             parts.append("budget friendly")
            
#         return ". ".join(parts)
    
#     def encode_laptops(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
#         """Encode all laptops to embeddings"""
#         descriptions = df.apply(self.create_laptop_description, axis=1).tolist()
        
#         print(f"Encoding {len(descriptions)} laptops...")
#         embeddings = self.model.encode(
#             descriptions, 
#             batch_size=batch_size,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )
#         return embeddings
    
#     def encode_query(self, query: str) -> np.ndarray:
#         """Encode user query to embedding"""
#         return self.model.encode(query, convert_to_numpy=True)
    
#     def encode_preferences(
#         self, 
#         usage_type: str = None,
#         brand: str = None,
#         cpu_tier: str = None,
#         gpu_type: str = None
#     ) -> np.ndarray:
#         """Encode user preferences as synthetic query"""
#         parts = ["laptop"]
        
#         if usage_type:
#             parts.append(f"for {usage_type}")
#         if brand:
#             parts.append(f"by {brand}")
#         if cpu_tier:
#             parts.append(f"with {cpu_tier} processor")
#         if gpu_type:
#             parts.append(f"and {gpu_type} graphics")
            
#         query = " ".join(parts)
#         return self.encode_query(query)

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

class LaptopTextEncoder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dim: {self.embedding_dim}")
        
    def create_laptop_description(self, row: pd.Series) -> str:
        parts = [
            f"{row['brand']} laptop",
            str(row['name']),
            f"with {row['cpu']}",
            f"and {row['gpu']}",
            f"{int(row['ram_capacity'])}GB RAM",
            f"{int(row['ssd'])}GB SSD",
            f"{row['screen_size']} inch screen",
            f"priced at {int(row['price'])} rupees",
            f"rated {row['user_rating']} stars"
        ]
        
        if row.get('gpu_vram', 0) > 0:
            parts.append("gaming capable")
        if row['price'] > 100000:
            parts.append("premium professional")
        elif row['price'] < 40000:
            parts.append("budget friendly")
            
        return ". ".join(parts)
    
    def encode_laptops(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        descriptions = df.apply(self.create_laptop_description, axis=1).tolist()
        print(f"Encoding {len(descriptions)} laptops...")
        
        embeddings = self.model.encode(
            descriptions, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)