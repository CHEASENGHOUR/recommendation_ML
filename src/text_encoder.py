import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class LaptopTextEncoder:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[text_encoder] Loading model: {model_name} …")
        self.model_name    = model_name
        self.model         = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[text_encoder] Ready — embedding_dim={self.embedding_dim}")

    def create_laptop_description(self, row: pd.Series) -> str:
        """
        Build a rich natural-language description from a laptop row.
        All fields are safely converted to str so NaN never breaks join.
        """
        parts = [
            f"{row.get('brand', 'Unknown')} laptop",
            str(row.get("name", "")),
            f"with {row.get('cpu', 'unknown CPU')}",
            f"and {row.get('gpu', 'integrated graphics')}",
            f"{int(row.get('ram_capacity', 8))}GB RAM",
            f"{int(row.get('ssd', 256))}GB SSD",
            f"{row.get('screen_size', 15)} inch screen",
            # f"priced at {int(row.get('price', 0))} rupees",
            f"priced at {int(row.get('price_usd', 0))} US dollars",
            f"rated {row.get('user_rating', 0):.1f} stars",
        ]

        # Contextual hints that improve semantic matching
        if row.get("gpu_vram", 0) > 0:
            parts.append("gaming capable")
        # price = row.get("price", 50000)
        # if price > 100_000:
        #     parts.append("premium professional laptop")
        # elif price < 40_000:
        #     parts.append("budget friendly affordable")
        
        price = row.get("price_usd", 500)

        if price > 1500:
            parts.append("premium professional laptop")
        elif price < 500:
            parts.append("budget friendly affordable")

        usage = row.get("usage_type", "")
        if usage:
            parts.append(f"{usage} use")

        return ". ".join(parts)

    def encode_laptops(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """
        Encode every row in *df* into a (N, embedding_dim) float32 array.
        Embeddings are L2-normalised so cosine similarity == dot product.
        """
        descriptions = df.apply(self.create_laptop_description, axis=1).tolist()
        print(f"[text_encoder] Encoding {len(descriptions)} laptops …")

        embeddings = self.model.encode(
            descriptions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm built-in
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single free-text query.
        Returns a (embedding_dim,) float32 array, L2-normalised.
        """
        emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    def encode_preferences(
        self,
        usage_type:  str | None = None,
        brand:       str | None = None,
        cpu_tier:    str | None = None,
        gpu_type:    str | None = None,
    ) -> np.ndarray:
        parts = ["laptop"]
        if usage_type:
            parts.append(f"for {usage_type}")
        if brand:
            parts.append(f"by {brand}")
        if cpu_tier:
            parts.append(f"with {cpu_tier} processor")
        if gpu_type:
            parts.append(f"and {gpu_type} graphics")
        return self.encode_query(" ".join(parts))