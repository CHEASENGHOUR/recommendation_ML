import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.text_encoder import LaptopTextEncoder
from src.knn_index import LaptopKNNIndex


class LaptopRecommendationEngine:
    """ High-level engine used both during training and in the Django API. """

    def __init__(self):
        self.encoder:      Optional[LaptopTextEncoder] = None
        self.index:        Optional[LaptopKNNIndex]    = None
        self.df:           Optional[pd.DataFrame]      = None
        self._encoder_model: str = "all-MiniLM-L6-v2"

    def fit(
        self,
        df:            pd.DataFrame,
        encoder_model: str = "all-MiniLM-L6-v2",
    ) -> "LaptopRecommendationEngine":
        """Encode all laptops and build the FAISS KNN index."""
        self.df             = df.copy()
        self._encoder_model = encoder_model
        self.encoder        = LaptopTextEncoder(encoder_model)

        # Encode → already L2-normalised inside encode_laptops()
        embeddings = self.encoder.encode_laptops(df)

        # Build metadata lookup  {laptop_id → dict}
        metadata = {
            int(row["laptop_id"]): {
                "name":       row["name"],
                "brand":      row["brand"],
                "price":      float(row["price"]),
                "cpu":        row["cpu"],
                "gpu":        row["gpu"],
                "ram":        int(row["ram_capacity"]),
                "ssd":        int(row["ssd"]),
                "rating":     float(row["user_rating"]),
                "usage_type": row.get("usage_type", "unknown"),
                "screen_size":row.get("screen_size", 15),
            }
            for _, row in df.iterrows()
        }

        # Build KNN index
        self.index = LaptopKNNIndex(embedding_dim=self.encoder.embedding_dim)
        self.index.build_index(embeddings, df["laptop_id"].tolist(), metadata)

        return self

    def get_similar_laptops(
        self,
        laptop_id:   int,
        n:           int = 5,
        price_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict]:
        """Return laptops most similar to *laptop_id*."""
        self._check_fitted()
        if laptop_id not in self.index.metadata:
            return []

        meta  = self.index.metadata[laptop_id]
        text  = f"{meta['name']}. {meta['cpu']}. {meta['gpu']}"
        q_emb = self.encoder.encode_query(text)

        results = self.index.search(q_emb, k=n + 1, price_range=price_range)
        return self._format(results, exclude_id=laptop_id)[:n]

    def search_by_text(
        self,
        query:        str,
        n:            int = 5,
        price_range:  Optional[Tuple[float, float]] = None,
        usage_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Search laptops using a natural-language query string."""
        self._check_fitted()
        q_emb   = self.encoder.encode_query(query)
        results = self.index.search(
            q_emb, k=n,
            price_range=price_range,
            usage_filter=usage_filter,
        )
        return self._format(results)

    def get_recommendations_by_preferences(
        self,
        usage_type:      Optional[str]   = None,
        max_price:       Optional[float] = None,
        min_price:       Optional[float] = None,
        preferred_brand: Optional[str]   = None,
        min_ram:         Optional[int]   = None,
        n:               int = 5,
    ) -> List[Dict]:
        """Structured preference-based recommendations."""
        self._check_fitted()
        parts = ["laptop"]
        if usage_type:      parts.append(f"for {usage_type}")
        if preferred_brand: parts.append(f"by {preferred_brand}")
        if min_ram:         parts.append(f"with {min_ram}GB RAM")

        q_emb       = self.encoder.encode_query(" ".join(parts))
        price_range = None
        if min_price is not None or max_price is not None:
            price_range = (min_price or 0, max_price or float("inf"))

        results = self.index.search(
            q_emb, k=n,
            price_range=price_range,
            usage_filter=usage_type,
        )
        return self._format(results)

    def get_personalized_recommendations(
        self,
        user_history: List[int],
        n:            int = 5,
    ) -> List[Dict]:
        """Blend embeddings of previously viewed laptops (recency-weighted)."""
        self._check_fitted()
        if not user_history:
            return self._popular(n)

        embs = []
        for lid in user_history:
            if lid in self.index.metadata:
                meta = self.index.metadata[lid]
                text = f"{meta['name']}. {meta['cpu']}. {meta['gpu']}"
                embs.append(self.encoder.encode_query(text))

        if not embs:
            return self._popular(n)

        weights = np.exp(np.linspace(-1, 0, len(embs)))
        weights /= weights.sum()
        q_emb = np.average(embs, axis=0, weights=weights).astype("float32")

        results  = self.index.search(q_emb, k=n * 2)
        filtered = [r for r in results if r["laptop_id"] not in user_history]
        return self._format(filtered[:n])
    
    def save(self, path_prefix: str) -> None:

        self.index.save(f"{path_prefix}_index")
        config = {
            "encoder_model": self._encoder_model,
            "df":            self.df,
        }
        with open(f"{path_prefix}_config.pkl", "wb") as f:
            pickle.dump(config, f)
        print(f"[engine] Model saved → {path_prefix}_*")

    def load(self, path_prefix: str) -> "LaptopRecommendationEngine":
        self.index = LaptopKNNIndex()
        self.index.load(f"{path_prefix}_index")

        # Load config
        with open(f"{path_prefix}_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.df             = config["df"]
        self._encoder_model = config["encoder_model"]
        self.encoder        = LaptopTextEncoder(self._encoder_model)

        print(f"[engine] Model loaded ← {path_prefix}_*")
        return self

    def _format(
        self,
        results:    List[dict],
        exclude_id: Optional[int] = None,
    ) -> List[Dict]:
        out = []
        for r in results:
            if exclude_id is not None and r["laptop_id"] == exclude_id:
                continue
            m = r["metadata"]
            out.append({
                "laptop_id":        r["laptop_id"],
                "name":             m["name"],
                "brand":            m["brand"],
                "price":            m["price"],
                "cpu":              m["cpu"],
                "gpu":              m["gpu"],
                "ram_capacity":     m["ram"],
                "ssd":              m["ssd"],
                "user_rating":      m["rating"],
                "usage_type":       m["usage_type"],
                "similarity_score": round(r["similarity_score"], 4),
            })
        return out

    def _popular(self, n: int) -> List[Dict]:
        """Fallback — return highest-rated laptops when no history available."""
        top = self.df.nlargest(n, "user_rating")
        return [
            {
                "laptop_id":        int(row["laptop_id"]),
                "name":             row["name"],
                "brand":            row["brand"],
                "price":            float(row["price"]),
                "cpu":              row["cpu"],
                "gpu":              row["gpu"],
                "ram_capacity":     int(row["ram_capacity"]),
                "ssd":              int(row["ssd"]),
                "user_rating":      float(row["user_rating"]),
                "usage_type":       row.get("usage_type", "unknown"),
                "similarity_score": 1.0,
            }
            for _, row in top.iterrows()
        ]

    def _check_fitted(self) -> None:
        if self.index is None or self.encoder is None:
            raise RuntimeError("Engine not fitted. Call fit() or load() first.")