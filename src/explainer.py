from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Words to ignore when extracting matching tokens
_STOPWORDS = {
    "a", "an", "the", "with", "and", "for", "to", "of", "in",
    "is", "it", "at", "on", "by", "as", "or", "be", "this",
}


class RecommendationExplainer:
    """
    Explain recommendation decisions at three levels:

    1. explain_similarity()     — item-to-item comparison
    2. explain_text_query()     — why a laptop matches a free-text query
    3. explain_preference_match()— how laptop satisfies user preferences
    4. get_global_feature_importance() — dataset-level embedding analysis
    """

    def __init__(self, encoder, index, df: pd.DataFrame):
        self.encoder = encoder
        self.index   = index
        self.df      = df


    def explain_similarity(
        self,
        query_laptop_id:       int,
        recommended_laptop_id: int,
    ) -> Dict:
        query_row = self.df[self.df["laptop_id"] == query_laptop_id].iloc[0]
        rec_row   = self.df[self.df["laptop_id"] == recommended_laptop_id].iloc[0]

        comparisons = self._compare_features(query_row, rec_row)

        return {
            "query_laptop":       query_row["name"],
            "recommended_laptop": rec_row["name"],
            "overall_similarity": self._weighted_similarity(query_row, rec_row),
            "feature_comparisons": comparisons[:5],
            "key_insight":        self._generate_insight(query_row, rec_row, comparisons),
        }

    def explain_text_query(
        self,
        query:                 str,
        recommended_laptop_id: int,
    ) -> Dict:
        laptop_row  = self.df[self.df["laptop_id"] == recommended_laptop_id].iloc[0]
        laptop_text = self.encoder.create_laptop_description(laptop_row)
        laptop_emb  = self.encoder.encode_query(laptop_text)

        # Token-level attribution
        tokens = [t for t in query.lower().split() if t not in _STOPWORDS]
        token_scores: List[Dict] = []
        for token in tokens:
            t_emb = self.encoder.encode_query(token)
            sim   = float(cosine_similarity(
                t_emb.reshape(1, -1),
                laptop_emb.reshape(1, -1),
            )[0][0])
            token_scores.append({
                "token":           token,
                "relevance_score": round(sim, 3),
                "matched":         sim > 0.4,
            })

        token_scores.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "query":               query,
            "laptop_name":         laptop_row["name"],
            "top_matching_terms":  [t for t in token_scores if t["matched"]][:5],
            "query_intent":        self._infer_intent(query),
            "why_recommended":     self._text_explanation(query, laptop_row),
        }

    def explain_preference_match(
        self,
        preferences:           Dict,
        recommended_laptop_id: int,
    ) -> Dict:
        laptop = self.df[self.df["laptop_id"] == recommended_laptop_id].iloc[0]

        checks = [
            ("usage_type",     preferences.get("usage_type"),
             lambda p, l: p.lower() in str(l).lower()),
            ("max_price",      preferences.get("max_price"),
             lambda p, l: l <= p),
            ("min_ram",        preferences.get("min_ram"),
             lambda p, l: l >= p),
            ("preferred_brand",preferences.get("preferred_brand"),
             lambda p, l: p.lower() == str(l).lower()),
        ]

        col_map = {
            "usage_type":      "usage_type",
            "max_price":       "price",
            "min_ram":         "ram_capacity",
            "preferred_brand": "brand",
        }

        matches:    List[Dict] = []
        mismatches: List[Dict] = []

        for pref_name, pref_val, fn in checks:
            if pref_val is None:
                continue
            actual = laptop.get(col_map[pref_name], None)
            ok     = fn(pref_val, actual) if actual is not None else False
            entry  = {"preference": pref_name, "requested": pref_val, "actual": actual, "matched": ok}
            (matches if ok else mismatches).append(entry)

        total       = len(matches) + len(mismatches)
        match_score = len(matches) / total if total else 0.0

        return {
            "match_score":           round(match_score, 2),
            "satisfied_preferences": matches,
            "trade_offs":            mismatches,
            "overall_assessment":    self._assess_trade_offs(matches, mismatches),
        }

    def get_global_feature_importance(self) -> Dict:
        """Correlate embedding dimensions with price as a relevance proxy."""
        sample_size = min(100, len(self.df))
        sample      = self.df.sample(sample_size, random_state=42)

        embeddings = np.array([
            self.encoder.encode_query(self.encoder.create_laptop_description(row))
            for _, row in sample.iterrows()
        ])
        prices = sample["price"].values.astype(float)

        correlations = []
        for dim in range(embeddings.shape[1]):
            corr = np.corrcoef(embeddings[:, dim], prices)[0, 1]
            correlations.append(0.0 if np.isnan(corr) else abs(corr))

        top_dims = np.argsort(correlations)[-10:][::-1].tolist()

        return {
            "top_influential_dimensions": top_dims,
            "explanation":   "Embedding dims most correlated with laptop price/quality.",
            "interpretation":"Model prioritises: brand reputation, GPU tier, CPU generation, RAM.",
        }

    def _compare_features(self, q: pd.Series, r: pd.Series) -> List[Dict]:
        specs = [
            ("Price",    "price",        lambda x: f"₹{x:,.0f}",
             lambda a, b: "cheaper" if b < a else "premium"),
            ("RAM",      "ram_capacity", lambda x: f"{int(x)}GB",
             lambda a, b: "more RAM" if b > a else "less RAM"),
            ("Storage",  "ssd",          lambda x: f"{int(x)}GB SSD",
             lambda a, b: "more storage" if b > a else "less storage"),
            ("GPU VRAM", "gpu_vram",     lambda x: f"{int(x)}GB VRAM",
             lambda a, b: "better GPU" if b > a else "lighter GPU"),
            ("Rating",   "user_rating",  lambda x: f"{x:.2f}★",
             lambda a, b: "better rated" if b > a else "lower rated"),
        ]
        out = []
        for name, col, fmt, desc in specs:
            qv, rv = q.get(col), r.get(col)
            if pd.isna(qv) or pd.isna(rv):
                continue
            mx  = max(abs(float(qv)), abs(float(rv)), 1)
            sim = 1 - abs(float(qv) - float(rv)) / mx
            out.append({
                "feature":            name,
                "query_value":        fmt(qv),
                "recommended_value":  fmt(rv),
                "similarity_score":   round(sim, 3),
                "difference":         desc(qv, rv),
                "impact":             "high" if sim > 0.8 else "medium" if sim > 0.5 else "low",
            })
        out.sort(key=lambda x: x["similarity_score"], reverse=True)
        return out

    def _weighted_similarity(self, r1: pd.Series, r2: pd.Series) -> float:
        weights = {"price": 0.2, "ram_capacity": 0.15, "ssd": 0.1,
                   "gpu_vram": 0.25, "cpu_cores": 0.15, "user_rating": 0.15}
        score = 0.0
        for feat, w in weights.items():
            v1, v2 = r1.get(feat), r2.get(feat)
            if pd.isna(v1) or pd.isna(v2):
                continue
            mx = max(abs(float(v1)), abs(float(v2)), 1)
            score += (1 - abs(float(v1) - float(v2)) / mx) * w
        return round(score, 3)

    def _generate_insight(self, q: pd.Series, r: pd.Series, comps: List[Dict]) -> str:
        high = [c for c in comps if c["impact"] == "high"]
        if len(high) >= 3:
            return f"Very similar — matches in {', '.join(c['feature'] for c in high[:3])}"
        q_price = float(q.get("price", 1))
        r_price = float(r.get("price", 1))
        if r_price < q_price * 0.8:
            pct = int((1 - r_price / q_price) * 100)
            return f"Budget-friendly alternative — {pct}% cost saving"
        q_perf = float(q.get("performance_score", 0))
        r_perf = float(r.get("performance_score", 0))
        if q_perf and r_perf > q_perf * 1.2:
            return "Higher-performance upgrade"
        return "Alternative with a different feature balance"

    def _infer_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ("gaming", "rtx", "gtx")):
            return "gaming_performance"
        if any(w in q for w in ("cheap", "budget", "under", "below")):
            return "budget_conscious"
        if any(w in q for w in ("professional", "work", "business")):
            return "professional_use"
        if any(w in q for w in ("light", "portable", "student")):
            return "portability"
        return "general_purpose"

    def _text_explanation(self, query: str, laptop: pd.Series) -> str:
        intent = self._infer_intent(query)
        return {
            "gaming_performance": f"Matches gaming needs: {laptop.get('gpu','?')} + {int(laptop.get('ram_capacity',0))}GB RAM",
            "budget_conscious":   f"Fits budget at ₹{laptop.get('price',0):,.0f}",
            "professional_use":   f"Professional-grade: {laptop.get('cpu','?')}",
            "portability":        f"Portable: {laptop.get('screen_size','?')}\" screen",
            "general_purpose":    "Well-rounded with balanced specs",
        }.get(intent, "Good match for your query")

    def _assess_trade_offs(self, matches: List[Dict], mismatches: List[Dict]) -> str:
        if not mismatches:
            return "Perfect match for all preferences"
        if len(mismatches) == 1:
            return f"Minor trade-off: {mismatches[0]['preference']}"
        if len(matches) > len(mismatches):
            return "Good overall match with some compromises"
        return "Several preferences not met — consider relaxing filters"