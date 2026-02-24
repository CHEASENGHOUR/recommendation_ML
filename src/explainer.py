import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import shap
import json

class RecommendationExplainer:
    """Explain why certain laptops are recommended"""
    
    def __init__(self, encoder, index, df):
        self.encoder = encoder
        self.index = index
        self.df = df
        self.feature_names = [
            'price', 'ram_capacity', 'ssd', 'cpu_cores', 
            'gpu_vram', 'screen_size', 'user_rating', 'performance_score'
        ]
    
    def explain_similarity(
        self, 
        query_laptop_id: int, 
        recommended_laptop_id: int
    ) -> Dict:
        """Explain similarity between two laptops"""
        query_row = self.df[self.df['laptop_id'] == query_laptop_id].iloc[0]
        rec_row = self.df[self.df['laptop_id'] == recommended_laptop_id].iloc[0]
        
        explanations = []
        
        # Compare key features
        comparisons = [
            ('Price', 'price', lambda x: f"₹{x:,.0f}", 
             lambda q, r: "cheaper" if r < q else "premium"),
            ('RAM', 'ram_capacity', lambda x: f"{int(x)}GB", 
             lambda q, r: "more RAM" if r > q else "less RAM"),
            ('Storage', 'ssd', lambda x: f"{int(x)}GB SSD", 
             lambda q, r: "more storage" if r > q else "less storage"),
            ('GPU VRAM', 'gpu_vram', lambda x: f"{int(x)}GB", 
             lambda q, r: "better GPU" if r > q else "basic graphics"),
            ('Rating', 'user_rating', lambda x: f"{x:.2f}★", 
             lambda q, r: "better rated" if r > q else "lower rated")
        ]
        
        for name, col, formatter, descriptor in comparisons:
            q_val = query_row[col]
            r_val = rec_row[col]
            
            if pd.isna(q_val) or pd.isna(r_val):
                continue
            
            similarity = 1 - abs(q_val - r_val) / max(abs(q_val), abs(r_val), 1)
            
            explanations.append({
                'feature': name,
                'query_value': formatter(q_val),
                'recommended_value': formatter(r_val),
                'similarity_score': round(similarity, 3),
                'difference': descriptor(q_val, r_val),
                'impact': 'high' if similarity > 0.8 else 'medium' if similarity > 0.5 else 'low'
            })
        
        # Sort by similarity
        explanations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'query_laptop': query_row['name'],
            'recommended_laptop': rec_row['name'],
            'overall_similarity': self._compute_overall_similarity(query_row, rec_row),
            'feature_comparisons': explanations[:5],
            'key_insight': self._generate_insight(query_row, rec_row, explanations)
        }
    
    def explain_text_query(
        self, 
        query: str, 
        recommended_laptop_id: int
    ) -> Dict:
        """Explain why a laptop matches a text query"""
        # Encode query and laptop
        query_emb = self.encoder.encode_query(query)
        laptop_row = self.df[self.df['laptop_id'] == recommended_laptop_id].iloc[0]
        laptop_text = self.encoder.create_laptop_description(laptop_row)
        laptop_emb = self.encoder.encode_query(laptop_text)
        
        # Token-level attribution (simplified)
        query_tokens = query.lower().split()
        importance_scores = []
        
        for token in query_tokens:
            token_emb = self.encoder.encode_query(token)
            similarity = cosine_similarity(
                token_emb.reshape(1, -1),
                laptop_emb.reshape(1, -1)
            )[0][0]
            importance_scores.append({
                'token': token,
                'relevance_score': round(float(similarity), 3),
                'matched': similarity > 0.5
            })
        
        # Sort by relevance
        importance_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'query': query,
            'laptop_name': laptop_row['name'],
            'top_matching_terms': [t for t in importance_scores if t['matched']][:5],
            'query_intent': self._infer_intent(query),
            'why_recommended': self._generate_text_explanation(query, laptop_row)
        }
    
    def explain_preference_match(
        self,
        preferences: Dict,
        recommended_laptop_id: int
    ) -> Dict:
        """Explain how laptop matches user preferences"""
        laptop = self.df[self.df['laptop_id'] == recommended_laptop_id].iloc[0]
        
        matches = []
        mismatches = []
        
        pref_checks = [
            ('usage_type', preferences.get('usage_type'), 
             lambda p, l: p.lower() in l.lower() if p else True),
            ('max_price', preferences.get('max_price'),
             lambda p, l: l <= p if p else True),
            ('min_ram', preferences.get('min_ram'),
             lambda p, l: l >= p if p else True),
            ('preferred_brand', preferences.get('preferred_brand'),
             lambda p, l: p.lower() == l.lower() if p else True)
        ]
        
        for pref_name, pref_val, check_fn in pref_checks:
            if pref_val is None:
                continue
            
            actual_val = laptop.get(pref_name.replace('preferred_', '').replace('min_', '').replace('max_', ''))
            if pref_name == 'usage_type':
                actual_val = laptop.get('usage_type')
            
            is_match = check_fn(pref_val, actual_val)
            
            result = {
                'preference': pref_name,
                'requested': pref_val,
                'actual': actual_val,
                'matched': is_match
            }
            
            if is_match:
                matches.append(result)
            else:
                mismatches.append(result)
        
        match_score = len(matches) / (len(matches) + len(mismatches)) if (matches or mismatches) else 0
        
        return {
            'match_score': round(match_score, 2),
            'satisfied_preferences': matches,
            'trade_offs': mismatches,
            'overall_assessment': self._assess_trade_offs(matches, mismatches)
        }
    
    def get_global_feature_importance(self) -> Dict:
        """Global explanation of what features matter most"""
        # Analyze embedding dimensions
        sample_size = min(100, len(self.df))
        sample = self.df.sample(sample_size)
        
        embeddings = []
        prices = []
        
        for _, row in sample.iterrows():
            text = self.encoder.create_laptop_description(row)
            emb = self.encoder.encode_query(text)
            embeddings.append(emb)
            prices.append(row['price'])
        
        embeddings = np.array(embeddings)
        
        # Correlation between embedding dims and price
        correlations = []
        for dim in range(embeddings.shape[1]):
            corr = np.corrcoef(embeddings[:, dim], prices)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        # Top influential dimensions
        top_dims = np.argsort(correlations)[-10:][::-1]
        
        return {
            'top_influential_dimensions': top_dims.tolist(),
            'explanation': 'These embedding dimensions correlate most with laptop price/quality',
            'interpretation': 'The model learns to prioritize: brand reputation, GPU performance, CPU generation, RAM amount'
        }
    
    def _compute_overall_similarity(self, row1, row2) -> float:
        """Compute weighted similarity score"""
        weights = {
            'price': 0.2,
            'ram_capacity': 0.15,
            'ssd': 0.1,
            'gpu_vram': 0.25,
            'cpu_cores': 0.15,
            'user_rating': 0.15
        }
        
        score = 0
        for feat, weight in weights.items():
            v1, v2 = row1[feat], row2[feat]
            if pd.isna(v1) or pd.isna(v2):
                continue
            # Normalize similarity
            sim = 1 - abs(v1 - v2) / max(abs(v1), abs(v2), 1)
            score += sim * weight
        
        return round(score, 3)
    
    def _generate_insight(self, query, rec, comparisons) -> str:
        """Generate human-readable insight"""
        top_matches = [c for c in comparisons if c['impact'] == 'high']
        
        if len(top_matches) >= 3:
            return f"Very similar to your selected laptop, matching in {', '.join([t['feature'] for t in top_matches[:3]])}"
        elif rec['price'] < query['price'] * 0.8:
            return f"Budget-friendly alternative with {int((1 - rec['price']/query['price'])*100)}% cost savings"
        elif rec['performance_score'] > query['performance_score'] * 1.2:
            return "Higher performance upgrade with better specs"
        else:
            return "Alternative option with different feature balance"
    
    def _infer_intent(self, query: str) -> str:
        """Infer user intent from query"""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['gaming', 'rtx', 'gtx']):
            return 'gaming_performance'
        elif any(w in query_lower for w in ['cheap', 'budget', 'under', 'below']):
            return 'budget_conscious'
        elif any(w in query_lower for w in ['professional', 'work', 'business']):
            return 'professional_use'
        elif any(w in query_lower for w in ['light', 'portable', 'student']):
            return 'portability'
        else:
            return 'general_purpose'
    
    def _generate_text_explanation(self, query: str, laptop: pd.Series) -> str:
        """Generate natural language explanation"""
        intent = self._infer_intent(query)
        
        explanations = {
            'gaming_performance': f"Matches your gaming needs with {laptop['gpu']} and {int(laptop['ram_capacity'])}GB RAM",
            'budget_conscious': f"Fits your budget at ₹{laptop['price']:,.0f} with good value",
            'professional_use': f"Professional-grade with {laptop['cpu']} and reliable build",
            'portability': f"Portable option with {laptop['screen_size']}\" screen",
            'general_purpose': f"Well-rounded laptop with balanced specs"
        }
        
        return explanations.get(intent, explanations['general_purpose'])
    
    def _assess_trade_offs(self, matches, mismatches) -> str:
        """Assess if trade-offs are acceptable"""
        if not mismatches:
            return "Perfect match for all your preferences"
        elif len(mismatches) == 1:
            return f"Minor trade-off: {mismatches[0]['preference']}"
        elif len(matches) > len(mismatches):
            return "Good overall match with some compromises"
        else:
            return "Several preferences not met - consider adjusting filters"