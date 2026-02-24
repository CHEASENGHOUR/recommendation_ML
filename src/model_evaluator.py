import pandas as pd
import numpy as np
from typing import List, Dict

def evaluate_diversity(recommendations: List[Dict]) -> float:
    """Measure diversity of recommendations (0-1)"""
    if len(recommendations) <= 1:
        return 0.0
    
    brands = [r['brand'] for r in recommendations]
    unique_brands = len(set(brands))
    return unique_brands / len(brands)

def evaluate_price_coverage(recommendations: List[Dict], target_price: float) -> float:
    """How well recommendations cover different price points"""
    prices = [r['price'] for r in recommendations]
    price_variance = np.std(prices) / target_price if target_price > 0 else 0
    return min(price_variance, 1.0)  # Cap at 1

def evaluate_recommendation_quality(
    engine,
    test_cases: List[Dict]
) -> Dict:
    """Evaluate overall recommendation quality"""
    
    results = {
        'avg_similarity': [],
        'diversity_scores': [],
        'price_coverage': []
    }
    
    for test in test_cases:
        recs = engine.get_content_based_recommendations(
            test['laptop_id'], 
            n_recommendations=5
        )
        
        if recs:
            results['avg_similarity'].append(np.mean([r['similarity_score'] for r in recs]))
            results['diversity_scores'].append(evaluate_diversity(recs))
            results['price_coverage'].append(
                evaluate_price_coverage(recs, test.get('price', 50000))
            )
    
    return {
        'mean_similarity': np.mean(results['avg_similarity']) if results['avg_similarity'] else 0,
        'mean_diversity': np.mean(results['diversity_scores']) if results['diversity_scores'] else 0,
        'mean_price_coverage': np.mean(results['price_coverage']) if results['price_coverage'] else 0
    }