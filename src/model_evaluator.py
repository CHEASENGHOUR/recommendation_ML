from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def evaluate_diversity(recommendations: List[Dict]) -> float:
    """
    Brand diversity of the recommendation list.
    Returns the fraction of unique brands (0 = all same brand, 1 = all different).
    """
    if len(recommendations) <= 1:
        return 0.0
    brands = [r.get("brand", "") for r in recommendations]
    return len(set(brands)) / len(brands)


def evaluate_price_coverage(
    recommendations: List[Dict],
    target_price:    float,
) -> float:
    """
    How broadly the recommendations spread across price points
    relative to the target price.  Capped at 1.0.
    """
    if not recommendations or target_price <= 0:
        return 0.0
    prices        = [r.get("price", 0) for r in recommendations]
    price_std     = float(np.std(prices))
    coverage      = price_std / target_price
    return min(coverage, 1.0)


def evaluate_avg_similarity(recommendations: List[Dict]) -> float:
    """Mean cosine similarity score of the recommendation list."""
    if not recommendations:
        return 0.0
    scores = [r.get("similarity_score", 0.0) for r in recommendations]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------

def evaluate_recommendation_quality(
    engine,
    test_cases: List[Dict],
    n:          int = 5,
) -> Dict:
    """
    Run the engine on *test_cases* and aggregate quality metrics.

    Parameters
    ----------
    engine     : LaptopRecommendationEngine (already fitted)
    test_cases : list of dicts with keys ``laptop_id`` and ``price``
    n          : number of recommendations to request per test case

    Returns
    -------
    dict with mean_similarity, mean_diversity, mean_price_coverage
    """
    avg_similarities: List[float] = []
    diversity_scores: List[float] = []
    price_coverages:  List[float] = []

    for case in test_cases:
        laptop_id = case.get("laptop_id")
        price     = case.get("price", 50_000)

        # FIX: was get_content_based_recommendations() — correct name is get_similar_laptops()
        recs = engine.get_similar_laptops(laptop_id, n=n)

        if not recs:
            continue

        avg_similarities.append(evaluate_avg_similarity(recs))
        diversity_scores.append(evaluate_diversity(recs))
        price_coverages.append(evaluate_price_coverage(recs, price))

    def _safe_mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    results = {
        "mean_similarity":    round(_safe_mean(avg_similarities), 4),
        "mean_diversity":     round(_safe_mean(diversity_scores),  4),
        "mean_price_coverage":round(_safe_mean(price_coverages),   4),
        "n_test_cases":       len(test_cases),
        "n_evaluated":        len(avg_similarities),
    }

    print(f"[evaluator] mean_similarity={results['mean_similarity']} | "
          f"diversity={results['mean_diversity']} | "
          f"price_coverage={results['mean_price_coverage']}")
    return results