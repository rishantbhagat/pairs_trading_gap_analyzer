"""Fundamental divergence and gap classification analysis."""

from typing import Any, Dict

def compute_gap_classification(
    z_score: float,
    fundamental_divergence: Dict[str, Any],
    coint_meta: Dict[str, Any],
    threshold_z: float = 2.0,
    max_parity_deviation: float = 0.08
) -> Dict[str, Any]:
    """
    Classify whether the current price gap is:
    - An Arbitrage Opportunity (unwarranted statistical dislocation with matching fundamentals)
    - A Justified Divergence (fundamentals justify the price drop/rise; avoid mean reversion)
    - In-Range / Converged (normal fluctuations)
    """
    parity_score = fundamental_divergence.get("parity_score", 0.0)
    is_cointegrated = coint_meta.get("is_cointegrated", False)
    
    abs_z = abs(z_score)
    is_extreme_gap = abs_z >= threshold_z
    fundamentals_aligned = abs(parity_score) <= max_parity_deviation
    
    # Gap analysis logic
    if not is_extreme_gap:
        gap_type = "IN_RANGE"
        signal = "NEUTRAL"
        recommendation = "No substantial price gap. Spread is trading within normal statistical bounds."
        confidence = "Moderate"
    else:
        if fundamentals_aligned:
            gap_type = "STATISTICAL_ARBITRAGE_GAP"
            confidence = "High" if is_cointegrated else "Moderate"
            if z_score <= -threshold_z:
                signal = "LONG_PAIR"
                recommendation = (
                    "Stock A is statistically oversold relative to Stock B while fundamentals remain equivalent. "
                    "Recommend LONG Pair (Buy Stock A / Short Stock B) for mean reversion."
                )
            else:
                signal = "SHORT_PAIR"
                recommendation = (
                    "Stock A is statistically overbought relative to Stock B while fundamentals remain equivalent. "
                    "Recommend SHORT Pair (Short Stock A / Buy Stock B) for mean reversion."
                )
        else:
            # Fundamentals decouple
            gap_type = "JUSTIFIED_STRUCTURAL_GAP"
            confidence = "High"
            signal = "AVOID_REVERSION"
            recommendation = (
                "Price divergence is JUSTIFIED by fundamental operational divergence (e.g. earnings, margins, or guidance). "
                "Do NOT attempt a mean-reversion trade; the spread trend may persist."
            )
            
    return {
        "gap_type": gap_type,
        "signal": signal,
        "recommendation": recommendation,
        "confidence": confidence,
        "current_z": round(z_score, 2),
        "parity_score": round(parity_score, 4),
        "is_cointegrated": is_cointegrated,
        "fundamentals_aligned": fundamentals_aligned,
    }
