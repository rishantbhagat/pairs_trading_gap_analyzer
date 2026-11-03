"""Feature engineering modules: statistical arbitrage, fundamental disparities, and sentiment."""
from .stats_arb import (
    calculate_ols_spread,
    compute_rolling_zscore,
    test_cointegration,
    estimate_half_life,
    estimate_hurst_exponent,
    build_stats_arb_feature_set,
)
from .fundamental_diff import compute_gap_classification
from .sentiment import analyze_news_sentiment, compute_sentiment_gap
from .support_resistance import calculate_support_resistance_channels

__all__ = [
    "calculate_ols_spread",
    "compute_rolling_zscore",
    "test_cointegration",
    "estimate_half_life",
    "estimate_hurst_exponent",
    "build_stats_arb_feature_set",
    "compute_gap_classification",
    "analyze_news_sentiment",
    "compute_sentiment_gap",
    "calculate_support_resistance_channels",
]
