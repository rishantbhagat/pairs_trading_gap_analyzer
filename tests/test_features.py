"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd

from src.features.stats_arb import (
    calculate_ols_spread,
    compute_rolling_zscore,
    test_cointegration,
    estimate_half_life,
    estimate_hurst_exponent,
    build_stats_arb_feature_set,
)
from src.features.fundamental_diff import compute_gap_classification
from src.features.sentiment import score_headline, compute_sentiment_gap
from src.features.support_resistance import calculate_support_resistance_channels
from src.ui.charts import plot_support_resistance_channels

def test_ols_and_rolling_zscore():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    s2 = pd.Series(100.0 + np.cumsum(np.random.randn(100)), index=dates)
    s1 = pd.Series(1.2 * s2 + 5.0 + np.random.randn(100) * 0.5, index=dates)
    
    beta, alpha, spread = calculate_ols_spread(s1, s2)
    assert abs(beta - 1.2) < 0.2
    assert abs(alpha - 5.0) < 2.0
    
    z_df = compute_rolling_zscore(spread, window=20)
    assert "z_score" in z_df.columns
    assert "upper_band_2" in z_df.columns
    assert "lower_band_2" in z_df.columns
    assert len(z_df) == 100

def test_half_life_and_hurst():
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)
    # Mean-reverting AR(1) process
    spread = [0.0]
    for _ in range(199):
        spread.append(0.8 * spread[-1] + np.random.randn())
    s = pd.Series(spread, index=dates)
    
    hl = estimate_half_life(s)
    assert 1.0 <= hl <= 50.0
    
    hurst = estimate_hurst_exponent(s)
    assert 0.0 <= hurst <= 1.0

def test_gap_classification():
    coint_meta = {"is_cointegrated": True}
    
    # Unwarranted Statistical Dislocation (Long)
    gap_long = compute_gap_classification(
        z_score=-2.5,
        fundamental_divergence={"parity_score": 0.01},
        coint_meta=coint_meta,
        threshold_z=2.0,
    )
    assert gap_long["signal"] == "LONG_PAIR"
    assert gap_long["gap_type"] == "STATISTICAL_ARBITRAGE_GAP"
    
    # Justified Structural Gap (Avoid)
    gap_justified = compute_gap_classification(
        z_score=2.8,
        fundamental_divergence={"parity_score": 0.15},
        coint_meta=coint_meta,
        threshold_z=2.0,
    )
    assert gap_justified["signal"] == "AVOID_REVERSION"
    assert gap_justified["gap_type"] == "JUSTIFIED_STRUCTURAL_GAP"
    
    # Normal In-Range
    gap_normal = compute_gap_classification(
        z_score=0.8,
        fundamental_divergence={"parity_score": 0.02},
        coint_meta=coint_meta,
        threshold_z=2.0,
    )
    assert gap_normal["signal"] == "NEUTRAL"
    assert gap_normal["gap_type"] == "IN_RANGE"

def test_sentiment_scoring():
    bullish = score_headline("Coca-Cola profits surge as revenue beats consensus estimates")
    bearish = score_headline("PepsiCo drops as revenue misses analyst estimates amid lawsuit")
    assert bullish > 0.0
    assert bearish < 0.0


def test_support_resistance_channels_use_confirmed_pivots_and_rank_strength():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    frame = pd.DataFrame(index=dates)
    frame["open_X"] = 100.0
    frame["close_X"] = 100.0
    frame["high_X"] = 101.0
    frame["low_X"] = 99.0
    frame.loc[dates[[5, 15]], "high_X"] = 110.0
    frame.loc[dates[[8, 22]], "low_X"] = 90.0

    channels, pivots = calculate_support_resistance_channels(
        frame,
        "X",
        pivot_period=2,
        channel_width_pct=8,
        min_strength=2,
        loopback=100,
    )

    assert len(channels) == 2
    assert all(channel["pivot_count"] == 2 for channel in channels)
    assert any(channel["upper"] == 110.0 for channel in channels)
    assert {pivot["kind"] for pivot in pivots} == {"high", "low"}
    assert all(pivot["confirmation_index"] == dates.get_loc(pivot["index"]) + 2 for pivot in pivots)
    figure = plot_support_resistance_channels(
        frame, "X", channels, pivots, show_pivots=True, show_breakouts=True
    )
    assert any(trace.type == "candlestick" for trace in figure.data)
    assert len(figure.layout.shapes) == len(channels)


def test_support_resistance_rejects_missing_ohlc():
    frame = pd.DataFrame({"close_X": [100.0] * 10})
    try:
        calculate_support_resistance_channels(frame, "X")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected missing OHLC columns to be rejected.")
