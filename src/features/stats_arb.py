"""Quantitative Statistical Arbitrage Feature Engineering:
OLS Hedge Ratio, Cointegration, Rolling Z-Scores, Half-Life, and Hurst Exponent.
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_ols_spread(
    s1: pd.Series,
    s2: pd.Series
) -> Tuple[float, float, pd.Series]:
    """
    Fit Ordinary Least Squares (OLS) regression: S1 = beta * S2 + alpha.
    
    Returns:
        beta: Dynamic hedge ratio
        alpha: Spread intercept
        spread: Residual time series (S1 - beta * S2 - alpha)
    """
    # Clean aligned series
    aligned = pd.concat([s1, s2], axis=1).dropna()
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    beta = float(slope)
    alpha = float(intercept)
    
    spread = s1 - (beta * s2 + alpha)
    spread.name = "spread"
    return beta, alpha, spread

def compute_rolling_zscore(
    spread: pd.Series,
    window: int = 30
) -> pd.DataFrame:
    """
    Compute rolling mean, standard deviation, Bollinger divergence bands, and Z-scores.
    """
    rolling_mean = spread.rolling(window=window, min_periods=max(5, window // 2)).mean()
    rolling_std = spread.rolling(window=window, min_periods=max(5, window // 2)).std()
    
    # Avoid zero division
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.fillna(0.0)
    
    df = pd.DataFrame(index=spread.index)
    df["spread"] = spread
    df["rolling_mean"] = rolling_mean
    df["rolling_std"] = rolling_std
    df["z_score"] = zscore
    df["upper_band_1"] = rolling_mean + 1.0 * rolling_std
    df["lower_band_1"] = rolling_mean - 1.0 * rolling_std
    df["upper_band_2"] = rolling_mean + 2.0 * rolling_std
    df["lower_band_2"] = rolling_mean - 2.0 * rolling_std
    
    return df

def test_cointegration(s1: pd.Series, s2: pd.Series) -> Dict[str, Any]:
    """
    Perform Engle-Granger cointegration test.
    """
    try:
        from statsmodels.tsa.stattools import coint
        aligned = pd.concat([s1, s2], axis=1).dropna()
        score, p_value, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return {
            "test_stat": round(float(score), 4),
            "p_value": round(float(p_value), 4),
            "is_cointegrated": bool(p_value < 0.05),
            "confidence_level": "95%+" if p_value < 0.05 else ("90%+" if p_value < 0.10 else "Low"),
        }
    except Exception as e:
        logger.warning(f"Error in cointegration test: {e}")
        # Fallback OLS residual ADF approximation
        try:
            from statsmodels.tsa.stattools import adfuller
            _, _, spread = calculate_ols_spread(s1, s2)
            res = adfuller(spread.dropna())
            p_val = float(res[1])
            return {
                "test_stat": round(float(res[0]), 4),
                "p_value": round(p_val, 4),
                "is_cointegrated": bool(p_val < 0.05),
                "confidence_level": "95%+" if p_val < 0.05 else "Low",
            }
        except Exception:
            return {
                "test_stat": 0.0,
                "p_value": 0.50,
                "is_cointegrated": False,
                "confidence_level": "N/A",
            }

def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life (in trading days) via Ornstein-Uhlenbeck process.
    Spread_diff_t = theta * Spread_{t-1} + e
    Half-life = -ln(2) / theta
    """
    clean_spread = spread.dropna()
    if len(clean_spread) < 15:
        return 30.0
        
    spread_lag = clean_spread.shift(1).iloc[1:].values
    spread_diff = (clean_spread - clean_spread.shift(1)).iloc[1:].values
    
    slope, intercept, _, _, _ = stats.linregress(spread_lag, spread_diff)
    theta = float(slope)
    
    if theta >= 0:
        # Non mean-reverting process
        return 999.0
        
    half_life = -np.log(2.0) / theta
    return round(float(min(max(half_life, 1.0), 250.0)), 1)

def estimate_hurst_exponent(ts: pd.Series, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent.
    H < 0.5: Mean-reverting regime
    H = 0.5: Geometric random walk
    H > 0.5: Persistent / trending regime
    """
    clean = ts.dropna().values
    if len(clean) < max_lag * 2:
        return 0.5
        
    lags = range(2, max_lag)
    tau = [np.std(clean[lag:] - clean[:-lag]) for lag in lags]
    
    # Linear fit of log(tau) on log(lags)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = float(poly[0])
    return round(float(min(max(hurst, 0.0), 1.0)), 3)

def build_stats_arb_feature_set(
    df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 30
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct complete time-series feature matrix for the pair.
    """
    close_a = df[f"close_{ticker_a}"]
    close_b = df[f"close_{ticker_b}"]
    
    # 1. OLS Spread
    beta, alpha, spread = calculate_ols_spread(close_a, close_b)
    
    # 2. Rolling Z-Score and Bollinger bands
    z_df = compute_rolling_zscore(spread, window=window)
    
    # 3. Log Price Ratio
    log_ratio = np.log(close_a / close_b)
    
    # 4. Statistical properties
    coint_res = test_cointegration(close_a, close_b)
    half_life = estimate_half_life(spread)
    hurst = estimate_hurst_exponent(spread)
    
    # Assemble feature DataFrame
    feat_df = df.copy()
    feat_df["spread"] = spread
    feat_df["z_score"] = z_df["z_score"]
    feat_df["rolling_mean"] = z_df["rolling_mean"]
    feat_df["rolling_std"] = z_df["rolling_std"]
    feat_df["upper_band_1"] = z_df["upper_band_1"]
    feat_df["lower_band_1"] = z_df["lower_band_1"]
    feat_df["upper_band_2"] = z_df["upper_band_2"]
    feat_df["lower_band_2"] = z_df["lower_band_2"]
    feat_df["log_ratio"] = log_ratio
    
    # Spread velocity / momentum
    feat_df["spread_velocity_5d"] = spread.diff(5).fillna(0.0)
    feat_df["spread_velocity_10d"] = spread.diff(10).fillna(0.0)
    feat_df["volatility_ratio"] = (feat_df[f"ret_{ticker_a}"].rolling(window).std() / 
                                   feat_df[f"ret_{ticker_b}"].rolling(window).std().replace(0, np.nan)).fillna(1.0)
                                   
    stats_meta = {
        "beta": round(beta, 4),
        "alpha": round(alpha, 4),
        "coint_p_value": coint_res["p_value"],
        "is_cointegrated": coint_res["is_cointegrated"],
        "confidence_level": coint_res["confidence_level"],
        "half_life_days": half_life,
        "hurst_exponent": hurst,
        "regime": "Mean-Reverting" if hurst < 0.45 else ("Trending" if hurst > 0.55 else "Random Walk"),
        "current_zscore": round(float(feat_df["z_score"].iloc[-1]), 2),
        "current_spread": round(float(feat_df["spread"].iloc[-1]), 2),
    }
    return feat_df, stats_meta
