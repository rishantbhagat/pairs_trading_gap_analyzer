"""Fundamental data ingestion: financial metrics, quarterly earnings, and peer divergences."""

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_company_metrics(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive fundamental valuation and operating metrics.
    """
    sym = ticker.strip().upper()
    t = yf.Ticker(sym)
    info = t.info or {}
    
    def safe_get(key: str, default: Any = None, round_digits: Optional[int] = 2):
        val = info.get(key)
        if val is None or val == "None" or val == "Infinity" or val == "-Infinity":
            return default
        try:
            val_f = float(val)
            if np.isnan(val_f) or np.isinf(val_f):
                return default
            return round(val_f, round_digits) if round_digits is not None else val_f
        except (ValueError, TypeError):
            return default

    # Extract metrics
    metrics = {
        "ticker": sym,
        "trailing_pe": safe_get("trailingPE"),
        "forward_pe": safe_get("forwardPE"),
        "peg_ratio": safe_get("pegRatio"),
        "ev_to_ebitda": safe_get("enterpriseToEbitda"),
        "price_to_book": safe_get("priceToBook"),
        "operating_margin": safe_get("operatingMargins", round_digits=4),
        "profit_margin": safe_get("profitMargins", round_digits=4),
        "return_on_equity": safe_get("returnOnEquity", round_digits=4),
        "revenue_growth": safe_get("revenueGrowth", round_digits=4),
        "earnings_growth": safe_get("earningsGrowth", round_digits=4),
        "dividend_yield": safe_get("dividendYield", round_digits=4),
        "debt_to_equity": safe_get("debtToEquity"),
        "free_cashflow": safe_get("freeCashflow", round_digits=0),
        "target_mean_price": safe_get("targetMeanPrice"),
        "recommendation_key": info.get("recommendationKey", "N/A"),
    }
    return metrics

def fetch_earnings_history(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly historical earnings reports (Actual EPS, Estimated EPS, Surprise %).
    """
    sym = ticker.strip().upper()
    t = yf.Ticker(sym)
    
    records = []
    try:
        # Check earnings_dates if available
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            ed_clean = ed.dropna(subset=["Reported EPS"]).head(8)
            for idx, row in ed_clean.iterrows():
                date_str = str(idx).split(" ")[0] if hasattr(idx, "strftime") else str(idx)[:10]
                eps_est = row.get("EPS Estimate")
                eps_act = row.get("Reported EPS")
                surp = row.get("Surprise(%)")
                
                records.append({
                    "date": date_str,
                    "eps_estimate": round(float(eps_est), 2) if pd.notna(eps_est) else None,
                    "eps_actual": round(float(eps_act), 2) if pd.notna(eps_act) else None,
                    "surprise_pct": round(float(surp), 2) if pd.notna(surp) else None,
                })
    except Exception as e:
        logger.warning(f"Error fetching earnings dates for {sym}: {e}")
        
    df = pd.DataFrame(records)
    if df.empty:
        # Create empty placeholder DataFrame with schema
        df = pd.DataFrame(columns=["date", "eps_estimate", "eps_actual", "surprise_pct"])
    return df

def compute_fundamental_divergence(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate comparative fundamental differentials and composite parity score.
    """
    sym_a = metrics_a.get("ticker", "A")
    sym_b = metrics_b.get("ticker", "B")
    
    def diff(key: str) -> Optional[float]:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is not None and val_b is not None:
            return round(float(val_a - val_b), 4)
        return None

    pe_diff = diff("forward_pe") or diff("trailing_pe")
    ev_ebitda_diff = diff("ev_to_ebitda")
    op_margin_diff = diff("operating_margin")
    rev_growth_diff = diff("revenue_growth")
    roe_diff = diff("return_on_equity")
    
    # Calculate Composite Fundamental Parity Score
    # Values close to 0 denote near-identical operational performance
    # Positive scores imply A is fundamentally outperforming B
    # Negative scores imply B is fundamentally outperforming A
    components = []
    if op_margin_diff is not None:
        components.append(op_margin_diff * 2.0)  # Margins carry high weight
    if rev_growth_diff is not None:
        components.append(rev_growth_diff * 2.0)
    if roe_diff is not None:
        components.append(roe_diff * 1.0)
        
    parity_score = round(float(np.mean(components)), 4) if components else 0.0
    
    # Determine parity status:
    # If absolute parity score is small (< 0.05 / 5%), fundamentals are essentially tracking each other.
    if abs(parity_score) < 0.05:
        parity_status = "High Parity (Fundamentals Tracking)"
    elif parity_score > 0:
        parity_status = f"{sym_a} Fundamentally Outperforming {sym_b}"
    else:
        parity_status = f"{sym_b} Fundamentally Outperforming {sym_a}"

    return {
        "pe_diff": pe_diff,
        "ev_ebitda_diff": ev_ebitda_diff,
        "op_margin_diff": op_margin_diff,
        "rev_growth_diff": rev_growth_diff,
        "roe_diff": roe_diff,
        "parity_score": parity_score,
        "parity_status": parity_status,
    }
