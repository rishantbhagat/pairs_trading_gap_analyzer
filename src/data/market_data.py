"""Market data ingestion using yfinance."""

import logging
from typing import Dict, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_pair_prices(
    ticker_a: str,
    ticker_b: str,
    period: str = "2y",
    interval: str = "1d"
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Fetch historical and current price data for a pair of tickers.
    
    Returns:
        df: DataFrame indexed by Date with columns:
            [close_a, close_b, norm_a, norm_b, return_a, return_b]
        meta: Metadata dict containing latest prices, currency, company names, etc.
    """
    ticker_a = ticker_a.strip().upper()
    ticker_b = ticker_b.strip().upper()
    if not ticker_a or not ticker_b:
        raise ValueError("Both ticker symbols must be non-empty.")
    if ticker_a == ticker_b:
        raise ValueError("Ticker symbols must be different.")
    
    # Download synchronized data
    try:
        raw = yf.download(
            tickers=[ticker_a, ticker_b],
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        logger.exception("Failed to download price history for %s and %s.", ticker_a, ticker_b)
        raise RuntimeError(
            f"Market data request failed for '{ticker_a}' and '{ticker_b}'."
        ) from exc
    
    if raw.empty:
        raise ValueError(f"No price data returned for tickers '{ticker_a}' and '{ticker_b}'.")
        
    def extract_field(field: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            for level in range(raw.columns.nlevels):
                if field in raw.columns.get_level_values(level):
                    extracted = raw.xs(field, axis=1, level=level)
                    if isinstance(extracted, pd.Series):
                        return extracted.to_frame()
                    return extracted
            return pd.DataFrame(index=raw.index)
        if field in raw.columns:
            return raw[[field]]
        return pd.DataFrame(index=raw.index)

    # yfinance may place ticker and OHLC labels in either MultiIndex level.
    close_df = extract_field("Close")
        
    if ticker_a not in close_df.columns or ticker_b not in close_df.columns:
        raise ValueError(f"Failed to find Close prices for both '{ticker_a}' and '{ticker_b}' in data.")
        
    df = pd.DataFrame(index=close_df.index)
    df[f"close_{ticker_a}"] = close_df[ticker_a]
    df[f"close_{ticker_b}"] = close_df[ticker_b]

    for field in ("Open", "High", "Low"):
        field_df = extract_field(field)
        for symbol in (ticker_a, ticker_b):
            column = f"{field.lower()}_{symbol}"
            df[column] = field_df[symbol] if symbol in field_df else df[f"close_{symbol}"]
    
    # Clean data: drop NaNs and ensure chronological order
    df = df.dropna().sort_index()
    
    if len(df) < 20:
        raise ValueError(f"Insufficient overlapping trading days ({len(df)}) for '{ticker_a}' and '{ticker_b}'.")
        
    # Normalized prices (Base 100 at start of period)
    df[f"norm_{ticker_a}"] = (df[f"close_{ticker_a}"] / df[f"close_{ticker_a}"].iloc[0]) * 100.0
    df[f"norm_{ticker_b}"] = (df[f"close_{ticker_b}"] / df[f"close_{ticker_b}"].iloc[0]) * 100.0
    
    # Daily returns
    df[f"ret_{ticker_a}"] = df[f"close_{ticker_a}"].pct_change().fillna(0.0)
    df[f"ret_{ticker_b}"] = df[f"close_{ticker_b}"].pct_change().fillna(0.0)
    
    # Fetch ticker summary metadata
    meta = {}
    for sym in [ticker_a, ticker_b]:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            curr_price = info.get("currentPrice") or info.get("regularMarketPrice") or float(df[f"close_{sym}"].iloc[-1])
            prev_close = info.get("regularMarketPreviousClose") or (float(df[f"close_{sym}"].iloc[-2]) if len(df) > 1 else curr_price)
            chg_pct = ((curr_price - prev_close) / prev_close) * 100.0 if prev_close else 0.0
            
            meta[sym] = {
                "short_name": info.get("shortName") or sym,
                "sector": info.get("sector") or "N/A",
                "industry": info.get("industry") or "N/A",
                "current_price": round(float(curr_price), 2),
                "change_pct": round(float(chg_pct), 2),
                "currency": info.get("currency") or "USD",
                "market_cap": info.get("marketCap", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch detailed info for {sym}: {e}")
            curr_price = float(df[f"close_{sym}"].iloc[-1])
            meta[sym] = {
                "short_name": sym,
                "sector": "N/A",
                "industry": "N/A",
                "current_price": round(curr_price, 2),
                "change_pct": 0.0,
                "currency": "USD",
                "market_cap": 0,
            }
            
    return df, meta
