"""Data ingestion modules for market prices, fundamentals, earnings, and news."""
from .market_data import fetch_pair_prices
from .fundamentals import fetch_company_metrics, fetch_earnings_history, compute_fundamental_divergence
from .news_data import fetch_ticker_news

__all__ = [
    "fetch_pair_prices",
    "fetch_company_metrics",
    "fetch_earnings_history",
    "compute_fundamental_divergence",
    "fetch_ticker_news",
]
