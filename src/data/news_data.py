"""News data ingestion using yfinance news feeds."""

from datetime import datetime
import logging
from typing import Any, Dict, List
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_ticker_news(ticker: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch and clean recent news articles for a given ticker.
    """
    sym = ticker.strip().upper()
    t = yf.Ticker(sym)
    news_items = []
    
    try:
        raw_news = t.news or []
        for item in raw_news[:limit]:
            title = item.get("title", "")
            if not title:
                continue
                
            pub_time_raw = item.get("providerPublishTime")
            if pub_time_raw:
                try:
                    pub_date = datetime.fromtimestamp(pub_time_raw).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pub_date = "Recent"
            else:
                pub_date = "Recent"
                
            publisher = item.get("publisher", "Financial News")
            link = item.get("link", "")
            
            news_items.append({
                "ticker": sym,
                "title": title,
                "publisher": publisher,
                "publish_date": pub_date,
                "link": link,
            })
    except Exception as e:
        logger.warning(f"Error fetching news for {sym}: {e}")
        
    return news_items

def get_combined_news_digest(ticker_a: str, ticker_b: str, limit_each: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch and group news for both tickers in a pair.
    """
    return {
        ticker_a: fetch_ticker_news(ticker_a, limit=limit_each),
        ticker_b: fetch_ticker_news(ticker_b, limit=limit_each),
    }
