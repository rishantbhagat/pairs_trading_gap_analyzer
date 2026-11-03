"""Financial headline sentiment analysis and sentiment divergence calculation."""

import re
from typing import Any, Dict, List

# Financial lexicon
BULLISH_KEYWORDS = {
    "surge", "surges", "surged", "jump", "jumps", "jumped", "beat", "beats", "beating",
    "profit", "profits", "profitable", "growth", "growing", "gains", "gained", "record",
    "outperform", "outperformed", "bullish", "rally", "rallies", "dividend", "hiked",
    "upgrade", "upgraded", "higher", "strong", "boost", "boosts", "soars", "soared",
    "expansion", "expanding", "acquisition", "winner", "positive", "exceeds", "exceeded",
}

BEARISH_KEYWORDS = {
    "drop", "drops", "dropped", "plunge", "plunges", "plunged", "miss", "misses", "missed",
    "loss", "losses", "falling", "falls", "fell", "slump", "slumps", "slumped", "down",
    "underperform", "bearish", "decline", "declines", "declining", "cut", "cuts", "cutting",
    "downgrade", "downgraded", "lawsuit", "investigation", "probe", "recall", "warning",
    "recession", "weak", "weakness", "inflation", "tariff", "tariffs", "struggle", "crisis",
}

def score_headline(text: str) -> float:
    """
    Score a single headline between -1.0 (extremely bearish) and +1.0 (extremely bullish).
    """
    if not text:
        return 0.0
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
        
    pos_count = sum(1 for w in words if w in BULLISH_KEYWORDS)
    neg_count = sum(1 for w in words if w in BEARISH_KEYWORDS)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
        
    score = (pos_count - neg_count) / float(total)
    return round(score, 3)

def analyze_news_sentiment(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate sentiment metrics for a list of news items.
    """
    if not news_items:
        return {
            "average_sentiment": 0.0,
            "sentiment_label": "Neutral",
            "bullish_articles": 0,
            "bearish_articles": 0,
            "neutral_articles": 0,
            "article_count": 0,
        }
        
    scores = []
    bullish = 0
    bearish = 0
    neutral = 0
    
    for item in news_items:
        s = score_headline(item.get("title", ""))
        item["sentiment_score"] = s
        scores.append(s)
        if s > 0.15:
            bullish += 1
        elif s < -0.15:
            bearish += 1
        else:
            neutral += 1
            
    avg_score = round(float(sum(scores) / len(scores)), 3) if scores else 0.0
    
    if avg_score > 0.15:
        label = "Bullish"
    elif avg_score < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"
        
    return {
        "average_sentiment": avg_score,
        "sentiment_label": label,
        "bullish_articles": bullish,
        "bearish_articles": bearish,
        "neutral_articles": neutral,
        "article_count": len(news_items),
    }

def compute_sentiment_gap(
    news_a: List[Dict[str, Any]],
    news_b: List[Dict[str, Any]],
    ticker_a: str,
    ticker_b: str
) -> Dict[str, Any]:
    """
    Calculate the relative sentiment gap between the two peer tickers.
    """
    stats_a = analyze_news_sentiment(news_a)
    stats_b = analyze_news_sentiment(news_b)
    
    gap = round(float(stats_a["average_sentiment"] - stats_b["average_sentiment"]), 3)
    
    if abs(gap) < 0.10:
        gap_label = "Balanced Sentiment"
    elif gap > 0:
        gap_label = f"Sentiment favors {ticker_a} (+{gap})"
    else:
        gap_label = f"Sentiment favors {ticker_b} ({gap})"
        
    return {
        "ticker_a_sentiment": stats_a,
        "ticker_b_sentiment": stats_b,
        "sentiment_gap": gap,
        "gap_label": gap_label,
    }
