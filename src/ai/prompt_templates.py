"""Prompt templates for institutional-grade pairs trading synthesis."""

from typing import Any, Dict, List

def build_investment_memo_prompt(
    ticker_a: str,
    ticker_b: str,
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    stats_meta: Dict[str, Any],
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    fundamental_diff: Dict[str, Any],
    earnings_a: List[Dict[str, Any]],
    earnings_b: List[Dict[str, Any]],
    news_a: List[Dict[str, Any]],
    news_b: List[Dict[str, Any]],
    sentiment_stats: Dict[str, Any]
) -> str:
    """
    Construct a prompt for OpenRouter LLMs.
    """
    # Recent earnings summaries
    latest_eps_a = earnings_a[0] if earnings_a else {}
    latest_eps_b = earnings_b[0] if earnings_b else {}
    
    # Headlines preview
    headlines_a_str = "\n".join([f"  - {n.get('title')} ({n.get('publisher')})" for n in news_a[:4]]) or "  - No recent headlines"
    headlines_b_str = "\n".join([f"  - {n.get('title')} ({n.get('publisher')})" for n in news_b[:4]]) or "  - No recent headlines"
    
    prompt = f"""You are a Senior Quantitative Portfolio Manager and Equity Arbitrage Strategist.
Analyze the following relative valuation gap between two category peers: {ticker_a} and {ticker_b}.

Your objective is to determine whether any observed price divergence constitutes:
1. An UNWARRANTED STATISTICAL ARBITRAGE GAP (mean-reverting trading opportunity: buy the undervalued peer, sell/short the overvalued peer).
2. A JUSTIFIED STRUCTURAL DECOUPLING (where underlying fundamentals, earnings, or catalysts explain the price drop; mean-reversion should be avoided).

---
### 1. MARKET PRICING & STATISTICAL ARBITRAGE METRICS:
- Pair: {ticker_a} (${meta_a.get('current_price')}) vs {ticker_b} (${meta_b.get('current_price')})
- Current Spread Z-Score: {stats_meta.get('current_zscore')} σ
- OLS Hedge Ratio (Beta): {stats_meta.get('beta')} (Stock A = Beta * Stock B + Alpha)
- Engle-Granger Cointegration: p-value = {stats_meta.get('coint_p_value')} (Cointegrated: {stats_meta.get('is_cointegrated')})
- Mean-Reversion Half-Life: {stats_meta.get('half_life_days')} trading days
- Hurst Exponent: {stats_meta.get('hurst_exponent')} (Regime: {stats_meta.get('regime')})

---
### 2. FUNDAMENTAL RELATIVE VALUATION & EARNINGS SURPRISES:
- Forward P/E: {ticker_a} = {metrics_a.get('forward_pe', 'N/A')}x vs {ticker_b} = {metrics_b.get('forward_pe', 'N/A')}x (Diff: {fundamental_diff.get('pe_diff')}x)
- Operating Margin: {ticker_a} = {metrics_a.get('operating_margin', 'N/A')} vs {ticker_b} = {metrics_b.get('operating_margin', 'N/A')} (Diff: {fundamental_diff.get('op_margin_diff')})
- YoY Revenue Growth: {ticker_a} = {metrics_a.get('revenue_growth', 'N/A')} vs {ticker_b} = {metrics_b.get('revenue_growth', 'N/A')} (Diff: {fundamental_diff.get('rev_growth_diff')})
- Composite Fundamental Parity Status: {fundamental_diff.get('parity_status')} (Score: {fundamental_diff.get('parity_score')})
- Latest Reported EPS vs Estimate:
  * {ticker_a}: Reported ${latest_eps_a.get('eps_actual', 'N/A')} vs Est ${latest_eps_a.get('eps_estimate', 'N/A')} (Surprise: {latest_eps_a.get('surprise_pct', 'N/A')}%)
  * {ticker_b}: Reported ${latest_eps_b.get('eps_actual', 'N/A')} vs Est ${latest_eps_b.get('eps_estimate', 'N/A')} (Surprise: {latest_eps_b.get('surprise_pct', 'N/A')}%)

---
### 3. RECENT NEWS & CATALYST DIGEST:
- Relative News Sentiment Gap: {sentiment_stats.get('sentiment_gap')} ({sentiment_stats.get('gap_label')})
- Top Headlines for {ticker_a}:
{headlines_a_str}
- Top Headlines for {ticker_b}:
{headlines_b_str}

---
### REQUIRED OUTPUT FORMAT:
Provide a rigorous, institutional investment brief formatted in Markdown with these specific sections:
1. **Executive Action**: Explicit Signal: [LONG PAIR (Buy {ticker_a} / Sell {ticker_b}), SHORT PAIR (Sell {ticker_a} / Buy {ticker_b}), or AVOID / NEUTRAL] with Confidence (High/Medium/Low).
2. **Gap Diagnosis**: Explain whether this is a temporary statistical dislocation or a fundamentally justified structural repricing.
3. **Fundamental & Catalyst Alignment**: Discuss how the quarterly earnings surprise, margin disparities, and recent news feed support or contradict the signal.
4. **Execution & Risk Management**: State entry criteria, expected convergence horizon (referencing half-life), and explicit invalidation / stop-loss conditions.
"""
    return prompt
