"""OpenRouter API client with free model support and deterministic offline fallback."""

import logging
import os
from typing import Any, Dict, List, Optional
import requests

from .prompt_templates import build_investment_memo_prompt

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_ai_trade_summary(
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
    sentiment_stats: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "google/gemini-2.0-flash-exp:free"
) -> Dict[str, Any]:
    """
    Generate an institutional AI investment memo using OpenRouter free models,
    with automatic offline quantitative heuristic fallback if key is missing or fails.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    key = key.strip() if key else ""
    fallback_reason = "api_key_not_configured"
    prompt = build_investment_memo_prompt(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        meta_a=meta_a,
        meta_b=meta_b,
        stats_meta=stats_meta,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        fundamental_diff=fundamental_diff,
        earnings_a=earnings_a,
        earnings_b=earnings_b,
        news_a=news_a,
        news_b=news_b,
        sentiment_stats=sentiment_stats,
    )
    
    # Try OpenRouter if API key is present
    if key and not key.lower().startswith("your_"):
        try:
            headers = {
                "Authorization": f"Bearer {key.strip()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/pairs-gap-analyzer",
                "X-Title": "Pairs Trading Gap Analyzer",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an institutional quantitative equity research director specializing in statistical arbitrage and relative value analysis."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1200,
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=25)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices") if isinstance(data, dict) else None
                first_choice = (
                    choices[0]
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                    else None
                )
                message = first_choice.get("message") if first_choice else None
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str) and content.strip():
                    return {
                        "source": f"OpenRouter ({model_name})",
                        "memo": content.strip(),
                        "prompt": prompt,
                        "status": "success",
                    }
                fallback_reason = "invalid_response_payload"
                logger.warning("OpenRouter returned a successful response without message content.")
            else:
                fallback_reason = f"http_{response.status_code}"
                logger.warning("OpenRouter returned status %s.", response.status_code)
        except (requests.RequestException, ValueError) as e:
            fallback_reason = "request_or_response_error"
            logger.warning("Error calling OpenRouter: %s", e)
    elif key:
        fallback_reason = "placeholder_api_key"
            
    # Fallback: Deterministic Quantitative Heuristic Engine
    heuristic_memo = generate_offline_heuristic_memo(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        meta_a=meta_a,
        meta_b=meta_b,
        stats_meta=stats_meta,
        fundamental_diff=fundamental_diff,
        sentiment_stats=sentiment_stats,
    )
    return {
        "source": "Quantitative Heuristic Engine (Offline Fallback)",
        "memo": heuristic_memo,
        "prompt": prompt,
        "status": "offline_fallback",
        "fallback_reason": fallback_reason,
    }

def generate_offline_heuristic_memo(
    ticker_a: str,
    ticker_b: str,
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    stats_meta: Dict[str, Any],
    fundamental_diff: Dict[str, Any],
    sentiment_stats: Dict[str, Any]
) -> str:
    """
    Generate a quantitative research memo when running offline.
    """
    z = stats_meta.get("current_zscore", 0.0)
    is_coint = stats_meta.get("is_cointegrated", False)
    half_life = stats_meta.get("half_life_days", 30.0)
    beta = stats_meta.get("beta", 1.0)
    regime = stats_meta.get("regime", "Mean-Reverting")
    parity_score = fundamental_diff.get("parity_score", 0.0)
    parity_status = fundamental_diff.get("parity_status", "Aligned")
    
    abs_z = abs(z)
    fundamentals_aligned = abs(parity_score) <= 0.08
    
    if abs_z < 1.5:
        action = f"NEUTRAL / HOLD (In Normal Bounds)"
        action_class = "IN_RANGE"
        thesis = (
            f"The spread Z-score ({z:+.2f}σ) is within normal statistical distribution parameters. "
            f"No compelling risk-adjusted relative-value dislocation is present at this time."
        )
    elif fundamentals_aligned:
        if z <= -1.5:
            action = f"LONG PAIR: Buy {ticker_a} / Short {ticker_b}"
            action_class = "UNWARRANTED STATISTICAL ARBITRAGE GAP"
            thesis = (
                f"{ticker_a} is statistically undervalued relative to {ticker_b} (Z-score: {z:+.2f}σ), "
                f"while fundamental operational metrics ({parity_status}) show consistent parity. "
                f"This presents an asymmetric mean-reversion opportunity."
            )
        else:
            action = f"SHORT PAIR: Short {ticker_a} / Buy {ticker_b}"
            action_class = "UNWARRANTED STATISTICAL ARBITRAGE GAP"
            thesis = (
                f"{ticker_a} is statistically overextended relative to {ticker_b} (Z-score: {z:+.2f}σ), "
                f"unsupported by fundamental operational divergences. "
                f"Statistical mean reversion back toward fair value is expected."
            )
    else:
        action = f"AVOID MEAN REVERSION / STRUCTURAL DRIFT"
        action_class = "JUSTIFIED STRUCTURAL DIVERGENCE"
        thesis = (
            f"While the pricing spread shows a notable dislocation (Z-score: {z:+.2f}σ), "
            f"fundamental decoupling is evident ({parity_status}, Parity Score: {parity_score:+.4f}). "
            f"The price gap is FUNDAMENTALLY JUSTIFIED and fading this move carries high drift risk."
        )

    memo = f"""### 1. Executive Action
- **Signal**: **{action}**
- **Gap Classification**: `{action_class}`
- **Statistical Confidence**: {"High (Cointegrated & OU Validated)" if is_coint else "Moderate (Monitor Cointegration)"}

---
### 2. Gap Diagnosis
{thesis}
- **Current Spread Z-Score**: `{z:+.2f}σ`
- **Estimated Half-Life**: `{half_life} trading days`
- **Market Regime**: `{regime}` (Hurst Exponent: `{stats_meta.get('hurst_exponent', 0.5)}`)
- **Cointegration Test**: `{'Passed (p < 0.05)' if is_coint else 'Not Statistically Significant (p >= 0.05)'}`

---
### 3. Fundamental & Catalyst Alignment
- **Fundamental Operational Status**: `{parity_status}`
- **Valuation Differential (Forward P/E)**: `{fundamental_diff.get('pe_diff', 'N/A')}x`
- **Operating Margin Differential**: `{fundamental_diff.get('op_margin_diff', 'N/A')}`
- **News Sentiment Gap**: `{sentiment_stats.get('sentiment_gap', 0.0):+.2f}` ({sentiment_stats.get('gap_label')})

---
### 4. Execution & Risk Management
- **Hedge Ratio**: Allocate **1.0x {ticker_a}** vs **{beta:.3f}x {ticker_b}** (dollar-adjusted).
- **Target Exit**: Reversion to mean spread ($Z = 0.0$) within an estimated horizon of ~`{half_life}` days.
- **Stop Loss / Invalidation**: Immediate exit if spread exceeds **|Z| >= 3.50σ** or if fundamental parity degrades significantly.
"""
    return memo
