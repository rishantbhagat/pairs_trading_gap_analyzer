"""Unit tests for AI prompt construction and offline heuristic generator."""

from unittest.mock import patch

from src.ai.prompt_templates import build_investment_memo_prompt
from src.ai.openrouter_client import (
    generate_ai_trade_summary,
    generate_offline_heuristic_memo,
)

def test_prompt_and_offline_heuristic():
    meta_a = {"current_price": 60.5, "short_name": "Coca-Cola"}
    meta_b = {"current_price": 165.2, "short_name": "PepsiCo"}
    stats_meta = {
        "current_zscore": -2.3,
        "beta": 0.35,
        "coint_p_value": 0.02,
        "is_cointegrated": True,
        "half_life_days": 18.5,
        "hurst_exponent": 0.38,
        "regime": "Mean-Reverting",
    }
    metrics_a = {"forward_pe": 22.5, "operating_margin": 0.28, "revenue_growth": 0.06}
    metrics_b = {"forward_pe": 20.8, "operating_margin": 0.15, "revenue_growth": 0.04}
    fundamental_diff = {"pe_diff": 1.7, "op_margin_diff": 0.13, "rev_growth_diff": 0.02, "parity_score": 0.02, "parity_status": "High Parity"}
    earnings_a = [{"date": "2024-07-23", "eps_actual": 0.84, "eps_estimate": 0.81, "surprise_pct": 3.7}]
    earnings_b = [{"date": "2024-07-11", "eps_actual": 2.28, "eps_estimate": 2.16, "surprise_pct": 5.5}]
    news_a = [{"title": "Coca-Cola reports strong Q2 earnings beat", "publisher": "Reuters"}]
    news_b = [{"title": "PepsiCo raises annual dividend", "publisher": "Bloomberg"}]
    sentiment_stats = {"sentiment_gap": 0.15, "gap_label": "Sentiment favors KO"}
    
    prompt = build_investment_memo_prompt(
        ticker_a="KO",
        ticker_b="PEP",
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
    assert "KO" in prompt and "PEP" in prompt
    assert "-2.3" in prompt
    assert "Executive Action" in prompt
    
    memo = generate_offline_heuristic_memo(
        ticker_a="KO",
        ticker_b="PEP",
        meta_a=meta_a,
        meta_b=meta_b,
        stats_meta=stats_meta,
        fundamental_diff=fundamental_diff,
        sentiment_stats=sentiment_stats,
    )
    assert "LONG PAIR" in memo
    assert "STATISTICAL ARBITRAGE GAP" in memo


def test_openrouter_request_auth_and_response_validation():
    kwargs = {
        "ticker_a": "AAA",
        "ticker_b": "BBB",
        "meta_a": {},
        "meta_b": {},
        "stats_meta": {},
        "metrics_a": {},
        "metrics_b": {},
        "fundamental_diff": {},
        "earnings_a": [],
        "earnings_b": [],
        "news_a": [],
        "news_b": [],
        "sentiment_stats": {},
        "api_key": "  test-key  ",
    }

    success_response = type(
        "Response",
        (),
        {
            "status_code": 200,
            "json": lambda self: {"choices": [{"message": {"content": "Memo"}}]},
        },
    )()
    with patch(
        "src.ai.openrouter_client.requests.post", return_value=success_response
    ) as post:
        result = generate_ai_trade_summary(**kwargs)

    assert result["status"] == "success"
    assert result["memo"] == "Memo"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"

    invalid_response = type(
        "Response", (), {"status_code": 200, "json": lambda self: {"choices": []}}
    )()
    with patch(
        "src.ai.openrouter_client.requests.post", return_value=invalid_response
    ):
        fallback = generate_ai_trade_summary(**kwargs)

    assert fallback["status"] == "offline_fallback"
    assert fallback["fallback_reason"] == "invalid_response_payload"
