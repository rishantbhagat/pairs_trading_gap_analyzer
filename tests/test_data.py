"""Integration tests for data ingestion with real or mocked yfinance endpoints."""

from unittest.mock import patch

import pandas as pd

from src.data.market_data import fetch_pair_prices
from src.data.fundamentals import compute_fundamental_divergence

def test_compute_fundamental_divergence():
    metrics_a = {
        "ticker": "KO",
        "forward_pe": 22.0,
        "operating_margin": 0.28,
        "revenue_growth": 0.05,
        "return_on_equity": 0.40,
    }
    metrics_b = {
        "ticker": "PEP",
        "forward_pe": 20.0,
        "operating_margin": 0.16,
        "revenue_growth": 0.04,
        "return_on_equity": 0.50,
    }
    res = compute_fundamental_divergence(metrics_a, metrics_b)
    assert res["pe_diff"] == 2.0
    assert abs(res["op_margin_diff"] - 0.12) < 0.001
    assert "parity_score" in res
    assert "parity_status" in res


def test_fetch_pair_prices_rejects_invalid_pair():
    for ticker_a, ticker_b in [("", "PEP"), ("KO", "KO")]:
        try:
            fetch_pair_prices(ticker_a, ticker_b)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid pair rejection for {ticker_a!r}, {ticker_b!r}")


def test_fetch_pair_prices_handles_multiindex_level_order():
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    fields = ["Open", "High", "Low", "Close"]
    columns = pd.MultiIndex.from_product([["KO", "PEP"], fields], names=["Ticker", "Price"])
    raw = pd.DataFrame(
        [
            [
                value + i
                for base in (100, 80)
                for value in (base - 1, base + 1, base - 2, base)
            ]
            for i in range(len(dates))
        ],
        index=dates,
        columns=columns,
    )

    ticker_info = {
        "shortName": "Company",
        "currentPrice": 125,
        "regularMarketPreviousClose": 124,
        "currency": "USD",
    }
    with (
        patch("src.data.market_data.yf.download", return_value=raw),
        patch(
            "src.data.market_data.yf.Ticker",
            return_value=type("Ticker", (), {"info": ticker_info})(),
        ),
    ):
        prices, metadata = fetch_pair_prices("ko", "pep")

    assert len(prices) == 25
    assert prices["close_KO"].iloc[0] == 100
    assert prices["open_KO"].iloc[0] == 99
    assert prices["high_KO"].iloc[0] == 101
    assert prices["low_PEP"].iloc[0] == 78
    assert metadata["KO"]["short_name"] == "Company"
