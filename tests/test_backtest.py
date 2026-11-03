"""Unit tests for the Pairs Backtester."""

import numpy as np
import pandas as pd

from src.ml.backtester import PairsBacktester, run_comparative_backtest

def test_pairs_backtester_simulation():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    
    close_a = 100.0 * (1.0 + np.random.randn(100) * 0.01).cumprod()
    close_b = 80.0 * (1.0 + np.random.randn(100) * 0.01).cumprod()
    
    df = pd.DataFrame(index=dates)
    df["close_A"] = close_a
    df["close_B"] = close_b
    df["ret_A"] = pd.Series(close_a).pct_change().fillna(0.0).values
    df["ret_B"] = pd.Series(close_b).pct_change().fillna(0.0).values
    
    # Alternating signals
    signals = pd.Series([1]*25 + [0]*25 + [-1]*25 + [0]*25, index=dates)
    
    bt = PairsBacktester(initial_capital=100_000.0, transaction_cost_bps=5.0)
    res = bt.simulate(df, signals, "A", "B", beta=1.0, strategy_name="Test Strategy")
    
    assert res["strategy_name"] == "Test Strategy"
    assert "metrics" in res
    assert "equity_curve" in res
    assert len(res["equity_curve"]) == 100
    assert "total_return_pct" in res["metrics"]
    assert "sharpe_ratio" in res["metrics"]
    assert "max_drawdown_pct" in res["metrics"]


def test_backtester_does_not_charge_inactive_initial_position():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "close_A": [100.0, 100.0, 100.0],
            "close_B": [80.0, 80.0, 80.0],
            "ret_A": [0.0, 0.0, 0.0],
            "ret_B": [0.0, 0.0, 0.0],
        },
        index=dates,
    )
    signals = pd.Series(0, index=dates)

    result = PairsBacktester(transaction_cost_bps=5.0).simulate(
        df, signals, "A", "B", beta=1.0
    )

    assert result["metrics"]["total_return_pct"] == 0.0
    assert result["equity_curve"]["equity"].iloc[-1] == 100_000.0


def test_backtester_charges_both_legs_on_entry_and_exit():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    df = pd.DataFrame(
        {
            "close_A": [100.0] * 4,
            "close_B": [80.0] * 4,
            "ret_A": [0.0] * 4,
            "ret_B": [0.0] * 4,
        },
        index=dates,
    )
    signals = pd.Series([1, 1, 0, 0], index=dates)

    result = PairsBacktester(transaction_cost_bps=5.0).simulate(
        df, signals, "A", "B", beta=1.0
    )

    assert np.isclose(result["metrics"]["total_return_pct"], -0.2, atol=0.01)
