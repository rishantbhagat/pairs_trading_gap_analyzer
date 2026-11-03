"""Production-Grade Vectorized Pairs Trading Backtester.
Supports In-Sample (Train) vs. Out-of-Sample (Test) Partitioning and Transaction Costs.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PairsBacktester:
    """
    Quantitative Pairs Trading Backtest Simulator.
    Simulates dollar-neutral Long/Short positions with dynamic beta hedging,
    slippage, and execution costs.
    """
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 5.0, # 5 bps = 0.05%
        risk_free_rate: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.cost_rate = transaction_cost_bps / 10_000.0
        self.rf = risk_free_rate

    def simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        ticker_a: str,
        ticker_b: str,
        beta: float,
        strategy_name: str = "Strategy",
        split_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Run vectorized simulation for a given signal series (-1, 0, +1).
        Computes overall metrics plus In-Sample vs Out-of-Sample breakdowns.
        """
        sim_df = pd.DataFrame(index=df.index)
        close_a = df[f"close_{ticker_a}"]
        close_b = df[f"close_{ticker_b}"]
        ret_a = df[f"ret_{ticker_a}"]
        ret_b = df[f"ret_{ticker_b}"]
        
        aligned_signals = signals.reindex(df.index).fillna(0)
        if not aligned_signals.isin([-1, 0, 1]).all():
            raise ValueError("signals must contain only -1, 0, or 1.")
        sim_df["signal"] = aligned_signals.astype(int)
        sim_df["position"] = sim_df["signal"].shift(1).fillna(0).astype(int)
        
        # Dollar-neutral pairs spread return
        spread_daily_ret = (ret_a - beta * (close_b.shift(1) / close_a.shift(1).replace(0, np.nan)) * ret_b).fillna(0.0)
        strategy_gross_ret = sim_df["position"] * spread_daily_ret
        
        # Transaction costs
        pos_change = sim_df["position"].diff().abs()
        pos_change.iloc[0] = abs(sim_df["position"].iloc[0])
        costs = pos_change * self.cost_rate * 2.0
        
        sim_df["net_ret"] = strategy_gross_ret - costs
        sim_df["equity"] = self.initial_capital * (1.0 + sim_df["net_ret"]).cumprod()
        
        # Benchmark
        bnh_daily_ret = 0.5 * ret_a + 0.5 * ret_b
        sim_df["bnh_equity"] = self.initial_capital * (1.0 + bnh_daily_ret).cumprod()
        
        # Trade log parsing
        trades = self._extract_trades(sim_df, df, ticker_a, ticker_b)
        
        # Overall metrics
        metrics = self._calculate_metrics(
            sim_df["net_ret"],
            sim_df["equity"],
            trades,
            strategy_name,
            starting_equity=self.initial_capital,
        )
        
        # In-sample vs Out-of-sample breakdown
        in_sample_metrics = None
        out_of_sample_metrics = None
        if split_date is not None and split_date in sim_df.index:
            # In-sample
            is_df = sim_df.loc[:split_date]
            is_trades = [t for t in trades if pd.to_datetime(t["entry_date"]) <= split_date]
            in_sample_metrics = self._calculate_metrics(
                is_df["net_ret"],
                is_df["equity"],
                is_trades,
                f"{strategy_name} (In-Sample)",
                starting_equity=self.initial_capital,
            )
            
            # Out-of-sample
            oos_df = sim_df.loc[split_date:]
            if len(oos_df) > 5:
                oos_base_equity = oos_df["equity"].iloc[0]
                oos_norm_equity = self.initial_capital * (oos_df["equity"] / (oos_base_equity or 1.0))
                oos_trades = [t for t in trades if pd.to_datetime(t["entry_date"]) > split_date]
                out_of_sample_metrics = self._calculate_metrics(
                    oos_df["net_ret"].iloc[1:],
                    oos_norm_equity,
                    oos_trades,
                    f"{strategy_name} (Out-of-Sample)",
                    starting_equity=self.initial_capital,
                )
        
        return {
            "strategy_name": strategy_name,
            "metrics": metrics,
            "in_sample_metrics": in_sample_metrics,
            "out_of_sample_metrics": out_of_sample_metrics,
            "equity_curve": sim_df[["equity", "bnh_equity", "position"]],
            "trades": trades,
            "split_date": split_date,
        }

    def _extract_trades(
        self,
        sim_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        ticker_a: str,
        ticker_b: str
    ) -> List[Dict[str, Any]]:
        """Extract individual completed trade records."""
        trades = []
        in_trade = False
        entry_date = None
        entry_pos = 0
        entry_equity = 0.0
        
        dates = sim_df.index
        positions = sim_df["position"].values
        equities = sim_df["equity"].values
        
        for i in range(len(dates)):
            curr_pos = positions[i]
            
            if not in_trade and curr_pos != 0:
                in_trade = True
                entry_date = dates[i]
                entry_pos = curr_pos
                entry_equity = equities[i-1] if i > 0 else self.initial_capital
            elif in_trade and (curr_pos == 0 or curr_pos != entry_pos):
                exit_date = dates[i]
                exit_equity = equities[i]
                pnl = exit_equity - entry_equity
                pnl_pct = (pnl / entry_equity) * 100.0 if entry_equity else 0.0
                
                trade_dur = (exit_date - entry_date).days if hasattr(exit_date - entry_date, "days") else 1
                
                trades.append({
                    "entry_date": str(entry_date)[:10],
                    "exit_date": str(exit_date)[:10],
                    "direction": "LONG Spread" if entry_pos > 0 else "SHORT Spread",
                    "duration_days": trade_dur,
                    "pnl": round(float(pnl), 2),
                    "return_pct": round(float(pnl_pct), 2),
                    "result": "WIN" if pnl > 0 else "LOSS",
                })
                
                if curr_pos != 0:
                    entry_date = dates[i]
                    entry_pos = curr_pos
                    entry_equity = equities[i]
                else:
                    in_trade = False
                    
        return trades

    def _calculate_metrics(
        self,
        net_returns: pd.Series,
        equity: pd.Series,
        trades: List[Dict[str, Any]],
        name: str,
        starting_equity: float,
    ) -> Dict[str, Any]:
        """Compute institutional performance and risk metrics."""
        n_days = len(net_returns)
        if n_days < 2:
            return {"strategy_name": name, "total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
            
        compounded_return = float(np.prod(1.0 + net_returns.to_numpy(dtype=float)))
        total_ret = (compounded_return - 1.0) * 100.0
        cagr = (
            (compounded_return ** (252.0 / n_days) - 1.0) * 100.0
            if compounded_return > 0
            else -100.0
        )
        
        daily_vol = net_returns.std()
        ann_vol = daily_vol * np.sqrt(252) * 100.0
        
        excess_ret = net_returns.mean() - (self.rf / 252.0)
        sharpe = (excess_ret / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0
        
        downside_returns = net_returns[net_returns < 0]
        downside_std = downside_returns.std()
        sortino = (excess_ret / downside_std) * np.sqrt(252) if (downside_std and downside_std > 0) else sharpe
        
        equity_values = np.concatenate(
            ([starting_equity], equity.to_numpy(dtype=float))
        )
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max
        max_dd = float(np.min(drawdowns)) * 100.0
        
        total_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = (len(wins) / total_trades) * 100.0 if total_trades else 0.0
        
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0
        
        return {
            "strategy_name": name,
            "total_return_pct": round(float(total_ret), 2),
            "cagr_pct": round(float(cagr), 2),
            "annualized_vol_pct": round(float(ann_vol), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "total_trades": total_trades,
            "win_rate_pct": round(float(win_rate), 1),
            "profit_factor": profit_factor,
        }

def run_comparative_backtest(
    feat_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    dl_signals: Optional[np.ndarray] = None,
    split_date: Optional[pd.Timestamp] = None,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_loss_z: float = 3.5,
    parity_threshold: float = 0.06,
    initial_capital: float = 100_000.0
) -> Dict[str, Any]:
    """
    Run 3-way comparative simulation with In-Sample and Out-of-Sample metrics.
    """
    backtester = PairsBacktester(initial_capital=initial_capital)
    z = feat_df["z_score"]
    
    # 1. Statistical Z-score signals
    pos_stat = 0
    signals_stat = []
    for val in z:
        if pos_stat == 0:
            if val <= -entry_z:
                pos_stat = 1
            elif val >= entry_z:
                pos_stat = -1
        elif pos_stat == 1:
            if val >= exit_z or val <= -stop_loss_z:
                pos_stat = 0
        elif pos_stat == -1:
            if val <= -exit_z or val >= stop_loss_z:
                pos_stat = 0
        signals_stat.append(pos_stat)
    signals_stat_series = pd.Series(signals_stat, index=feat_df.index)
    res_stat = backtester.simulate(feat_df, signals_stat_series, ticker_a, ticker_b, beta, "Statistical Z-Score", split_date=split_date)
    
    # 2. Fundamentally-Filtered Signals
    parity_score = feat_df.get("scaled_parity", feat_df.get("feat_parity_score", pd.Series(0.0, index=feat_df.index)))
    signals_filtered = []
    pos_filt = 0
    for idx, val in enumerate(z):
        p_score = abs(float(parity_score.iloc[idx]))
        if pos_filt == 0:
            if p_score <= parity_threshold:
                if val <= -entry_z:
                    pos_filt = 1
                elif val >= entry_z:
                    pos_filt = -1
        elif pos_filt == 1:
            if val >= exit_z or val <= -stop_loss_z:
                pos_filt = 0
        elif pos_filt == -1:
            if val <= -exit_z or val >= stop_loss_z:
                pos_filt = 0
        signals_filtered.append(pos_filt)
    signals_filt_series = pd.Series(signals_filtered, index=feat_df.index)
    res_filt = backtester.simulate(feat_df, signals_filt_series, ticker_a, ticker_b, beta, "Fundamentally-Filtered", split_date=split_date)
    
    # 3. Deep Learning Model Signals
    if dl_signals is not None and len(dl_signals) == len(feat_df):
        dl_series = pd.Series(dl_signals, index=feat_df.index)
        res_dl = backtester.simulate(feat_df, dl_series, ticker_a, ticker_b, beta, "Deep Learning (DeepGapNet)", split_date=split_date)
    else:
        res_dl = None
        
    return {
        "statistical": res_stat,
        "filtered": res_filt,
        "deep_learning": res_dl,
        "split_date": split_date,
    }
