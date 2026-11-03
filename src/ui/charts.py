"""Plotly interactive charts for price comparisons, spread gaps, deep learning, and backtests."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_normalized_prices(df: pd.DataFrame, ticker_a: str, ticker_b: str) -> go.Figure:
    """
    Plot normalized price series (Base 100) to visualize relative performance.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[f"norm_{ticker_a}"],
        name=f"{ticker_a} (Base 100)",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>" + f"{ticker_a}: %{{y:.2f}}<extra></extra>",
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[f"norm_{ticker_b}"],
        name=f"{ticker_b} (Base 100)",
        line=dict(color="#ff7f0e", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>" + f"{ticker_b}: %{{y:.2f}}<extra></extra>",
    ))
    
    fig.update_layout(
        title=f"<b>Normalized Performance Comparison (Base 100)</b>: {ticker_a} vs {ticker_b}",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_dark",
    )
    return fig

def plot_spread_and_bands(
    feat_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str
) -> go.Figure:
    """
    Plot OLS Spread with rolling mean, Bollinger ±1σ, ±2σ bands, and shaded 'Potential Gap' zones.
    """
    fig = go.Figure()
    
    dates = feat_df.index
    spread = feat_df["spread"]
    mean = feat_df["rolling_mean"]
    u2 = feat_df["upper_band_2"]
    l2 = feat_df["lower_band_2"]
    z = feat_df["z_score"]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=u2,
        name="+2.0σ Upper Band",
        line=dict(color="rgba(239, 83, 80, 0.5)", width=1, dash="dash"),
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=l2,
        name="-2.0σ Lower Band",
        line=dict(color="rgba(76, 175, 80, 0.5)", width=1, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(120, 120, 120, 0.05)",
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=mean,
        name="Fair Value (Rolling Mean)",
        line=dict(color="#9e9e9e", width=1.5, dash="dot"),
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=spread,
        name="Spread (OLS Residual)",
        line=dict(color="#00e5ff", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Spread: %{y:.2f}<extra></extra>",
    ))
    
    gap_long = feat_df[z <= -2.0]
    if not gap_long.empty:
        fig.add_trace(go.Scatter(
            x=gap_long.index,
            y=gap_long["spread"],
            mode="markers",
            name="Potential Gap: Oversold (Long Signal)",
            marker=dict(color="#00e676", size=8, symbol="triangle-up"),
            hovertemplate="Long Signal: Z = %{text:.2f}<extra></extra>",
            text=gap_long["z_score"],
        ))
        
    gap_short = feat_df[z >= 2.0]
    if not gap_short.empty:
        fig.add_trace(go.Scatter(
            x=gap_short.index,
            y=gap_short["spread"],
            mode="markers",
            name="Potential Gap: Overbought (Short Signal)",
            marker=dict(color="#ff1744", size=8, symbol="triangle-down"),
            hovertemplate="Short Signal: Z = %{text:.2f}<extra></extra>",
            text=gap_short["z_score"],
        ))
        
    fig.update_layout(
        title=f"<b>OLS Price Spread & Statistical Gap Zones</b> ({ticker_a} - β·{ticker_b})",
        xaxis_title="Date",
        yaxis_title="Spread Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_dark",
    )
    return fig

def plot_zscore_timeline(feat_df: pd.DataFrame) -> go.Figure:
    """
    Plot rolling Z-score with overbought/oversold threshold bounds.
    """
    fig = go.Figure()
    dates = feat_df.index
    z = feat_df["z_score"]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=z,
        name="Z-Score",
        line=dict(color="#ffd600", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Z-Score: %{y:.2f}σ<extra></extra>",
    ))
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ff5252", annotation_text="+2.0σ Short Entry")
    fig.add_hline(y=0.0, line_dash="dot", line_color="#ffffff", annotation_text="0.0 Mean")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="#69f0ae", annotation_text="-2.0σ Long Entry")
    
    fig.update_layout(
        title="<b>Rolling Z-Score Time Series</b>",
        xaxis_title="Date",
        yaxis_title="Standard Deviations (σ)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_dark",
    )
    return fig

def plot_support_resistance_channels(
    price_df: pd.DataFrame,
    ticker: str,
    channels: List[Dict[str, Any]],
    pivots: List[Dict[str, Any]],
    show_pivots: bool = False,
    show_breakouts: bool = False,
    moving_averages: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    """Plot OHLC candles with the latest pivot-based support/resistance zones."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df[f"open_{ticker}"],
            high=price_df[f"high_{ticker}"],
            low=price_df[f"low_{ticker}"],
            close=price_df[f"close_{ticker}"],
            name=ticker,
        )
    )

    latest_close = float(price_df[f"close_{ticker}"].iloc[-1])
    for channel_idx, channel in enumerate(channels, start=1):
        lower = float(channel["lower"])
        upper = float(channel["upper"])
        color = (
            "rgba(244, 67, 54, 0.20)"
            if lower > latest_close
            else "rgba(76, 175, 80, 0.20)"
            if upper < latest_close
            else "rgba(158, 158, 158, 0.28)"
        )
        fig.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor=color,
            line=dict(color=color.replace("0.20", "0.85").replace("0.28", "0.85")),
            layer="below",
            annotation_text=f"S/R {channel_idx} · strength {channel['strength']}",
            annotation_position="top left",
        )

    if show_pivots and pivots:
        for kind, color, symbol in (
            ("high", "#ff5252", "triangle-down"),
            ("low", "#69f0ae", "triangle-up"),
        ):
            selected = [pivot for pivot in pivots if pivot["kind"] == kind]
            if selected:
                fig.add_trace(
                    go.Scatter(
                        x=[pivot["index"] for pivot in selected],
                        y=[pivot["price"] for pivot in selected],
                        mode="markers",
                        name="Pivot High" if kind == "high" else "Pivot Low",
                        marker=dict(color=color, size=8, symbol=symbol),
                        customdata=[
                            price_df.index[pivot["confirmation_index"]]
                            for pivot in selected
                        ],
                        hovertemplate=(
                            "%{x|%Y-%m-%d}<br>Pivot: %{y:.2f}"
                            "<br>Confirmed: %{customdata|%Y-%m-%d}<extra></extra>"
                        ),
                    )
                )

    if moving_averages:
        close = price_df[f"close_{ticker}"]
        for average in moving_averages:
            length = int(average["length"])
            if average["type"] == "EMA":
                values = close.ewm(span=length, adjust=False, min_periods=length).mean()
            else:
                values = close.rolling(length, min_periods=length).mean()
            fig.add_trace(
                go.Scatter(
                    x=price_df.index,
                    y=values,
                    name=f"{average['type']} {length}",
                    line=dict(color=average["color"], width=1.5),
                )
            )

    if show_breakouts and channels and len(price_df) > 1:
        previous_close = float(price_df[f"close_{ticker}"].iloc[-2])
        broken_resistance = any(
            previous_close <= channel["upper"] < latest_close for channel in channels
        )
        broken_support = any(
            previous_close >= channel["lower"] > latest_close for channel in channels
        )
        if broken_resistance:
            fig.add_trace(
                go.Scatter(
                    x=[price_df.index[-1]],
                    y=[price_df[f"low_{ticker}"].iloc[-1]],
                    mode="markers",
                    name="Resistance Broken",
                    marker=dict(color="#69f0ae", size=11, symbol="triangle-up"),
                )
            )
        if broken_support:
            fig.add_trace(
                go.Scatter(
                    x=[price_df.index[-1]],
                    y=[price_df[f"high_{ticker}"].iloc[-1]],
                    mode="markers",
                    name="Support Broken",
                    marker=dict(color="#ff5252", size=11, symbol="triangle-down"),
                )
            )

    fig.update_layout(
        title=f"<b>{ticker} Support / Resistance Channels</b>",
        xaxis_title="Date",
        yaxis_title="Adjusted Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )
    return fig

def plot_backtest_equity_curves(res_dict: Dict[str, Any], split_date: Optional[pd.Timestamp] = None) -> go.Figure:
    """
    Plot cumulative equity curves with explicit In-Sample vs Out-of-Sample demarcation.
    """
    fig = go.Figure()
    
    stat_res = res_dict.get("statistical")
    filt_res = res_dict.get("filtered")
    dl_res = res_dict.get("deep_learning")
    split = split_date or res_dict.get("split_date")
    
    if stat_res:
        eq_df = stat_res["equity_curve"]
        # Benchmark
        fig.add_trace(go.Scatter(
            x=eq_df.index,
            y=eq_df["bnh_equity"],
            name="Buy & Hold (50/50 Benchmark)",
            line=dict(color="#757575", width=1.8, dash="dot"),
        ))
        # Statistical
        fig.add_trace(go.Scatter(
            x=eq_df.index,
            y=eq_df["equity"],
            name=f"Statistical Z-Score (Ret: {stat_res['metrics']['total_return_pct']:+.1f}%)",
            line=dict(color="#29b6f6", width=2.5),
        ))
        
    if filt_res:
        eq_df = filt_res["equity_curve"]
        fig.add_trace(go.Scatter(
            x=eq_df.index,
            y=eq_df["equity"],
            name=f"Fundamentally-Filtered (Ret: {filt_res['metrics']['total_return_pct']:+.1f}%)",
            line=dict(color="#ab47bc", width=2.5),
        ))
        
    if dl_res:
        eq_df = dl_res["equity_curve"]
        # Add labels showing Out-of-sample return if available
        oos_m = dl_res.get("out_of_sample_metrics")
        oos_ret_str = f" | OOS: {oos_m['total_return_pct']:+.1f}%" if oos_m else ""
        fig.add_trace(go.Scatter(
            x=eq_df.index,
            y=eq_df["equity"],
            name=f"Deep Learning Model (Total: {dl_res['metrics']['total_return_pct']:+.1f}%{oos_ret_str})",
            line=dict(color="#00e676", width=3.0),
        ))
        
    # Vertical line & shading for Out-of-Sample evaluation window
    if split is not None and stat_res:
        dates = stat_res["equity_curve"].index
        # Convert to ISO string to avoid Plotly/Pandas Timestamp arithmetic bug
        split_str = str(split)[:10]
        end_str = str(dates[-1])[:10]
        fig.add_shape(
            type="line",
            x0=split_str, x1=split_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="#ffab00", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=split_str, y=1.0, yref="paper",
            text="Train / Test Split (Out-of-Sample -->)",
            showarrow=False,
            font=dict(color="#ffab00", size=11),
            xanchor="right",
            yanchor="bottom",
        )
        fig.add_shape(
            type="rect",
            x0=split_str, x1=end_str,
            y0=0, y1=1,
            yref="paper",
            fillcolor="rgba(255, 171, 0, 0.08)",
            layer="below",
            line_width=0,
        )
        
    fig.update_layout(
        title="<b>Cumulative Equity Growth Comparison ($100,000 Initial Capital)</b>",
        xaxis_title="Date",
        yaxis_title="Portfolio Equity ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_dark",
    )
    return fig

def plot_dl_loss_curves(train_losses: List[float], val_losses: List[float]) -> go.Figure:
    """
    Plot Deep Neural Network training and validation Huber loss per epoch.
    """
    fig = go.Figure()
    epochs = list(range(1, len(train_losses) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        name="Training Loss (Huber)",
        line=dict(color="#29b6f6", width=2.2),
    ))
    
    if val_losses:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            name="Validation Loss (Huber)",
            line=dict(color="#ff9100", width=2.2, dash="dash"),
        ))
        
        # Best validation checkpoint marker
        best_epoch = int(np.argmin(val_losses)) + 1
        best_loss = min(val_losses)
        fig.add_trace(go.Scatter(
            x=[best_epoch],
            y=[best_loss],
            mode="markers",
            name=f"Best Model Checkpoint (Epoch {best_epoch})",
            marker=dict(color="#00e676", size=11, symbol="star"),
        ))
        
    fig.update_layout(
        title="<b>DeepGapNet Learning Curves (Huber Loss vs Epoch)</b>",
        xaxis_title="Epoch",
        yaxis_title="Huber Loss (Smooth L1)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_dark",
    )
    return fig

def plot_feature_importance(importances: Dict[str, float]) -> go.Figure:
    """
    Plot horizontal bar chart of feature importance rankings.
    """
    features = list(reversed(list(importances.keys())))
    scores = list(reversed(list(importances.values())))
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale="Viridis",
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title="<b>Deep Learning Permutation Feature Importance (%)</b>",
        xaxis_title="Relative Importance Weight (%)",
        yaxis_title="Engineered Feature",
        margin=dict(l=180, r=40, t=60, b=40),
        template="plotly_dark",
    )
    return fig
