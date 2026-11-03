"""Pivot-based support and resistance channel detection."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_support_resistance_channels(
    price_df: pd.DataFrame,
    ticker: str,
    pivot_period: int = 10,
    source: str = "High/Low",
    channel_width_pct: float = 5.0,
    min_strength: int = 1,
    max_channels: int = 6,
    loopback: int = 290,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return strongest current pivot channels and confirmed pivot markers.

    Pivots are dated at their actual bar but are only included once the
    ``pivot_period`` bars to their right have closed. The returned channels
    describe the latest data window; they are not historical trading signals.
    """
    if pivot_period < 1:
        raise ValueError("pivot_period must be positive.")
    if not 1 <= channel_width_pct <= 100:
        raise ValueError("channel_width_pct must be between 1 and 100.")
    if min_strength < 1 or max_channels < 1 or loopback < 1:
        raise ValueError("min_strength, max_channels, and loopback must be positive.")
    if source not in {"High/Low", "Close/Open"}:
        raise ValueError("source must be 'High/Low' or 'Close/Open'.")

    columns = {
        "open": f"open_{ticker}",
        "high": f"high_{ticker}",
        "low": f"low_{ticker}",
        "close": f"close_{ticker}",
    }
    missing = [column for column in columns.values() if column not in price_df]
    if missing:
        raise ValueError(f"price_df is missing required columns: {', '.join(missing)}.")
    if not price_df.index.is_monotonic_increasing or not price_df.index.is_unique:
        raise ValueError("price_df must have a unique, chronological index.")

    data = price_df[list(columns.values())].astype(float)
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("price_df must contain finite OHLC data.")
    if (data[[columns["high"], columns["low"]]] <= 0).any().any():
        raise ValueError("High and low prices must be positive.")

    if source == "High/Low":
        pivot_high_source = data[columns["high"]].to_numpy()
        pivot_low_source = data[columns["low"]].to_numpy()
    else:
        pivot_high_source = np.maximum(
            data[columns["close"]].to_numpy(), data[columns["open"]].to_numpy()
        )
        pivot_low_source = np.minimum(
            data[columns["close"]].to_numpy(), data[columns["open"]].to_numpy()
        )

    highs = data[columns["high"]].to_numpy()
    lows = data[columns["low"]].to_numpy()
    size = len(data)
    confirmed_after = max(0, size - loopback)
    pivots: List[Dict[str, Any]] = []

    for pivot_idx in range(pivot_period, size - pivot_period):
        confirmation_idx = pivot_idx + pivot_period
        if confirmation_idx < confirmed_after:
            continue

        high_window = pivot_high_source[pivot_idx - pivot_period:pivot_idx + pivot_period + 1]
        low_window = pivot_low_source[pivot_idx - pivot_period:pivot_idx + pivot_period + 1]
        pivot_high = pivot_high_source[pivot_idx]
        pivot_low = pivot_low_source[pivot_idx]

        if pivot_high == np.max(high_window) and np.count_nonzero(high_window == pivot_high) == 1:
            pivots.append(
                {
                    "index": price_df.index[pivot_idx],
                    "kind": "high",
                    "price": float(pivot_high),
                    "confirmation_index": confirmation_idx,
                }
            )
        if pivot_low == np.min(low_window) and np.count_nonzero(low_window == pivot_low) == 1:
            pivots.append(
                {
                    "index": price_df.index[pivot_idx],
                    "kind": "low",
                    "price": float(pivot_low),
                    "confirmation_index": confirmation_idx,
                }
            )

    if not pivots:
        return [], []

    range_start = max(0, size - 300)
    channel_width = (
        float(np.max(highs[range_start:]) - np.min(lows[range_start:]))
        * channel_width_pct
        / 100.0
    )
    if channel_width <= 0:
        return [], pivots

    pivot_prices = np.array([pivot["price"] for pivot in pivots], dtype=float)
    touch_start = max(0, size - loopback)
    touch_highs = highs[touch_start:]
    touch_lows = lows[touch_start:]
    candidates: List[Dict[str, Any]] = []

    for seed in range(len(pivots)):
        in_channel = np.abs(pivot_prices - pivot_prices[seed]) <= channel_width
        member_prices = pivot_prices[in_channel]
        lower = float(np.min(member_prices))
        upper = float(np.max(member_prices))
        pivot_count = int(np.count_nonzero(in_channel))
        touches = int(
            np.count_nonzero(
                ((touch_highs >= lower) & (touch_highs <= upper))
                | ((touch_lows >= lower) & (touch_lows <= upper))
            )
        )
        strength = pivot_count * 20 + touches
        if pivot_count >= min_strength:
            candidate = {
                "lower": lower,
                "upper": upper,
                "strength": strength,
                "pivot_count": pivot_count,
                "touch_count": touches,
            }
            if candidate not in candidates:
                candidates.append(candidate)

    candidates.sort(key=lambda item: (-item["strength"], item["lower"]))
    channels: List[Dict[str, Any]] = []
    for candidate in candidates:
        overlaps = any(
            candidate["lower"] <= chosen["upper"]
            and candidate["upper"] >= chosen["lower"]
            for chosen in channels
        )
        if not overlaps:
            channels.append(candidate)
            if len(channels) >= max_channels:
                break

    return channels, pivots
