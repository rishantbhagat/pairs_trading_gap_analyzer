"""Global configuration, default parameters, and presets for the Gap Analyzer."""

import os
from typing import Dict, List, Tuple

# Pre-configured peer stock pairs
PEER_PRESETS: Dict[str, Tuple[str, str]] = {
    "Coca-Cola (KO) vs PepsiCo (PEP) [Beverages]": ("KO", "PEP"),
    "Microsoft (MSFT) vs Alphabet (GOOGL) [Big Tech]": ("MSFT", "GOOGL"),
    "Visa (V) vs Mastercard (MA) [Payments]": ("V", "MA"),
    "ExxonMobil (XOM) vs Chevron (CVX) [Energy]": ("XOM", "CVX"),
    "Home Depot (HD) vs Lowe's (LOW) [Retail / Home]": ("HD", "LOW"),
    "Walmart (WMT) vs Target (TGT) [Retail Discounter]": ("WMT", "TGT"),
}

# OpenRouter Free Models
OPENROUTER_FREE_MODELS: List[str] = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-r1:free",
]

DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"

# Default Quantitative Hyperparameters
DEFAULT_LOOKBACK_PERIOD = "2y"   # Historical window for price & ratios
DEFAULT_ROLLING_WINDOW = 30      # Days for rolling mean & std dev
DEFAULT_ENTRY_Z = 2.0            # Z-score entry threshold
DEFAULT_EXIT_Z = 0.0             # Mean reversion target
DEFAULT_STOP_LOSS_Z = 3.5        # Stop loss threshold
DEFAULT_TRANSACTION_COST_BPS = 5.0 # Slippage + brokerage per trade leg

# Deep Learning Default Hyperparameters
DEFAULT_DL_EPOCHS = 40
DEFAULT_DL_BATCH_SIZE = 16
DEFAULT_DL_LR = 0.005
DEFAULT_DL_PRED_HORIZON = 10     # Days ahead to predict convergence
