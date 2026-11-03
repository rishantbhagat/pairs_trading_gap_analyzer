"""Baseline comparative machine learning model (Random Forest / XGBoost)."""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

def train_baseline_comparator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Train a baseline Random Forest / Gradient Boosted model to benchmark against DeepGapNet.
    """
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    except Exception:
        model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    
    importances = model.feature_importances_
    total = sum(importances) or 1.0
    feat_imp = {
        name: round(float(imp / total) * 100.0, 2)
        for name, imp in zip(feature_names, importances)
    }
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
    
    metrics = {
        "val_mse": round(val_mse, 5),
        "model_type": type(model).__name__,
    }
    return model, metrics, feat_imp
