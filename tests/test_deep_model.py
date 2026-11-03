"""Unit tests for the Deep Learning model (DeepGapNet)."""

import numpy as np
import pandas as pd

from src.ml.deep_model import DeepGapNet, prepare_deep_dataset, train_deep_gap_model

def test_deep_gap_net_train_and_predict():
    np.random.seed(42)
    N = 120
    X = np.random.randn(N, 6)
    # Non-linear synthetic target with noise
    y = 0.5 * X[:, 0] - 0.3 * (X[:, 1] ** 2) + 0.2 * X[:, 2] + np.random.randn(N) * 0.1
    
    model = DeepGapNet(
        hidden_dims=(32, 16),
        learning_rate=0.01,
        epochs=15,
        batch_size=16,
        dropout_rate=0.1,
    )
    
    train_res = model.fit(X[:90], y[:90], X[90:], y[90:])
    assert model.is_trained
    assert len(train_res["train_losses"]) == 15
    assert train_res["final_train_loss"] < train_res["train_losses"][0]
    
    preds = model.predict(X[90:])
    assert len(preds) == 30
    
    signals, confidence = model.generate_signals(X[90:], threshold=0.3)
    assert len(signals) == 30
    assert set(signals).issubset({-1, 0, 1})
    assert np.all(confidence >= 0.0) and np.all(confidence <= 100.0)

def test_permutation_feature_importance():
    np.random.seed(42)
    N = 100
    X = np.random.randn(N, 4)
    # Only feature 0 truly determines y
    y = 2.0 * X[:, 0] + np.random.randn(N) * 0.05
    
    model = DeepGapNet(hidden_dims=(16,), epochs=20, learning_rate=0.02)
    feat_names = ["dominant_feat", "noise1", "noise2", "noise3"]
    model.fit(X, y, X_val=X, y_val=y, feature_names=feat_names)
    
    importances = model.compute_feature_importance(X, y)
    assert len(importances) == 4
    top_feature = list(importances.keys())[0]
    assert top_feature == "dominant_feat"


def test_batch_normalization_backprop_matches_finite_differences():
    model = DeepGapNet(
        hidden_dims=(4,),
        dropout_rate=0.0,
        input_noise_std=0.0,
        weight_decay=0.0,
        seed=7,
    )
    X = np.array(
        [[-1.0, 0.5], [0.2, 1.4], [1.1, -0.7], [2.0, 0.3]],
        dtype=float,
    )
    y = np.array([0.3, -0.2, 0.8, 0.1]).reshape(-1, 1)
    model._init_weights(X.shape[1])

    predictions, activations, caches = model._forward(X, training=True)
    _, grad_output = model._huber_loss_and_grad(predictions, y)
    grad_weights, _, grad_gamma, _ = model._backward(
        grad_output, activations, caches
    )

    epsilon = 1e-5

    def loss() -> float:
        prediction, _, _ = model._forward(X, training=True)
        return model._huber_loss_and_grad(prediction, y)[0]

    original_weight = model.weights[0][0, 0]
    model.weights[0][0, 0] = original_weight + epsilon
    loss_plus = loss()
    model.weights[0][0, 0] = original_weight - epsilon
    loss_minus = loss()
    model.weights[0][0, 0] = original_weight
    numeric_weight_grad = (loss_plus - loss_minus) / (2 * epsilon)

    original_gamma = model.bn_gamma[0][0, 0]
    model.bn_gamma[0][0, 0] = original_gamma + epsilon
    loss_plus = loss()
    model.bn_gamma[0][0, 0] = original_gamma - epsilon
    loss_minus = loss()
    model.bn_gamma[0][0, 0] = original_gamma
    numeric_gamma_grad = (loss_plus - loss_minus) / (2 * epsilon)

    assert np.isclose(grad_weights[0][0, 0], numeric_weight_grad, atol=1e-5)
    assert np.isclose(grad_gamma[0][0, 0], numeric_gamma_grad, atol=1e-5)


def test_deep_dataset_does_not_backfill_features_or_targets_or_current_fundamentals():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    feat_df = pd.DataFrame(index=dates)
    feat_df["spread"] = np.linspace(-2.0, 2.0, len(dates))
    feat_df["z_score"] = feat_df["spread"]
    feat_df["rolling_std"] = 1.0
    feat_df["spread_velocity_5d"] = feat_df["spread"].diff(5).fillna(0.0)
    feat_df["spread_velocity_10d"] = feat_df["spread"].diff(10).fillna(0.0)
    feat_df["volatility_ratio"] = 1.0
    feat_df["norm_A"] = 100.0 + feat_df["spread"]
    feat_df["norm_B"] = 100.0 - feat_df["spread"]

    full_features, feature_names, target = prepare_deep_dataset(
        feat_df, "A", "B", {"parity_score": 0.01}, 0.1, pred_horizon=10
    )
    prefix_features, _, _ = prepare_deep_dataset(
        feat_df.iloc[:70], "A", "B", {"parity_score": 0.99}, -0.8, pred_horizon=10
    )

    pd.testing.assert_frame_equal(full_features.iloc[:70], prefix_features)
    assert target.iloc[-10:].isna().all()
    assert not any("fund" in name or "sentiment" in name for name in feature_names)


def test_train_deep_gap_model_pipeline():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    feat_df = pd.DataFrame(index=dates)
    feat_df["spread"] = np.random.randn(100).cumsum()
    feat_df["z_score"] = (feat_df["spread"] - feat_df["spread"].mean()) / (feat_df["spread"].std() or 1.0)
    feat_df["rolling_std"] = feat_df["spread"].rolling(10, min_periods=2).std().fillna(1.0)
    feat_df["spread_velocity_5d"] = feat_df["spread"].diff(5).fillna(0.0)
    feat_df["spread_velocity_10d"] = feat_df["spread"].diff(10).fillna(0.0)
    feat_df["volatility_ratio"] = 1.0
    feat_df["norm_A"] = 100.0 + feat_df["spread"]
    feat_df["norm_B"] = 100.0 - feat_df["spread"]
    
    fund_diff = {"pe_diff": 2.5, "parity_score": 0.03}
    
    model, metrics, signals, importances, split_date = train_deep_gap_model(
        feat_df=feat_df,
        ticker_a="A",
        ticker_b="B",
        fundamental_diff=fund_diff,
        sentiment_gap=0.1,
        epochs=10,
    )
    assert model.is_trained
    assert len(signals) == 100
    assert split_date is not None
    assert "fund_parity_interaction" not in importances
    assert len(importances) > 0
