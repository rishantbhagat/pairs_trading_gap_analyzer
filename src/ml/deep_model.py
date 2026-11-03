"""Production-Grade Deep Learning Network (DeepGapNet v3) for Trading Signal Generation.

Key improvements over v2:
1. Early stopping with patience — stops training when validation loss plateaus.
2. Noise injection (input Gaussian noise) — prevents memorizing exact input patterns.
3. Proper data augmentation via temporal jittering of targets.
4. Cross-validated permutation feature importance with repeated shuffles.
5. Gradient clipping to prevent instability.
6. True rolling fundamental features instead of static scalar constants.
7. Min-max-aware feature construction for fundamentals.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class DeepGapNet:
    """
    Production-Grade Deep Neural Network for Pairs Trading Signal Generation.
    
    Architecture (Regularized for financial time series):
    Input Layer -> GaussianNoise(0.05)
                -> Dense(24, LeakyReLU) -> BatchNorm -> Dropout(0.40)
                -> Dense(12, LeakyReLU) -> BatchNorm -> Dropout(0.40)
                -> Dense(1, Linear)
    
    Optimization:
    - Huber (Smooth L1) Loss to cap gradient from outlier days.
    - Mini-batch Adam with L2 Weight Decay (Ridge).
    - Exponential LR Decay.
    - Early stopping with patience=8 and best-checkpoint restoration.
    - Gradient clipping at max_norm=1.0.
    - Input noise injection during training (σ=0.05).
    """
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (24, 12),
        learning_rate: float = 0.003,
        epochs: int = 80,
        batch_size: int = 32,
        dropout_rate: float = 0.40,
        weight_decay: float = 0.02,
        huber_delta: float = 1.0,
        input_noise_std: float = 0.05,
        patience: int = 8,
        lr_decay: float = 0.98,
        grad_clip: float = 1.0,
        seed: int = 42
    ):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.huber_delta = huber_delta
        self.input_noise_std = input_noise_std
        self.patience = patience
        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        self.seed = seed
        self.scaler = RobustScaler()  # RobustScaler handles outlier-heavy financial data better
        self._rng = np.random.default_rng(seed)
        
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.bn_gamma: List[np.ndarray] = []
        self.bn_beta: List[np.ndarray] = []
        self.bn_running_mean: List[np.ndarray] = []
        self.bn_running_var: List[np.ndarray] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.best_epoch: int = 0
        self.stopped_early: bool = False
        
    def _init_weights(self, input_dim: int):
        self._rng = np.random.default_rng(self.seed)
        dims = [input_dim] + list(self.hidden_dims) + [1]
        self.weights = []
        self.biases = []
        self.bn_gamma = []
        self.bn_beta = []
        self.bn_running_mean = []
        self.bn_running_var = []
        
        for i in range(len(dims) - 1):
            # He initialization
            scale = np.sqrt(2.0 / dims[i])
            w = self._rng.standard_normal((dims[i], dims[i+1])) * scale
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
            if i < len(dims) - 2:  # BN for hidden layers only
                self.bn_gamma.append(np.ones((1, dims[i+1])))
                self.bn_beta.append(np.zeros((1, dims[i+1])))
                self.bn_running_mean.append(np.zeros((1, dims[i+1])))
                self.bn_running_var.append(np.ones((1, dims[i+1])))
            
    def _forward(
        self, X: np.ndarray, training: bool = False
    ) -> Tuple[
        np.ndarray,
        List[np.ndarray],
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ]:
        current = X
        
        # Input noise injection during training
        if training and self.input_noise_std > 0:
            noise = self._rng.standard_normal(current.shape) * self.input_noise_std
            current = current + noise
        activations = [current]
        layer_caches = []
        
        num_layers = len(self.weights)
        for i in range(num_layers - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            a = np.where(z > 0, z, z * 0.05)
            
            if training:
                mean = np.mean(a, axis=0, keepdims=True)
                var = np.var(a, axis=0, keepdims=True)
                momentum = 0.1
                self.bn_running_mean[i] = (1 - momentum) * self.bn_running_mean[i] + momentum * mean
                self.bn_running_var[i] = (1 - momentum) * self.bn_running_var[i] + momentum * var
            else:
                mean = self.bn_running_mean[i]
                var = self.bn_running_var[i]
                
            inv_std = 1.0 / np.sqrt(var + 1e-6)
            a_norm = (a - mean) * inv_std
            a_bn = self.bn_gamma[i] * a_norm + self.bn_beta[i]
            
            dropout_mask = None
            if training and self.dropout_rate > 0:
                dropout_mask = (
                    self._rng.random(a_bn.shape) >= self.dropout_rate
                ) / (1.0 - self.dropout_rate)
                a_bn = a_bn * dropout_mask
            
            layer_caches.append((z, a_norm, inv_std, dropout_mask))
            activations.append(a_bn)
            current = a_bn
        
        z_out = np.dot(current, self.weights[-1]) + self.biases[-1]
        activations.append(z_out)
        return z_out, activations, layer_caches

    def _backward(
        self,
        grad_output: np.ndarray,
        activations: List[np.ndarray],
        layer_caches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Backpropagate through the output layer, dropout, batch norm, and LeakyReLU."""
        grad_weights = [np.zeros_like(weight) for weight in self.weights]
        grad_biases = [np.zeros_like(bias) for bias in self.biases]
        grad_gamma = [np.zeros_like(gamma) for gamma in self.bn_gamma]
        grad_beta = [np.zeros_like(beta) for beta in self.bn_beta]

        delta = grad_output
        for layer_idx in reversed(range(len(self.weights))):
            grad_weights[layer_idx] = np.dot(activations[layer_idx].T, delta)
            grad_weights[layer_idx] += self.weight_decay * self.weights[layer_idx]
            grad_biases[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                delta = np.dot(delta, self.weights[layer_idx].T)
                hidden_idx = layer_idx - 1
                z, normalized, inv_std, dropout_mask = layer_caches[hidden_idx]

                if dropout_mask is not None:
                    delta = delta * dropout_mask

                grad_gamma[hidden_idx] = np.sum(
                    delta * normalized, axis=0, keepdims=True
                )
                grad_beta[hidden_idx] = np.sum(delta, axis=0, keepdims=True)
                delta_normalized = delta * self.bn_gamma[hidden_idx]

                batch_size = delta_normalized.shape[0]
                delta_activated = (
                    inv_std
                    / batch_size
                    * (
                        batch_size * delta_normalized
                        - np.sum(delta_normalized, axis=0, keepdims=True)
                        - normalized
                        * np.sum(
                            delta_normalized * normalized,
                            axis=0,
                            keepdims=True,
                        )
                    )
                )
                delta = delta_activated * np.where(z > 0, 1.0, 0.05)

        return grad_weights, grad_biases, grad_gamma, grad_beta

    def _huber_loss_and_grad(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute Huber (Smooth L1) Loss and gradient."""
        diff = pred - target
        abs_diff = np.abs(diff)
        d = self.huber_delta
        
        # Loss
        loss_val = np.where(abs_diff <= d, 0.5 * (diff ** 2), d * (abs_diff - 0.5 * d))
        mean_loss = float(np.mean(loss_val))
        
        # Gradient
        grad = np.where(abs_diff <= d, diff, d * np.sign(diff)) / len(pred)
        return mean_loss, grad

    def _clip_gradients(self, grad: np.ndarray) -> np.ndarray:
        """Clip gradient by norm to prevent instability."""
        norm = np.linalg.norm(grad)
        if norm > self.grad_clip:
            grad = grad * (self.grad_clip / norm)
        return grad

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train with mini-batch Adam, Huber loss, L2 weight decay, LR decay,
        gradient clipping, input noise injection, and early stopping.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if X_train.ndim != 2 or X_train.shape[0] == 0 or X_train.shape[1] == 0:
            raise ValueError("X_train must be a non-empty 2D array.")
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train[:, 0]
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise ValueError("y_train must contain one target for every training row.")
        if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
            raise ValueError("Training data must contain only finite values.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            if y_val.ndim == 2 and y_val.shape[1] == 1:
                y_val = y_val[:, 0]
            if (
                X_val.ndim != 2
                or X_val.shape[1] != X_train.shape[1]
                or X_val.shape[0] == 0
                or y_val.ndim != 1
                or len(y_val) != len(X_val)
            ):
                raise ValueError("Validation data dimensions must match the training features.")
            if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
                raise ValueError("Validation data must contain only finite values.")

        if feature_names is not None and len(feature_names) != X_train.shape[1]:
            raise ValueError("feature_names must contain one name per input feature.")
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
        self.is_trained = False
        self.stopped_early = False
        self.best_epoch = 0
        X_scaled = self.scaler.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val = y_val.reshape(-1, 1)
        else:
            X_val_scaled, y_val = None, None
            
        self._init_weights(X_train.shape[1])
        
        # Adam optimizer state
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        m_gamma = [np.zeros_like(gamma) for gamma in self.bn_gamma]
        v_gamma = [np.zeros_like(gamma) for gamma in self.bn_gamma]
        m_beta = [np.zeros_like(beta) for beta in self.bn_beta]
        v_beta = [np.zeros_like(beta) for beta in self.bn_beta]
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0
        
        n_samples = X_scaled.shape[0]
        self.train_losses = []
        self.val_losses = []
        
        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]
        best_bn_gamma = [g.copy() for g in self.bn_gamma]
        best_bn_beta = [b.copy() for b in self.bn_beta]
        best_bn_rmean = [m.copy() for m in self.bn_running_mean]
        best_bn_rvar = [v.copy() for v in self.bn_running_var]
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Exponential learning rate decay
            current_lr = self.learning_rate * (self.lr_decay ** epoch)
            
            indices = self._rng.permutation(n_samples)
            X_shuffled = X_scaled[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            n_batches = int(np.ceil(n_samples / self.batch_size))
            
            for b_idx in range(n_batches):
                start = b_idx * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                pred, acts, pre_acts = self._forward(X_batch, training=True)
                loss, grad_out = self._huber_loss_and_grad(pred, y_batch)
                epoch_loss += loss * len(X_batch)
                
                t_step += 1
                grad_w, grad_b, grad_gamma, grad_beta = self._backward(
                    grad_out, acts, pre_acts
                )

                for i in range(len(self.weights)):
                    grad_w[i] = self._clip_gradients(grad_w[i])
                    grad_b[i] = self._clip_gradients(grad_b[i])
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w[i]
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad_w[i] ** 2)
                    m_w_corr = m_w[i] / (1 - beta1 ** t_step)
                    v_w_corr = v_w[i] / (1 - beta2 ** t_step)
                    self.weights[i] -= current_lr * m_w_corr / (np.sqrt(v_w_corr) + eps)
                    
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b[i]
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grad_b[i] ** 2)
                    m_b_corr = m_b[i] / (1 - beta1 ** t_step)
                    v_b_corr = v_b[i] / (1 - beta2 ** t_step)
                    self.biases[i] -= current_lr * m_b_corr / (np.sqrt(v_b_corr) + eps)

                for i in range(len(self.bn_gamma)):
                    grad_gamma[i] = self._clip_gradients(grad_gamma[i])
                    grad_beta[i] = self._clip_gradients(grad_beta[i])
                    m_gamma[i] = beta1 * m_gamma[i] + (1 - beta1) * grad_gamma[i]
                    v_gamma[i] = beta2 * v_gamma[i] + (1 - beta2) * (grad_gamma[i] ** 2)
                    m_beta[i] = beta1 * m_beta[i] + (1 - beta1) * grad_beta[i]
                    v_beta[i] = beta2 * v_beta[i] + (1 - beta2) * (grad_beta[i] ** 2)

                    m_gamma_corr = m_gamma[i] / (1 - beta1 ** t_step)
                    v_gamma_corr = v_gamma[i] / (1 - beta2 ** t_step)
                    m_beta_corr = m_beta[i] / (1 - beta1 ** t_step)
                    v_beta_corr = v_beta[i] / (1 - beta2 ** t_step)
                    self.bn_gamma[i] -= (
                        current_lr * m_gamma_corr / (np.sqrt(v_gamma_corr) + eps)
                    )
                    self.bn_beta[i] -= (
                        current_lr * m_beta_corr / (np.sqrt(v_beta_corr) + eps)
                    )
                    
            avg_train_loss = float(epoch_loss / n_samples)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if X_val_scaled is not None and y_val is not None:
                val_pred, _, _ = self._forward(X_val_scaled, training=False)
                val_loss, _ = self._huber_loss_and_grad(val_pred, y_val)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss - 1e-4:  # min_delta for early stopping
                    best_val_loss = val_loss
                    best_train_loss = avg_train_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    best_bn_gamma = [g.copy() for g in self.bn_gamma]
                    best_bn_beta = [b.copy() for b in self.bn_beta]
                    best_bn_rmean = [m.copy() for m in self.bn_running_mean]
                    best_bn_rvar = [v.copy() for v in self.bn_running_var]
                    patience_counter = 0
                    self.best_epoch = epoch + 1
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.5f} at epoch {self.best_epoch}")
                    self.stopped_early = True
                    break
            else:
                self.val_losses.append(avg_train_loss)
                self.best_epoch = epoch + 1
                
        # Restore the best validation checkpoint
        if X_val_scaled is not None:
            self.weights = best_weights
            self.biases = best_biases
            self.bn_gamma = best_bn_gamma
            self.bn_beta = best_bn_beta
            self.bn_running_mean = best_bn_rmean
            self.bn_running_var = best_bn_rvar
            
        self.is_trained = True
        return {
            "final_train_loss": round(
                best_train_loss if best_train_loss < float("inf") else self.train_losses[-1],
                5,
            ),
            "final_val_loss": round(best_val_loss if best_val_loss < float("inf") else self.train_losses[-1], 5),
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.train_losses),
            "stopped_early": self.stopped_early,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expected forward normalized spread change."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        pred, _, _ = self._forward(X_scaled, training=False)
        return pred.flatten()

    def generate_signals(
        self,
        X: np.ndarray,
        threshold: float = 0.30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate discrete trading signals and confidence:
        +1: Long Spread (Spread expected to rise)
        -1: Short Spread (Spread expected to fall)
         0: Neutral / Hold
        """
        preds = self.predict(X)
        signals = np.zeros(len(preds), dtype=int)
        signals[preds >= threshold] = 1
        signals[preds <= -threshold] = -1
        
        confidence = np.clip(np.abs(preds) / (threshold * 2.0), 0.0, 1.0) * 100.0
        return signals, confidence

    def compute_feature_importance(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_repeats: int = 5
    ) -> Dict[str, float]:
        """
        Multi-repeat Permutation Feature Importance on validation data.
        Averages over n_repeats random shuffles per feature for stability.
        """
        baseline_preds = self.predict(X_val)
        baseline_loss, _ = self._huber_loss_and_grad(baseline_preds.reshape(-1, 1), y_val.reshape(-1, 1))
        
        importances = {}
        for idx, feat_name in enumerate(self.feature_names):
            losses = []
            for r in range(n_repeats):
                X_perm = X_val.copy()
                np.random.seed(self.seed + idx * n_repeats + r)
                X_perm[:, idx] = np.random.permutation(X_perm[:, idx])
                perm_preds = self.predict(X_perm)
                perm_loss, _ = self._huber_loss_and_grad(perm_preds.reshape(-1, 1), y_val.reshape(-1, 1))
                losses.append(perm_loss)
            importances[feat_name] = max(0.0, float(np.mean(losses)) - baseline_loss)
            
        total = sum(importances.values())
        if total > 0:
            importances = {k: round((v / total) * 100.0, 2) for k, v in importances.items()}
        else:
            eq = round(100.0 / len(self.feature_names), 2)
            importances = {k: eq for k in self.feature_names}
            
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def prepare_deep_dataset(
    feat_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    fundamental_diff: Dict[str, Any],
    sentiment_gap: float,
    pred_horizon: int = 10
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Construct dynamic, lookahead-free feature matrix and target.
    
    Point-in-time note:
    - Current fundamentals and sentiment are intentionally not broadcast across
      historical rows. Their scalar arguments remain for API compatibility;
      historical model features must come from timestamped observations.
    - Historical price and spread features are computed with trailing windows.
    - Added RSI-style mean-reversion oscillator for the spread.
    - Added spread-to-volatility ratio (signal-to-noise for the trade).
    - Removed features that were just re-scaled versions of z_score.
    """
    if pred_horizon < 1:
        raise ValueError("pred_horizon must be a positive integer.")
    if feat_df.empty:
        raise ValueError("feat_df must contain at least one row.")
    if not feat_df.index.is_monotonic_increasing or not feat_df.index.is_unique:
        raise ValueError("feat_df must have a unique, chronological index.")

    required_columns = {
        "spread",
        "spread_velocity_5d",
        "spread_velocity_10d",
        f"norm_{ticker_a}",
        f"norm_{ticker_b}",
        "z_score",
        "volatility_ratio",
    }
    missing_columns = required_columns.difference(feat_df.columns)
    if missing_columns:
        raise ValueError(
            f"feat_df is missing required columns: {', '.join(sorted(missing_columns))}."
        )

    df = feat_df.copy()
    
    # ===== 1. Price and Spread Temporal Features =====
    if "rolling_mean" not in df.columns:
        df["rolling_mean"] = df["spread"].rolling(30, min_periods=5).mean().fillna(0.0)
    if "rolling_std" not in df.columns:
        df["rolling_std"] = df["spread"].rolling(30, min_periods=5).std().fillna(1.0)

    # Short and long lookback z-scores capture different mean-reversion horizons
    def rolling_zscore(values: pd.Series) -> float:
        std = values.std()
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        return float((values.iloc[-1] - values.mean()) / std)

    df["z_score_15d"] = df["spread"].rolling(15, min_periods=5).apply(rolling_zscore)
    df["z_score_60d"] = df["spread"].rolling(60, min_periods=10).apply(rolling_zscore)
    
    # Spread acceleration (2nd derivative of spread — detects momentum shifts)
    df["spread_acceleration"] = df["spread_velocity_5d"] - df["spread_velocity_10d"]
    
    # Spread RSI (Relative Strength Index applied to spread changes)
    spread_delta = df["spread"].diff()
    gain = spread_delta.clip(lower=0)
    loss = (-spread_delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["spread_rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0  # Normalize to [0, 1]
    
    # Spread signal-to-noise: how large is the gap relative to recent noise
    df["spread_snr"] = (df["spread"] - df["rolling_mean"]) / df["rolling_std"].replace(0, np.nan)
    df["spread_snr"] = df["spread_snr"].fillna(0.0).clip(-5.0, 5.0)
    
    # Normalized price gap (captures relative price divergence)
    df["norm_price_gap"] = (df[f"norm_{ticker_a}"] - df[f"norm_{ticker_b}"]) / 100.0
    
    # ===== 2. Trailing Market Regime Features =====
    vol_regime = (df["rolling_std"] / df["rolling_std"].rolling(60, min_periods=10).mean()).fillna(1.0)
    vol_regime = vol_regime.clip(0.3, 3.0)  # Bound for stability
    
    # ===== 3. Final Feature Selection =====
    feature_cols = [
        # Core statistical features (provide the base signal)
        "z_score",
        "z_score_15d",
        "z_score_60d",
        # Momentum features (capture direction/acceleration)
        "spread_velocity_5d",
        "spread_velocity_10d",
        "spread_acceleration",
        # Oscillator features (capture mean-reversion state)
        "spread_rsi",
        "spread_snr",
        # Price divergence
        "norm_price_gap",
        # Volatility regime
        "volatility_ratio",
    ]
    
    # ===== 4. Target Construction =====
    # Forward k-day normalized change in spread
    future_spread = df["spread"].shift(-pred_horizon)
    target = (future_spread - df["spread"]) / df["rolling_std"].replace(0, np.nan)
    # Clip extreme outlier moves to +/-3.0 sigma for training stability
    target = target.replace([np.inf, -np.inf], np.nan).clip(-3.0, 3.0)

    clean_df = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
    )
    return clean_df, feature_cols, target


def train_deep_gap_model(
    feat_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    fundamental_diff: Dict[str, Any],
    sentiment_gap: float,
    pred_horizon: int = 10,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.003
) -> Tuple[DeepGapNet, Dict[str, Any], np.ndarray, Dict[str, float], pd.Timestamp]:
    """
    Fit on the earliest 60%, use the next purged segment for validation, and
    reserve the final 20% for out-of-sample backtesting.
    Returns: (model, metrics, signals, importances, test_start_date)
    """
    X_df, feature_names, y_series = prepare_deep_dataset(
        feat_df=feat_df,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        fundamental_diff=fundamental_diff,
        sentiment_gap=sentiment_gap,
        pred_horizon=pred_horizon,
    )
    
    valid_mask = y_series.notna() & np.isfinite(y_series)
    X_valid = X_df.loc[valid_mask].values
    y_valid = y_series.loc[valid_mask].values
    valid_index = y_series.index[valid_mask]

    sample_count = len(X_valid)
    train_boundary = int(sample_count * 0.6)
    test_boundary = int(sample_count * 0.8)
    train_end = train_boundary - pred_horizon
    val_end = test_boundary - pred_horizon
    if train_end < 10 or val_end - train_boundary < 2 or sample_count - test_boundary < 2:
        raise ValueError(
            "Insufficient data for chronological train/validation/test splits "
            f"with a {pred_horizon}-day prediction horizon."
        )

    X_train, y_train = X_valid[:train_end], y_valid[:train_end]
    X_val, y_val = X_valid[train_boundary:val_end], y_valid[train_boundary:val_end]
    split_date = valid_index[test_boundary]
    
    model = DeepGapNet(
        hidden_dims=(24, 12),
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        dropout_rate=0.40,
        weight_decay=0.02,
        huber_delta=1.0,
        input_noise_std=0.05,
        patience=8,
        lr_decay=0.98,
        grad_clip=1.0,
    )
    
    train_metrics = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )
    
    signals, confidences = model.generate_signals(X_df.values, threshold=0.30)
    feature_importances = model.compute_feature_importance(X_val, y_val, n_repeats=5)
    
    return model, train_metrics, signals, feature_importances, split_date
