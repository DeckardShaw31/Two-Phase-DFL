"""LightGBM point and multi-quantile forecasters shared by all forecast-based policies."""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from scipy.stats import norm

from .demand import make_features

QUANTILE_LEVELS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99])
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=4)


def build_dataset(episodes: list[dict], bases: np.ndarray, start: int, n_days: int, horizon: int):
    X, y = [], []
    for ep in episodes:
        N = ep["demand"].shape[0]
        for t in range(start, start + n_days):
            F = make_features(ep, t, horizon, bases)
            X.append(F)
            y.append(np.array([ep["demand"][i, t + k] for i in range(N) for k in range(horizon)]))
    return np.vstack(X), np.concatenate(y)


class Forecaster:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Forecaster":
        self.mean_model = lgb.LGBMRegressor(objective="regression", random_state=self.seed, **LGB_PARAMS)
        self.mean_model.fit(X, y, categorical_feature=[0])
        self.q_models = []
        for a in QUANTILE_LEVELS:
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=float(a), random_state=self.seed, **LGB_PARAMS)
            mdl.fit(X, y, categorical_feature=[0])
            self.q_models.append(mdl)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.maximum(self.mean_model.predict(X), 0.0)
        qs = np.stack([np.maximum(m.predict(X), 0.0) for m in self.q_models], axis=-1)
        qs = np.sort(qs, axis=-1)  # enforce non-crossing quantiles
        return mu, qs


def quantile_at(qs: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation of the predicted quantile grid at level alpha (clamped)."""
    a = np.clip(alpha, QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1])
    j = np.searchsorted(QUANTILE_LEVELS, a)
    j = np.clip(j, 1, len(QUANTILE_LEVELS) - 1)
    lo, hi = QUANTILE_LEVELS[j - 1], QUANTILE_LEVELS[j]
    w = (a - lo) / (hi - lo)
    return qs[..., j - 1] * (1 - w) + qs[..., j] * w


def sigma_from_quantiles(qs: np.ndarray) -> np.ndarray:
    """Robust scale estimate from the 10%-90% predicted quantile spread."""
    i10 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.1))[0][0])
    i90 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.9))[0][0])
    return np.maximum((qs[..., i90] - qs[..., i10]) / (2 * norm.ppf(0.9)), 1e-3)


def precompute_episode_forecasts(fc: Forecaster, ep: dict, bases: np.ndarray, start: int, n_days: int,
                                 horizon: int) -> dict:
    """Forecasts for every planning day of an episode: mu (T, N, H), qs (T, N, H, L), X (T, N*H, F)."""
    N = ep["demand"].shape[0]
    Xs = [make_features(ep, t, horizon, bases) for t in range(start, start + n_days)]
    X = np.vstack(Xs)
    mu, qs = fc.predict(X)
    T = n_days
    return {"mu": mu.reshape(T, N, horizon), "qs": qs.reshape(T, N, horizon, -1),
            "X": np.stack(Xs).reshape(T, N * horizon, -1)}
