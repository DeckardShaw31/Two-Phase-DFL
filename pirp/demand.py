"""Contextual demand generator for a fresh-food retail network.

Daily demand at store i on day t is negative binomial with mean

    mu[i,t] = base_i * dow[t] * L[i,t] * exp(b_disc * disc[i,t] + b_temp * (temp[t] - 20)) * hol[t]

where ``disc`` is an announced promotional discount, ``temp`` the regional
temperature, ``hol`` a holiday multiplier and ``L`` a latent store level that follows
a log-AR(1) process (unobserved; inferable from recent demand). The dispersion is
heteroscedastic: promotions lower the negative-binomial size parameter, so promoted
days are both larger and more variable. All parameters are hand-specified for this
controlled study; no retailer data are used.

Information available when planning on day t for target day t+k: store identity,
calendar of t+k, the announced discount and holiday flag of t+k, a temperature
forecast for t+k whose error grows with k, and demand history up to day t-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DOW = np.array([0.90, 0.85, 0.92, 1.00, 1.12, 1.30, 0.91])  # Mon..Sun


@dataclass(frozen=True)
class DemandParams:
    n_stores: int = 10
    base_low: float = 10.0
    base_high: float = 22.0
    nb_size: float = 8.0          # negative-binomial size on regular days (lower = noisier)
    promo_size_factor: float = 0.4  # size multiplier on promotion days
    promo_prob: float = 0.12
    b_disc: float = 2.0           # log-lift per unit discount (30% -> x1.82)
    b_temp: float = 0.025
    holiday_prob: float = 0.03
    holiday_lift: float = 1.45
    ar_rho: float = 0.92
    ar_sigma: float = 0.07
    temp_fc_sigma: tuple = (1.0, 2.0, 3.0, 3.5)


def store_bases(params: DemandParams, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(params.base_low, params.base_high, params.n_stores), 1)


def simulate_episode(params: DemandParams, bases: np.ndarray, n_days: int, seed: int) -> dict:
    """Simulate one independent episode of ``n_days`` days. Returns arrays (N, T) / (T,)."""
    rng = np.random.default_rng(seed)
    N, T = params.n_stores, n_days
    start_dow = rng.integers(0, 7)
    dow_idx = (start_dow + np.arange(T)) % 7
    day_of_year = rng.integers(0, 365)
    seasonal = 20 + 7 * np.sin(2 * np.pi * (day_of_year + np.arange(T)) / 365.0)
    temp = np.empty(T)
    e = 0.0
    for t in range(T):
        e = 0.7 * e + rng.normal(0, 2.0)
        temp[t] = seasonal[t] + e
    hol = (rng.random(T) < params.holiday_prob)
    promo = rng.random((N, T)) < params.promo_prob
    disc = np.where(promo, rng.choice([0.1, 0.2, 0.3], size=(N, T)), 0.0)
    logL = np.empty((N, T))
    x = rng.normal(0, params.ar_sigma / np.sqrt(1 - params.ar_rho ** 2), N)
    for t in range(T):
        x = params.ar_rho * x + rng.normal(0, params.ar_sigma, N)
        logL[:, t] = x
    mu = (bases[:, None] * DOW[dow_idx][None, :] * np.exp(logL)
          * np.exp(params.b_disc * disc + params.b_temp * (temp[None, :] - 20.0))
          * np.where(hol, params.holiday_lift, 1.0)[None, :])
    size = np.where(promo, params.nb_size * params.promo_size_factor, params.nb_size)
    p = size / (size + mu)
    demand = rng.negative_binomial(size, p).astype(float)
    # temperature forecasts issued each day for the next four days
    fc_err = np.stack([rng.normal(0, s, T) for s in params.temp_fc_sigma], axis=1)  # (T, 4)
    return {"demand": demand, "mu": mu, "size": size, "dow": dow_idx, "temp": temp, "holiday": hol.astype(float),
            "disc": disc, "temp_fc_err": fc_err}


def make_features(ep: dict, t: int, horizon: int, bases: np.ndarray) -> np.ndarray:
    """Feature rows for every (store, k) planned on day t: shape (N*horizon, F), store-major."""
    d = ep["demand"]
    N = d.shape[0]
    rows = []
    hist7 = d[:, t - 7:t]
    hist28 = d[:, t - 28:t]
    m7 = hist7.mean(1)
    m28 = hist28.mean(1)
    sd28 = hist28.std(1)
    last = d[:, t - 1]
    for i in range(N):
        for k in range(horizon):
            tk = t + k
            dw = ep["dow"][tk]
            same_dow = d[i, tk - 7] if tk - 7 <= t - 1 else d[i, tk - 14]
            temp_fc = ep["temp"][tk] + ep["temp_fc_err"][t, min(k, ep["temp_fc_err"].shape[1] - 1)]
            onehot = np.zeros(7)
            onehot[dw] = 1.0
            rows.append(np.concatenate([[i, bases[i], k, ep["disc"][i, tk], float(ep["disc"][i, tk] > 0),
                                         ep["holiday"][tk], temp_fc, last[i], m7[i], m28[i], sd28[i], same_dow,
                                         m7[i] / max(m28[i], 1e-6)], onehot]))
    return np.asarray(rows)


FEATURE_NAMES = ["store", "base", "k", "disc", "promo", "holiday", "temp_fc", "lag1", "mean7", "mean28", "sd28",
                 "same_dow", "trend"] + [f"dow{j}" for j in range(7)]
