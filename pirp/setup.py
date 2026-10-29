"""Experiment configuration, instance construction and cached data/forecast preparation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace

import joblib
import numpy as np

from .demand import DemandParams, simulate_episode, store_bases
from .forecasting import Forecaster, build_dataset, precompute_episode_forecasts
from .network import SubsetRouting, build_network
from .policies import Instance
from .simulator import CostParams

HIST = 35        # history days before the first decision (feature warm-up)
EVAL_DAYS = 21   # closed-loop decision days per episode
TAIL = 4         # extra simulated days so that lookahead forecasts and PI targets exist


@dataclass(frozen=True)
class Config:
    name: str = "base"
    layout: str = "R1"
    n_stores: int = 10
    n_vehicles: int = 2
    fleet_ratio: float = 1.25      # K*Q / mean total daily demand
    store_cap_days: float = 3.0    # shelf capacity in days of mean store demand
    shelf_life: int = 3
    horizon: int = 3
    h: float = 1.0
    p_waste: float = 5.0
    p_lost: float = 20.0
    c_route: float = 1.0
    nb_size: float = 8.0
    lifo_share: float = 0.0
    n_train: int = 200
    n_val: int = 50
    n_test: int = 100
    demand_source: str = "synthetic"   # or "fresh": FreshRetailNet-50K episodes (pirp/fresh.py)

    def demand_params(self) -> DemandParams:
        return DemandParams(n_stores=self.n_stores, nb_size=self.nb_size)

    def demand_key(self) -> str:
        from .forecasting import QUANTILE_LEVELS
        return hashlib.md5(json.dumps([self.n_stores, self.nb_size, self.n_train, self.n_val, self.n_test,
                                       self.horizon, QUANTILE_LEVELS.tolist()]
                                      + ([self.demand_source] if self.demand_source != "synthetic" else [])
                                      ).encode()).hexdigest()[:10]


def episodes(cfg: Config, split: str) -> list[dict]:
    dp = cfg.demand_params()
    bases = store_bases(dp)
    n, off = {"train": (cfg.n_train, 1000), "val": (cfg.n_val, 5000), "test": (cfg.n_test, 9000)}[split]
    T = HIST + EVAL_DAYS + TAIL
    return [simulate_episode(dp, bases, T, seed=off + e) for e in range(n)]


def prepare(cfg: Config, cache_dir: str = "results/cache") -> dict:
    """Generate episodes, fit the shared forecaster and precompute forecasts (cached by demand key)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"data_{cfg.demand_key()}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    dp = cfg.demand_params()
    bases = store_bases(dp)
    if cfg.demand_source == "fresh":
        from .fresh import build_episodes
        raw = os.environ.get("FRESH_DIR", "data/freshretailnet_raw")
        data = build_episodes(raw, bases, {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}, HIST)
    else:
        data = {s: episodes(cfg, s) for s in ("train", "val", "test")}
    H = cfg.horizon
    X, y = build_dataset(data["train"], bases, start=28, n_days=HIST + EVAL_DAYS + TAIL - 28 - H + 1, horizon=H)
    fc = Forecaster(seed=0).fit(X, y)
    fcs = {s: [precompute_episode_forecasts(fc, ep, bases, HIST, EVAL_DAYS, H) for ep in data[s]]
           for s in data}
    mean_store = np.mean([ep["mu"][:, HIST:HIST + EVAL_DAYS].mean(1) for ep in data["train"]], axis=0)
    out = {"episodes": data, "forecasts": fcs, "bases": bases, "mean_store": mean_store, "forecaster": fc,
           "train_xy": (X, y)}
    joblib.dump(out, path, compress=3)
    return out


def make_instance(cfg: Config, mean_store: np.ndarray) -> Instance:
    _, dist = build_network(cfg.layout, cfg.n_stores)
    Q = float(np.round(cfg.fleet_ratio * mean_store.sum() / cfg.n_vehicles, 0))
    cap = np.round(cfg.store_cap_days * mean_store, 0)
    costs = CostParams(h=cfg.h, p_waste=cfg.p_waste, p_lost=cfg.p_lost, c_route=cfg.c_route)
    router = SubsetRouting(dist, cfg.n_vehicles, Q)
    return Instance(dist=dist, store_cap=cap, n_vehicles=cfg.n_vehicles, vehicle_cap=Q, shelf_life=cfg.shelf_life,
                    horizon=cfg.horizon, costs=costs, tour_cost=router.tour_cost, lifo_share=cfg.lifo_share)


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)


def variant(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)
