"""Interpretation of the learned decision-focused corrections (base configuration).

For the decision states visited by LA-DFL on the test episodes, the script records the learned
correction delta (in forecast-scale units) and the implied quantile level of each planning target,
and summarises them by lookahead offset, promotion status and the age of the store's stock.

python -m experiments.analyze_dfl --run results/base --out revision_1/figures
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch

from experiments.run_config import CONFIGS
from pirp.dfl import DFLTargetModel, row_features
from pirp.evaluation import get_pool, make_policy, run_episode, shutdown_pools
from pirp.forecasting import QUANTILE_LEVELS, quantile_at, sigma_from_quantiles
from pirp.network import SubsetRouting
from pirp.setup import EVAL_DAYS, HIST, make_instance, prepare


def _states(args):
    spec, inst, demand, fc, seed = args
    router = SubsetRouting(inst.dist, inst.n_vehicles, inst.vehicle_cap)
    pol = make_policy(spec, inst, router)
    out = run_episode(pol, inst, router, demand, fc, HIST, EVAL_DAYS, seed, record_states=True)
    return out["states"]


def implied_level(qs_row: np.ndarray, value: float) -> float:
    """Quantile level at which the predicted quantile curve reaches `value` (clamped to the grid)."""
    return float(np.interp(value, qs_row, QUANTILE_LEVELS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/base")
    ap.add_argument("--out", default="revision_1/figures")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    cfg = CONFIGS["base"]
    D = prepare(cfg)
    inst = make_instance(cfg, D["mean_store"])
    chosen = joblib.load(os.path.join(a.run, "chosen_policies.joblib"))
    spec = chosen["DFL"]
    model = DFLTargetModel.from_state(spec["dfl_state"])
    eps, fcs = D["episodes"]["test"][:a.episodes], D["forecasts"]["test"][:a.episodes]
    args = [(spec, inst, ep["demand"], fc, e) for e, (ep, fc) in enumerate(zip(eps, fcs))]
    states = list(get_pool(a.workers).map(_states, args))
    shutdown_pools()
    rows = []
    H = inst.horizon
    for e, (ep, fc, st) in enumerate(zip(eps, fcs, states)):
        for s in st:
            day = s["day"]
            X, mu, qs, B = fc["X"][day], fc["mu"][day], fc["qs"][day], s["B"]
            feats = row_features(X, mu, qs, B, model.mean_store, H)
            with torch.no_grad():
                xn = (torch.from_numpy(feats) - torch.as_tensor(model.x_mean)) / torch.as_tensor(model.x_std)
                delta = model.net(xn).numpy().reshape(inst.N, H)
            base = quantile_at(qs, model.alpha)
            sig = sigma_from_quantiles(qs)
            target = np.maximum(base + sig * delta, 0)
            for i in range(inst.N):
                for k in range(H):
                    rows.append({"episode": e, "day": day, "store": i, "k": k, "delta": delta[i, k],
                                 "target": target[i, k], "base": base[i, k], "mu": mu[i, k],
                                 "level": implied_level(qs[i, k], target[i, k]),
                                 "promo": float(X[i * H + k, 4] > 0), "old_stock": float(B[i, 0] > 0),
                                 "stock_days": B[i].sum() / model.mean_store[i]})
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.run, "dfl_corrections.csv"), index=False)
    df["stock_band"] = pd.cut(df.stock_days, [-0.01, 0.25, 0.75, 10], labels=["<0.25 d", "0.25-0.75 d", ">0.75 d"])
    summ = []
    for name, g in [("all", df)] + [(f"k={k}", g) for k, g in df.groupby("k")] + \
            [("promotion" if p else "no promotion", g) for p, g in df.groupby("promo")] + \
            [(f"stock {b}", g) for b, g in df.groupby("stock_band", observed=True)] + \
            [("units expiring today" if o else "no expiring units", g) for o, g in df.groupby("old_stock")]:
        summ.append({"group": name, "n": len(g), "delta_mean": g.delta.mean(), "delta_sd": g.delta.std(),
                     "level_mean": g.level.mean(), "target_over_mu": (g.target / g.mu.clip(lower=1e-6)).mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.run, "dfl_corrections_summary.csv"), index=False)
    lines = [f"{r.group} & {r.n:,} & {r.delta_mean:+.2f} ({r.delta_sd:.2f}) & {r.level_mean:.3f} & {r.target_over_mu:.2f} \\\\"
             for r in summ.itertuples()]
    tex = ("\\begin{tabular}{lrrrr}\n\\toprule\nTargets & $n$ & Correction $\\delta_\\theta$ (SD) & "
           "Implied quantile level & Target / mean forecast \\\\\n\\midrule\n" + "\n".join(lines)
           + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(a.out, "tab_dfl_corrections.tex"), "w").write(tex)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
