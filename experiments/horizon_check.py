"""Is the lookahead horizon too short? LA-Quantile2 with H = 4 and 5 days (base case).

The integrated policies plan H = 3 days. A longer horizon lets the plan see more of the future
routing and demand, at the price of larger models with more relaxed (non-implemented) days. For
each H the forecaster is refitted for H offsets, LA-Quantile2 is re-tuned on the validation
episodes over the grid of run_config.py, and the selected policy is evaluated once on the fresh
base-case test episodes (same demand paths as results/base_fresh), where it is compared episode by
episode with the H = 3 policies.

Writes results/base_fresh/horizon_check.csv (episodes), horizon_tuning.csv and horizon_summary.csv.

python -m experiments.horizon_check --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from experiments.fresh_test import fresh_data
from experiments.run_config import CONFIGS, GRIDS, evaluate, log
from pirp.evaluation import paired_test, shutdown_pools
from pirp.setup import make_instance, prepare, variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=[4, 5])
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cache", default="results/cache")
    a = ap.parse_args()
    out = os.path.join(a.results, "base_fresh")
    ep_path, tune_path = os.path.join(out, "horizon_check.csv"), os.path.join(out, "horizon_tuning.csv")
    eps = pd.read_csv(ep_path) if os.path.exists(ep_path) else pd.DataFrame()
    tune = pd.read_csv(tune_path) if os.path.exists(tune_path) else pd.DataFrame()
    for H in a.horizons:
        name = f"PTO-Q2-H{H}"
        if len(eps) and (eps.policy == name).sum() == a.n:
            continue
        cfg = variant(CONFIGS["base"], name=f"base_h{H}", horizon=H)
        D = prepare(cfg, a.cache)
        inst = make_instance(cfg, D["mean_store"])
        key = f"{cfg.name}-{inst.vehicle_cap}"
        specs = {f"{name}#{j}": s for j, s in enumerate(GRIDS["PTO-Q2"])}
        val = evaluate(specs, inst, D, "val", key, a.workers).groupby("policy").cost_total.mean()
        rows = pd.DataFrame({"policy": name, "spec": [json.dumps(specs[k]) for k in val.index], "val_cost": val.values})
        tune = pd.concat([tune, rows], ignore_index=True)
        tune.to_csv(tune_path, index=False)
        best = specs[val.idxmin()]
        log(f"H={H}: tuned LA-Quantile2 {best['param']} val={val.min():.0f}")
        F = fresh_data(cfg, D, a.n, a.cache)
        E = {"episodes": {"test": F["episodes"]}, "forecasts": {"test": F["forecasts"]}}
        df = evaluate({name: best}, inst, D | E, "test", key, a.workers)
        eps = pd.concat([eps, df], ignore_index=True)
        eps.to_csv(ep_path, index=False)
        log(f"H={H}: test {df.cost_total.mean():.0f}, nonoptimal {int(df.n_nonoptimal.sum())}, "
            f"fallbacks {int(df.n_fallback.sum())}, secs/decision {df.solver_secs.mean() / 21:.2f}")
    shutdown_pools()
    base = pd.read_csv(os.path.join(out, "test_episodes.csv"))
    piv = pd.concat([base, eps]).pivot(index="episode", columns="policy", values="cost_total")
    rows = []
    for H in a.horizons:
        name = f"PTO-Q2-H{H}"
        g = eps[eps.policy == name]
        spec = json.loads(tune[tune.policy == name].sort_values("val_cost").iloc[0].spec)
        row = {"horizon": H, "param": spec["param"], "cost": g.cost_total.mean(), "fill": g.fill_rate.mean(),
               "waste_rate": g.waste_rate.mean(), "visits": g.visits.mean() / 21,
               "secs_per_decision": g.solver_secs.mean() / 21, "max_gap": g.max_gap.max(),
               "nonoptimal": int(g.n_nonoptimal.sum()), "fallbacks": int(g.n_fallback.sum())}
        for other in ("PTO-Q2", "UL-DFL", "UL"):
            r = paired_test(piv[name].values, piv[other].values)
            row.update({f"vs_{other}_rel": r["mean_rel_diff"], f"vs_{other}_ci_lo": r["ci_lo"],
                        f"vs_{other}_ci_hi": r["ci_hi"], f"vs_{other}_p": r["p_t"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "horizon_summary.csv"), index=False)
    log("\n" + pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
