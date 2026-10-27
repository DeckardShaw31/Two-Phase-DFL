"""Tune, train and evaluate every policy for one configuration.

Usage
-----
python -m experiments.run_config --config base --out results/base
python -m experiments.run_config --config lifo --out results/lifo --reuse results/base   # evaluation only

Steps: (1) generate/load episodes and shared forecasts, (2) tune rule and lookahead
policies on validation episodes, (3) train the DFL targets (states visited by the tuned
quantile lookahead on training episodes; early stopping on validation), (4) evaluate all
policies once on the test episodes, (5) write episode-level results, paired tests,
solver diagnostics and a manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from pirp.evaluation import get_pool, holm, make_policy, paired_test, run_episode, run_many, shutdown_pools
from pirp.network import SubsetRouting
from pirp.setup import EVAL_DAYS, HIST, Config, config_dict, make_instance, prepare, variant

BASE = Config()
CONFIGS = {
    "base": BASE,
    "m2": variant(BASE, name="m2", shelf_life=2, horizon=2),
    "m4": variant(BASE, name="m4", shelf_life=4),
    "pu10": variant(BASE, name="pu10", p_lost=10.0),
    "pu40": variant(BASE, name="pu40", p_lost=40.0),
    "fleet_tight": variant(BASE, name="fleet_tight", fleet_ratio=1.10),
    "fleet_loose": variant(BASE, name="fleet_loose", fleet_ratio=1.60),
    "highvar": variant(BASE, name="highvar", nb_size=3.0),
    "route3": variant(BASE, name="route3", c_route=3.0),
    "lifo": variant(BASE, name="lifo", lifo_share=0.3),
}

LOOK = dict(kind="lookahead", time_limit=20.0, gap=0.005, integer_days=1)
GRIDS = {
    "BSP": [dict(kind="bsp", R=R, alpha=a) for R in (1, 2) for a in (0.8, 0.9, 0.95, 0.975, 0.99)],
    "BSP-EW": [dict(kind="bsp_ew", R=R, alpha=a) for R in (1, 2) for a in (0.8, 0.9, 0.95, 0.975, 0.99)],
    "UL": [dict(kind="ul", lam=l, alpha=a, beta=b) for l in (1, 2, 3) for a in (0.9, 0.95, 0.975, 0.99)
           for b in (0.7, 0.9, 0.95)],
    "PTO-Q": [dict(LOOK, target="quantile", param=a) for a in (0.7, 0.8, 0.9, 0.95, 0.975)],
    "Robust": [dict(LOOK, target="robust", param=g) for g in (0.25, 0.5, 1.0, 1.5)],
}
FIXED = {
    "PTO-Mean": dict(LOOK, target="mean"),
    "ScenPO-SS": dict(kind="lookahead", target="scenario", n_scen=10, time_limit=60.0, gap=0.005, integer_days=1),
    "PI": dict(LOOK, target="pi"),
}
PRIMARY = ["PTO-Q", "ScenPO-SS", "Robust", "BSP-EW", "UL"]


def log(msg: str, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def episode_tasks(spec, inst, D, split, key):
    eps, fcs = D["episodes"][split], D["forecasts"][split]
    return [(spec, inst, ep["demand"], fc, HIST, EVAL_DAYS, e, key) for e, (ep, fc) in enumerate(zip(eps, fcs))]


def evaluate(specs: dict, inst, D, split, key, workers):
    tasks, owners = [], []
    for name, spec in specs.items():
        t = episode_tasks(spec, inst, D, split, key)
        tasks += t
        owners += [name] * len(t)
    res = run_many(tasks, workers)
    rows = []
    for name, r, (spec, *_rest) in zip(owners, res, tasks):
        e = _rest[5]
        rows.append({"policy": name, "episode": e, **{k: v for k, v in r.items() if not isinstance(v, list)},
                     "store_fill": r["store_fill"], "store_waste": r["store_waste"], "store_lost": r["store_lost"]})
    return pd.DataFrame(rows)


def _collect(args):
    spec, inst, demand, fc, start, n_days, seed = args
    router = SubsetRouting(inst.dist, inst.n_vehicles, inst.vehicle_cap)
    pol = make_policy(spec, inst, router)
    out = run_episode(pol, inst, router, demand, fc, start, n_days, seed, record_states=True)
    return out["states"]


def collect_samples(spec, inst, D, n_eps, workers):
    eps, fcs = D["episodes"]["train"][:n_eps], D["forecasts"]["train"][:n_eps]
    args = [(spec, inst, ep["demand"], fc, HIST, EVAL_DAYS, 500 + e) for e, (ep, fc) in enumerate(zip(eps, fcs))]
    states = list(get_pool(workers).map(_collect, args))
    H = inst.horizon
    samples = []
    for ep, fc, st in zip(eps, fcs, states):
        for s in st:
            day = s["day"]
            t = HIST + day
            samples.append({"B": s["B"], "X": fc["X"][day], "mu": fc["mu"][day], "qs": fc["qs"][day],
                            "d": ep["demand"][:, t:t + H]})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base", choices=list(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--reuse", default=None, help="evaluate the policies tuned/trained in another run")
    ap.add_argument("--dfl-epochs", type=int, default=8)
    ap.add_argument("--dfl-episodes", type=int, default=60)
    ap.add_argument("--cache", default="results/cache")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, "run.log"), "a")
    t_start = time.time()
    log(f"config {cfg}", fh)
    D = prepare(cfg, a.cache)
    inst = make_instance(cfg, D["mean_store"])
    key = f"{cfg.name}-{inst.vehicle_cap}"
    log(f"Q={inst.vehicle_cap} store_cap={inst.store_cap.tolist()} mean_total={D['mean_store'].sum():.1f}", fh)

    pre_dfl = os.path.join(a.out, "chosen_pre_dfl.joblib")
    if a.reuse:
        chosen = joblib.load(os.path.join(a.reuse, "chosen_policies.joblib"))
        tuning = None
        dfl_hist = None
    else:
        # ---------------------------------------------------------------- tuning
        tune_rows = []
        chosen = {}
        if os.path.exists(pre_dfl):  # resume after an interrupted DFL stage
            chosen = joblib.load(pre_dfl)
            log("resumed tuned policies from chosen_pre_dfl.joblib", fh)
        for fam, grid in ({} if chosen else GRIDS).items():
            specs = {f"{fam}#{j}": s for j, s in enumerate(grid)}
            df = evaluate(specs, inst, D, "val", key, a.workers)
            agg = df.groupby("policy")["cost_total"].mean()
            for name, v in agg.items():
                tune_rows.append({"family": fam, "spec": json.dumps(specs[name]), "val_cost": v})
            best = agg.idxmin()
            chosen[fam] = specs[best]
            log(f"tuned {fam}: {specs[best]} val={agg.min():.1f}", fh)
        if tune_rows:
            tuning = pd.DataFrame(tune_rows)
            tuning.to_csv(os.path.join(a.out, "validation_tuning.csv"), index=False)
        chosen.update(FIXED)
        joblib.dump(chosen, pre_dfl)
        # ---------------------------------------------------------------- DFL
        from pirp.dfl import train_dfl
        alpha = chosen["PTO-Q"]["param"]
        samples = collect_samples(chosen["PTO-Q"], inst, D, a.dfl_episodes, a.workers)
        log(f"DFL: {len(samples)} training decision states, start alpha={alpha}", fh)

        def val_fn(state):
            spec = dict(LOOK, target="dfl", dfl_state=state)
            df = evaluate({"DFL": spec}, inst, D, "val", key, a.workers)
            return float(df["cost_total"].mean())

        model, dfl_hist = train_dfl(inst, samples, alpha, D["mean_store"], val_fn, epochs=a.dfl_epochs,
                                    resolve_every=2, n_workers=a.workers, log=lambda s: log(s, fh),
                                    integer_days=LOOK["integer_days"])
        chosen["DFL"] = dict(LOOK, target="dfl", dfl_state=model.state())
        pd.DataFrame(dfl_hist).to_csv(os.path.join(a.out, "dfl_training.csv"), index=False)
        joblib.dump(chosen, os.path.join(a.out, "chosen_policies.joblib"))

    # -------------------------------------------------------------------- test
    order = ["BSP", "BSP-EW", "UL", "PTO-Mean", "PTO-Q", "Robust", "ScenPO-SS", "DFL", "PI"]
    test = evaluate({k: chosen[k] for k in order}, inst, D, "test", key, a.workers)
    test.to_csv(os.path.join(a.out, "test_episodes.csv"), index=False)
    piv = test.pivot(index="episode", columns="policy", values="cost_total")
    tests = []
    for other in order:
        if other == "DFL":
            continue
        r = paired_test(piv["DFL"].values, piv[other].values)
        tests.append({"comparison": f"DFL - {other}", "primary": other in PRIMARY, **r})
    tdf = pd.DataFrame(tests)
    prim = tdf["primary"]
    tdf.loc[prim, "p_holm_t"] = holm(tdf.loc[prim, "p_t"].tolist())
    tdf.loc[prim, "p_holm_wilcoxon"] = holm(tdf.loc[prim, "p_wilcoxon"].tolist())
    tdf.to_csv(os.path.join(a.out, "paired_tests.csv"), index=False)
    summ = test.groupby("policy").agg(
        cost=("cost_total", "mean"), cost_sd=("cost_total", "std"), route=("cost_route", "mean"),
        hold=("cost_hold", "mean"), waste=("cost_waste", "mean"), lost=("cost_lost", "mean"),
        fill=("fill_rate", "mean"), waste_rate=("waste_rate", "mean"), routes=("routes", "mean"),
        visits=("visits", "mean"), delivered=("delivered", "mean"), carried=("carried_mean", "mean"),
        solver_secs=("solver_secs", "mean"), max_gap=("max_gap", "max"), fallbacks=("n_fallback", "sum"),
        nonoptimal=("n_nonoptimal", "sum")).reindex(order)
    summ.to_csv(os.path.join(a.out, "test_summary.csv"))
    log("\n" + summ.round(3).to_string(), fh)
    log("\n" + tdf.round(4).to_string(), fh)
    manifest = {
        "config": config_dict(cfg), "vehicle_cap": inst.vehicle_cap, "store_cap": inst.store_cap.tolist(),
        "mean_store_demand": D["mean_store"].tolist(), "episode_seeds": {"train": "1000+e", "val": "5000+e",
                                                                        "test": "9000+e"},
        "chosen": {k: {kk: vv for kk, vv in v.items() if kk != "dfl_state"} for k, v in chosen.items()},
        "reused_from": a.reuse, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "runtime_sec": time.time() - t_start,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    shutdown_pools()
    log(f"done in {time.time() - t_start:.0f}s", fh)


if __name__ == "__main__":
    main()
