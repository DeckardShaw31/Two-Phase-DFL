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
    "net_c1": variant(BASE, name="net_c1", layout="C1"),
    "net_rc1": variant(BASE, name="net_rc1", layout="RC1"),
    "fresh": variant(BASE, name="fresh", demand_source="fresh"),
    "smoke": variant(BASE, name="smoke", n_train=6, n_val=3, n_test=3),
}

LOOK = dict(kind="lookahead", time_limit=20.0, gap=0.005, integer_days=1)
GRIDS = {
    "BSP": [dict(kind="bsp", R=R, alpha=a) for R in (1, 2) for a in (0.8, 0.9, 0.95, 0.975, 0.99)],
    "BSP-EW": [dict(kind="bsp_ew", R=R, alpha=a) for R in (1, 2) for a in (0.8, 0.9, 0.95, 0.975, 0.99)],
    "UL": [dict(kind="ul", lam=l, alpha=a, beta=b) for l in (1, 2, 3) for a in (0.9, 0.95, 0.975, 0.99)
           for b in (0.7, 0.9, 0.95)],
    "PTO-Q": [dict(LOOK, target="quantile", param=a) for a in (0.7, 0.8, 0.9, 0.95, 0.975)],
    "Robust": [dict(LOOK, target="robust", param=g) for g in (0.25, 0.5, 1.0, 1.5)],
    # ablation of what LA-DFL learns: one quantile level for the implemented day, one for later days
    "PTO-Q2": [dict(LOOK, target="quantile2", param=(a0, a1)) for a0 in (0.9, 0.95) for a1 in (0.6, 0.7, 0.8)],
}
FIXED = {
    "PTO-Mean": dict(LOOK, target="mean"),
    "ScenPO-SS": dict(kind="lookahead", target="scenario", n_scen=10, time_limit=60.0, gap=0.005, integer_days=1),
    "PI": dict(LOOK, target="pi"),
}
PRIMARY = ["PTO-Q", "ScenPO-SS", "Robust", "BSP-EW", "UL"]
PRIMARY_UL = ["UL", "BSP-EW", "ScenPO-SS", "DFL", "PTO-Q2"]


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
    ap.add_argument("--dfl-episodes", type=int, default=100)
    ap.add_argument("--dfl-seeds", default="0", help="LA-DFL training seeds; the best on validation is LA-DFL")
    ap.add_argument("--ul-dfl-seeds", default="0,1,2")
    ap.add_argument("--ul-dfl-epochs", type=int, default=30)
    ap.add_argument("--nn-dfl-seeds", default="0,1,2", help="NN-DFL ablation: neural orders without the rule")
    ap.add_argument("--nn-dfl-epochs", type=int, default=60)
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
        part = os.path.join(a.out, "chosen_policies.joblib")
        if os.path.exists(part):  # resume after an interrupted learning stage
            chosen.update({k: v for k, v in joblib.load(part).items() if k not in chosen})
        if "DFL" not in chosen:
            from pirp.dfl import train_dfl
            alpha = chosen["PTO-Q"]["param"]
            samples = collect_samples(chosen["PTO-Q"], inst, D, a.dfl_episodes, a.workers)
            log(f"DFL: {len(samples)} training decision states, start alpha={alpha}", fh)

            def val_fn(state):
                spec = dict(LOOK, target="dfl", dfl_state=state)
                df = evaluate({"DFL": spec}, inst, D, "val", key, a.workers)
                return float(df["cost_total"].mean())

            hist_all, best = [], None
            seeds = [int(x) for x in a.dfl_seeds.split(",")]
            for sd in seeds:
                model, hist = train_dfl(inst, samples, alpha, D["mean_store"], val_fn, epochs=a.dfl_epochs,
                                        resolve_every=2, n_workers=a.workers, log=lambda s: log(s, fh),
                                        integer_days=LOOK["integer_days"], seed=sd)
                hist_all += [dict(h, seed=sd) for h in hist]
                v = min(h["val_cost"] for h in hist)
                if len(seeds) > 1:
                    chosen[f"DFL@s{sd}"] = dict(LOOK, target="dfl", dfl_state=model.state())
                if best is None or v < best[0]:
                    best = (v, model)
            chosen["DFL"] = dict(LOOK, target="dfl", dfl_state=best[1].state())
            pd.DataFrame(hist_all).to_csv(os.path.join(a.out, "dfl_training.csv"), index=False)
            joblib.dump(chosen, part)
        if "UL-DFL" not in chosen:
            from pirp.rule_dfl import train_ul_dfl

            def ul_val_fn(state):
                df = evaluate({"UL-DFL": dict(kind="ul_dfl", dfl_state=state)}, inst, D, "val", key, a.workers)
                return float(df["cost_total"].mean())

            hist_all, best = [], None
            for sd in [int(x) for x in a.ul_dfl_seeds.split(",")]:
                model, hist = train_ul_dfl(inst, D, chosen["UL"], ul_val_fn, HIST, EVAL_DAYS, epochs=a.ul_dfl_epochs,
                                           seed=sd, log=lambda s: log(s, fh))
                hist_all += [dict(h, seed=sd) for h in hist]
                v = min(h["val_cost"] for h in hist)
                chosen[f"UL-DFL@s{sd}"] = dict(kind="ul_dfl", dfl_state=model.state())
                if best is None or v < best[0]:
                    best = (v, model)
            chosen["UL-DFL"] = dict(kind="ul_dfl", dfl_state=best[1].state())
            pd.DataFrame(hist_all).to_csv(os.path.join(a.out, "ul_dfl_training.csv"), index=False)
            joblib.dump(chosen, part)
        if "NN-DFL" not in chosen:  # ablation: same learner and inputs, no rule structure
            from pirp.rule_dfl import train_ul_dfl

            def nn_val_fn(state):
                df = evaluate({"NN-DFL": dict(kind="nn_dfl", dfl_state=state)}, inst, D, "val", key, a.workers)
                return float(df["cost_total"].mean())

            hist_all, best = [], None
            for sd in [int(x) for x in a.nn_dfl_seeds.split(",")]:
                model, hist = train_ul_dfl(inst, D, chosen["UL"], nn_val_fn, HIST, EVAL_DAYS, epochs=a.nn_dfl_epochs,
                                           seed=sd, log=lambda s: log(s, fh), kind="nn")
                hist_all += [dict(h, seed=sd) for h in hist]
                v = min(h["val_cost"] for h in hist)
                chosen[f"NN-DFL@s{sd}"] = dict(kind="nn_dfl", dfl_state=model.state())
                if best is None or v < best[0]:
                    best = (v, model)
            chosen["NN-DFL"] = dict(kind="nn_dfl", dfl_state=best[1].state())
            pd.DataFrame(hist_all).to_csv(os.path.join(a.out, "nn_dfl_training.csv"), index=False)
            joblib.dump(chosen, part)

    # -------------------------------------------------------------------- test
    order = [k for k in ["BSP", "BSP-EW", "UL", "UL-DFL", "NN-DFL", "PTO-Mean", "PTO-Q", "PTO-Q2", "Robust",
                         "ScenPO-SS", "DFL", "PI"] if k in chosen]
    order += sorted(k for k in chosen if "@s" in k)  # individual training seeds (robustness)
    test_path = os.path.join(a.out, "test_episodes.csv")
    old = pd.read_csv(test_path) if os.path.exists(test_path) else None
    if old is not None and old.groupby("policy").size().eq(len(D["episodes"]["test"])).all():
        # a finished run extended by new policies: the test episodes of the others are not re-run
        new = [k for k in order if k not in set(old.policy)]
        log(f"test: reusing {old.policy.nunique()} evaluated policies, evaluating {new}", fh)
        test = pd.concat([old] + ([evaluate({k: chosen[k] for k in new}, inst, D, "test", key, a.workers)]
                                  if new else []), ignore_index=True)
    else:
        test = evaluate({k: chosen[k] for k in order}, inst, D, "test", key, a.workers)
    test.to_csv(test_path, index=False)
    piv = test.pivot(index="episode", columns="policy", values="cost_total")
    tests = []
    for prop, primary in (("DFL", PRIMARY), ("UL-DFL", PRIMARY_UL)):
        if prop not in piv:
            continue
        fam = []
        for other in order:
            if other == prop or "@s" in other:
                continue
            r = paired_test(piv[prop].values, piv[other].values)
            fam.append({"comparison": f"{prop} - {other}", "primary": other in primary, **r})
        fdf = pd.DataFrame(fam)
        prim = fdf["primary"]
        fdf.loc[prim, "p_holm_t"] = holm(fdf.loc[prim, "p_t"].tolist())
        fdf.loc[prim, "p_holm_wilcoxon"] = holm(fdf.loc[prim, "p_wilcoxon"].tolist())
        tests.append(fdf)
    tdf = pd.concat(tests, ignore_index=True)
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
