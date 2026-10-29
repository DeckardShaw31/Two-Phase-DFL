"""Base-case prototypes: decision-focused UL rule (UL-DFL) and two-level quantile lookahead.

python -m experiments.prototype_base --part ul --out results/base_proto
python -m experiments.prototype_base --part q2 --out results/base_proto
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from experiments.run_config import CONFIGS, LOOK, evaluate
from pirp.evaluation import paired_test, shutdown_pools
from pirp.setup import EVAL_DAYS, HIST, make_instance, prepare


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ul", "q2"], required=True)
    ap.add_argument("--out", default="results/base_proto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, f"{a.part}.log"), "a")
    cfg = CONFIGS["base"]
    D = prepare(cfg)
    inst = make_instance(cfg, D["mean_store"])
    key = f"{cfg.name}-{inst.vehicle_cap}"
    chosen = joblib.load("results/base/chosen_pre_dfl.joblib")

    if a.part == "ul":
        from pirp.rule_dfl import train_ul_dfl
        ul = chosen["UL"]

        def val_fn(state):
            df = evaluate({"UL-DFL": dict(kind="ul_dfl", dfl_state=state)}, inst, D, "val", key, a.workers)
            return float(df["cost_total"].mean())

        ul_val = float(evaluate({"UL": ul}, inst, D, "val", key, a.workers)["cost_total"].mean())
        log(f"UL {ul} val={ul_val:.2f}", fh)
        epochs = 2 if a.smoke else a.epochs
        model, hist = train_ul_dfl(inst, D, ul, val_fn, HIST, EVAL_DAYS, epochs=epochs, lr=a.lr, seed=a.seed,
                                   log=lambda s: log(s, fh))
        if abs(hist[0]["val_cost"] - ul_val) > 1e-6:
            log(f"WARNING: zero-correction UL-DFL val {hist[0]['val_cost']:.4f} != UL {ul_val:.4f}", fh)
        tag = f"ul_dfl_seed{a.seed}"
        pd.DataFrame(hist).to_csv(os.path.join(a.out, f"{tag}_training.csv"), index=False)
        joblib.dump(model.state(), os.path.join(a.out, f"{tag}_state.joblib"))
        if a.smoke:
            shutdown_pools()
            return
        specs = {"UL": ul, "UL-DFL": dict(kind="ul_dfl", dfl_state=model.state())}
    else:
        grid = [(a0, a1) for a0 in (0.8, 0.9, 0.95) for a1 in (0.5, 0.6, 0.7, 0.8, 0.9)]
        if a.smoke:
            grid = grid[:1]
        specs = {f"Q2#{j}": dict(LOOK, target="quantile2", param=g) for j, g in enumerate(grid)}
        val = evaluate(specs, inst, D, "val", key, a.workers)
        agg = val.groupby("policy")["cost_total"].mean()
        rows = [{"spec": json.dumps(specs[n]), "alpha0": specs[n]["param"][0], "alpha1": specs[n]["param"][1],
                 "val_cost": v} for n, v in agg.items()]
        pd.DataFrame(rows).sort_values("val_cost").to_csv(os.path.join(a.out, "q2_tuning.csv"), index=False)
        best = specs[agg.idxmin()]
        log(f"tuned LA-Quantile2: {best} val={agg.min():.2f}", fh)
        if a.smoke:
            shutdown_pools()
            return
        specs = {"LA-Quantile2": best, "UL": chosen["UL"]}
        tag = "q2"

    test = evaluate(specs, inst, D, "test", key, a.workers)
    shutdown_pools()
    test.to_csv(os.path.join(a.out, f"{tag}_test_episodes.csv"), index=False)
    log(test.groupby("policy")[["cost_total", "cost_route", "cost_hold", "cost_waste", "cost_lost", "fill_rate",
                                "waste_rate", "visits"]].mean().to_string(), fh)
    piv = test.pivot(index="episode", columns="policy", values="cost_total")
    name = "UL-DFL" if a.part == "ul" else "LA-Quantile2"
    r = paired_test(piv[name].values, piv["UL"].values)
    log(f"{name} - UL: {json.dumps(r)}", fh)


if __name__ == "__main__":
    main()
