"""Evaluate the finished base-case policies on fresh test episodes (primary hypotheses).

The synthetic test demand paths (seeds 9000+e) had already been evaluated in an earlier round of
the study, and base-case episodes 0-39 while UL-DFL was prototyped. This script draws 100 new test
episodes from the same demand model (seeds 20000+e, disjoint from the training, validation and
earlier test seeds), forecasts them with the forecaster fitted on the training episodes, and
evaluates every tuned or trained base-case policy once on them. Nothing is tuned or trained here.

Policies (names as in the manuscript):
  UL, UL-DFL      UL_lambda re-tuned on the fine grid (retune_ul.py) and UL-DFL trained from it
                  (retrain_uldfl.py); UL-coarse and UL-DFL-coarse are the coarse-grid versions
  ScenPO-SS       scenario set of Jia et al. (2025), 20 scenarios (sample_scenarios_jia);
                  ScenPO-SS-bands is the first implementation (10 scenarios, band sampling)
  all others      as tuned and trained in run_config.py

Writes results/base_fresh/{test_episodes,test_summary,paired_tests}.csv and manifest.json.

python -m experiments.fresh_test --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from experiments.run_config import CONFIGS, evaluate, log, summarise
from pirp.demand import simulate_episode
from pirp.evaluation import shutdown_pools
from pirp.forecasting import precompute_episode_forecasts
from pirp.setup import EVAL_DAYS, HIST, TAIL, make_instance, prepare

SEED_OFFSET = 20000
SCENPO = dict(kind="lookahead", target="scenario", sampler="jia", n_scen=20, time_limit=60.0, gap=0.005,
              integer_days=1)
ORDER = ["BSP", "BSP-EW", "UL", "UL-DFL", "NN-DFL", "PTO-Mean", "PTO-Q", "PTO-Q2", "Robust", "ScenPO-SS", "DFL",
         "PI", "UL-coarse", "UL-DFL-coarse", "ScenPO-SS-bands"]


def fresh_data(cfg, D: dict, n: int, cache: str) -> dict:
    path = os.path.join(cache, f"fresh_test_{cfg.demand_key()}_{SEED_OFFSET}_{n}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    T = HIST + EVAL_DAYS + TAIL
    eps = [simulate_episode(cfg.demand_params(), D["bases"], T, seed=SEED_OFFSET + e) for e in range(n)]
    fcs = [precompute_episode_forecasts(D["forecaster"], ep, D["bases"], HIST, EVAL_DAYS, cfg.horizon) for ep in eps]
    out = {"episodes": eps, "forecasts": fcs}
    joblib.dump(out, path, compress=3)
    return out


def policies(results: str) -> dict:
    run = os.path.join(results, "base")
    old = joblib.load(os.path.join(run, "chosen_policies.joblib"))
    fine = joblib.load(os.path.join(run, "uldfl_fine_policies.joblib"))
    val = pd.read_csv(os.path.join(run, "ul_retune_val.csv"))
    specs = {k: old[k] for k in ("BSP", "BSP-EW", "NN-DFL", "PTO-Mean", "PTO-Q", "PTO-Q2", "Robust", "DFL", "PI")}
    specs.update({"UL": json.loads(val.loc[val.val_cost.idxmin(), "spec"]), "UL-DFL": fine["UL-DFL"],
                  "ScenPO-SS": SCENPO, "UL-coarse": old["UL"], "UL-DFL-coarse": old["UL-DFL"],
                  "ScenPO-SS-bands": old["ScenPO-SS"]})
    specs.update({k: fine[k] for k in fine if k.startswith("UL-DFL@")})
    specs.update({k: old[k] for k in old if k.startswith("NN-DFL@")})
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=None, help="evaluate only these policies (smoke test)")
    ap.add_argument("--cache", default="results/cache")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "base_fresh")
    os.makedirs(out, exist_ok=True)
    cfg = CONFIGS["base"]
    D = prepare(cfg, a.cache)
    inst = make_instance(cfg, D["mean_store"])
    key = f"{cfg.name}-{inst.vehicle_cap}"
    F = fresh_data(cfg, D, a.n, a.cache)
    E = {"episodes": {"test": F["episodes"]}, "forecasts": {"test": F["forecasts"]}}
    specs = policies(a.results)
    order = [k for k in ORDER if k in specs] + sorted(k for k in specs if "@s" in k)
    if a.only:
        order = [k for k in order if k in a.only]
    path = os.path.join(out, "test_episodes.csv")
    test = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    for name in order:  # one policy at a time, saved after each, so an interrupted run resumes
        if len(test) and (test.policy == name).sum() == a.n:
            continue
        log(f"fresh test: {name}")
        df = evaluate({name: specs[name]}, inst, D | E, "test", key, a.workers)
        test = pd.concat([test[test.policy != name] if len(test) else test, df], ignore_index=True)
        test.to_csv(path, index=False)
    shutdown_pools()
    summ, _ = summarise(test, [k for k in order if k in set(test.policy)], out)
    log("\n" + summ[["cost", "fill", "waste_rate", "visits", "max_gap", "fallbacks", "nonoptimal"]].round(4).to_string())
    manifest = {"config": "base", "episode_seeds": {"test": f"{SEED_OFFSET}+e"}, "n_test": a.n,
                "chosen": {k: {kk: vv for kk, vv in v.items() if kk != "dfl_state"} for k, v in specs.items()},
                "note": "fresh test episodes; policies tuned and trained in results/base, nothing re-tuned here"}
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


if __name__ == "__main__":
    main()
