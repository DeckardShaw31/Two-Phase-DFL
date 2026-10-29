"""Sensitivity of closed-loop lookahead costs to negligible perturbations of the planning targets.

LA-Quantile is re-run on the test episodes with its planning targets rounded to single
precision (relative perturbation below 6e-8). The MILP is deterministic, but it stops at the
first incumbent within the 0.5% gap, and several near-optimal plans exist; a different
day-0 plan changes the stock and hence every later decision. The episode-level cost changes
give the noise floor below which differences between lookahead policies are not interpretable.

python -m experiments.solver_sensitivity --results results
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from experiments.run_config import CONFIGS
from pirp.network import SubsetRouting
from pirp.evaluation import run_episode
from pirp.policies import LookaheadPolicy
from pirp.setup import EVAL_DAYS, HIST, make_instance, prepare


class Float32Targets(LookaheadPolicy):
    def planning_demand(self, ctx):
        return super().planning_demand(ctx).astype(np.float32).astype(float)


def _episode(args):
    cfg_name, e, spec, cache = args
    cfg = CONFIGS[cfg_name]
    D = prepare(cfg, cache)
    inst = make_instance(cfg, D["mean_store"])
    router = SubsetRouting(inst.dist, inst.n_vehicles, inst.vehicle_cap)
    pol = Float32Targets(inst, router, "quantile", param=spec["param"], time_limit=spec["time_limit"],
                         gap=spec["gap"], integer_days=spec["integer_days"])
    ep, fc = D["episodes"]["test"][e], D["forecasts"]["test"][e]
    return run_episode(pol, inst, router, ep["demand"], fc, HIST, EVAL_DAYS, e)["cost_total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for c in ["base", "net_c1", "net_rc1", "m2", "m4", "pu10", "pu40", "fleet_tight", "fleet_loose", "highvar",
                  "route3", "lifo", "fresh"]:
            run = os.path.join(a.results, c)
            if not os.path.exists(os.path.join(run, "test_episodes.csv")):
                continue
            chosen_path = os.path.join(run, "chosen_policies.joblib")
            if not os.path.exists(chosen_path):  # evaluation-only runs reuse the base policies
                chosen_path = os.path.join(a.results, "base", "chosen_policies.joblib")
            spec = joblib.load(chosen_path)["PTO-Q"]
            test = pd.read_csv(os.path.join(run, "test_episodes.csv"))
            ref = test[test.policy == "PTO-Q"].sort_values("episode")
            pert = list(ex.map(_episode, [(c, int(e), spec, a.cache) for e in ref.episode]))
            rel = 100 * (np.array(pert) - ref.cost_total.values) / ref.cost_total.values
            rows.append({"config": c, "n": len(rel), "changed": int(np.sum(np.abs(rel) > 1e-9)),
                         "mean_rel": rel.mean(), "sd_rel": rel.std(ddof=1), "se_mean_rel": rel.std(ddof=1) / np.sqrt(len(rel)),
                         "min_rel": rel.min(), "max_rel": rel.max()})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "solver_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
