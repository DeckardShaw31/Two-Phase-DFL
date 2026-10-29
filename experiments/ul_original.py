"""The deliver-up-to-level rule exactly as published by Crama et al. (2018), as a reference for UL_lambda.

The published rule uses one target service level TSL for both the visit trigger and the delivery
quantity, counts the total stock (no netting of units about to expire) and compares lambda = 1 with
lambda = m - 1. Here TSL is tuned on the validation episodes over the fine grid of the service
level used for UL_lambda (the paper fixes TSL = 0.9, which the grid contains), and the selected
rule is evaluated once on the test episodes of every configuration and on the fresh base-case test
episodes. As in the main study, lifo evaluates the rule tuned for base.

Writes results/<config>/ul_orig_val.csv and ul_orig_test_episodes.csv (policy "UL-orig"), and
results/base_fresh/ul_orig_test_episodes.csv.

python -m experiments.ul_original --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from experiments.fresh_test import fresh_data
from experiments.retune_ul import ORDER
from experiments.rule_frontier import ALPHAS
from experiments.run_config import CONFIGS, evaluate, log
from pirp.evaluation import shutdown_pools
from pirp.setup import make_instance, prepare


def grid(shelf_life: int) -> list[dict]:
    lams = sorted({1, shelf_life - 1})
    return [dict(kind="ul", lam=l, alpha=a, beta=a, net_waste=False) for l in lams for a in (0.8,) + ALPHAS]


def selected(results: str, c: str) -> dict:
    val = pd.read_csv(os.path.join(results, "base" if c == "lifo" else c, "ul_orig_val.csv"))
    return json.loads(val.loc[val.val_cost.idxmin(), "spec"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=ORDER)
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default="results/cache")
    a = ap.parse_args()
    for c in a.configs:
        run = os.path.join(a.results, c)
        cfg = CONFIGS[c]
        D = prepare(cfg, a.cache)
        inst = make_instance(cfg, D["mean_store"])
        key = f"{cfg.name}-{inst.vehicle_cap}"
        val_path = os.path.join(run, "ul_orig_val.csv")
        if c != "lifo" and not os.path.exists(val_path):
            specs = {json.dumps(s): s for s in grid(cfg.shelf_life)}
            val = evaluate(specs, inst, D, "val", key, a.workers).groupby("policy").cost_total.mean()
            pd.DataFrame({"spec": val.index, "val_cost": val.values}).sort_values("val_cost").to_csv(val_path, index=False)
        spec = selected(a.results, c)
        targets = [(run, D)]
        if c == "base":  # also the fresh test episodes of the primary hypotheses
            F = fresh_data(cfg, D, 100, a.cache)
            targets.append((os.path.join(a.results, "base_fresh"),
                            D | {"episodes": {"test": F["episodes"]}, "forecasts": {"test": F["forecasts"]}}))
        for out, data in targets:
            path = os.path.join(out, "ul_orig_test_episodes.csv")
            if not os.path.exists(path):
                evaluate({"UL-orig": spec}, inst, data, "test", key, a.workers).to_csv(path, index=False)
            t = pd.read_csv(path)
            ul = pd.read_csv(os.path.join(out, "test_episodes.csv")).query("policy == 'UL'").cost_total.mean()
            log(f"{os.path.basename(out)}: published UL (lambda={spec['lam']}, TSL={spec['alpha']}) "
                f"{t.cost_total.mean():.0f} (fill {100 * t.fill_rate.mean():.1f}%, waste {100 * t.waste_rate.mean():.2f}%) vs UL {ul:.0f}")
    shutdown_pools()


if __name__ == "__main__":
    main()
