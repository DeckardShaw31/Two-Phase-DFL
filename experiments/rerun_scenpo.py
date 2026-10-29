"""Re-evaluate ScenPO-SS with the scenario set of Jia et al. (2025) on the test episodes of every configuration.

The first implementation of ScenPO-SS drew ten scenarios from bands of the full, unequally spaced
quantile grid with equal probability, which over-weights high demand. The published method uses
the predicted deciles themselves as scenarios plus scenarios sampled between neighbouring deciles,
20 in total (pirp.policies.sample_scenarios_jia). ScenPO-SS is not tuned, so only the test
evaluation is repeated. Decisions that stop at the time limit are counted in the log; a machine
that sleeps during a run produces them, and such a configuration should be re-run.

Writes results/<config>/scenpo_jia_test_episodes.csv.

python -m experiments.rerun_scenpo --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from experiments.fresh_test import SCENPO
from experiments.retune_ul import ORDER
from experiments.run_config import CONFIGS, evaluate, log
from pirp.evaluation import shutdown_pools
from pirp.setup import make_instance, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[c for c in ORDER if c != "base"] + ["base"])
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    for c in a.configs:
        run = os.path.join(a.results, c)
        path = os.path.join(run, "scenpo_jia_test_episodes.csv")
        if os.path.exists(path):
            continue
        cfg = CONFIGS[c]
        D = prepare(cfg)
        inst = make_instance(cfg, D["mean_store"])
        key = f"{cfg.name}-{inst.vehicle_cap}"
        df = evaluate({"ScenPO-SS": SCENPO}, inst, D, "test", key, a.workers)
        df.to_csv(path, index=False)
        old = pd.read_csv(os.path.join(run, "test_episodes.csv"))
        old = old[old.policy == "ScenPO-SS"]
        log(f"{c}: ScenPO-SS jia {df.cost_total.mean():.0f} (fill {100 * df.fill_rate.mean():.1f}%, waste "
            f"{100 * df.waste_rate.mean():.2f}%) vs bands {old.cost_total.mean():.0f}; "
            f"nonoptimal {int(df.n_nonoptimal.sum())}, fallbacks {int(df.n_fallback.sum())}")
    shutdown_pools()


if __name__ == "__main__":
    main()
