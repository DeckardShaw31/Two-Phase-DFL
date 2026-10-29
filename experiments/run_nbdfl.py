"""Train and evaluate the neural base-stock ablation NB-DFL in every configuration.

NB-DFL separates the two things NN-DFL removes from UL-DFL. It keeps the order-up-to structure
(the network outputs an order-up-to level and the store receives that level minus its usable
stock) but neither the forecast quantile from which UL-DFL starts nor the rule's visit trigger.
Training is that of NN-DFL in run_config.py: same inputs, differentiable episode simulation, loss,
three seeds, 60 epochs and selection on closed-loop validation cost. As in the main study, lifo is
not trained: it evaluates the networks trained for base under mixed issuing.

Writes results/<config>/nbdfl_training.csv, nbdfl_policies.joblib and nbdfl_test_episodes.csv.

python -m experiments.run_nbdfl --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

from experiments.retrain_uldfl import SEEDS, fine_rule
from experiments.retune_ul import ORDER
from experiments.run_config import CONFIGS, evaluate, log
from pirp.evaluation import shutdown_pools
from pirp.rule_dfl import train_ul_dfl
from pirp.setup import EVAL_DAYS, HIST, make_instance, prepare


def train(c: str, results: str, inst, D, key: str, workers: int, epochs: int) -> dict:
    run = os.path.join(results, c)
    pol_path = os.path.join(run, "nbdfl_policies.joblib")
    chosen = joblib.load(pol_path) if os.path.exists(pol_path) else {}
    hist_path = os.path.join(run, "nbdfl_training.csv")
    hist_all = pd.read_csv(hist_path).to_dict("records") if os.path.exists(hist_path) else []

    def val_fn(state):
        df = evaluate({"NB-DFL": dict(kind="nb_dfl", dfl_state=state)}, inst, D, "val", key, workers)
        return float(df["cost_total"].mean())

    for sd in SEEDS:
        if f"NB-DFL@s{sd}" in chosen:  # resume after an interruption
            continue
        # the rule specification only supplies the feature layout; NB-DFL uses none of its terms
        model, hist = train_ul_dfl(inst, D, fine_rule(results, c), val_fn, HIST, EVAL_DAYS, epochs=epochs,
                                   seed=sd, log=lambda s: log(f"{c} {s}"), kind="nb")
        hist_all += [dict(h, seed=sd) for h in hist]
        chosen[f"NB-DFL@s{sd}"] = dict(kind="nb_dfl", dfl_state=model.state())
        chosen[f"val@s{sd}"] = min(h["val_cost"] for h in hist)
        pd.DataFrame(hist_all).to_csv(hist_path, index=False)
        joblib.dump(chosen, pol_path)
    best = min(SEEDS, key=lambda sd: chosen[f"val@s{sd}"])
    chosen["NB-DFL"] = chosen[f"NB-DFL@s{best}"]
    joblib.dump(chosen, pol_path)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=ORDER)
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    for c in a.configs:
        run = os.path.join(a.results, c)
        cfg = CONFIGS[c]
        D = prepare(cfg)
        inst = make_instance(cfg, D["mean_store"])
        key = f"{cfg.name}-{inst.vehicle_cap}"
        if c == "lifo":  # trained under FIFO in base, as every lifo policy
            chosen = joblib.load(os.path.join(a.results, "base", "nbdfl_policies.joblib"))
        else:
            chosen = train(c, a.results, inst, D, key, a.workers, a.epochs)
        test_path = os.path.join(run, "nbdfl_test_episodes.csv")
        if not os.path.exists(test_path):
            specs = {k: v for k, v in chosen.items() if k.startswith("NB-DFL")}
            evaluate(specs, inst, D, "test", key, a.workers).to_csv(test_path, index=False)
        test = pd.read_csv(test_path)
        old = pd.read_csv(os.path.join(run, "test_episodes.csv")).groupby("policy").cost_total.mean()
        nb = test[test.policy == "NB-DFL"]
        log(f"{c}: NB-DFL {nb.cost_total.mean():.0f} (fill {100 * nb.fill_rate.mean():.1f}%, waste "
            f"{100 * nb.waste_rate.mean():.2f}%) vs UL-DFL {old['UL-DFL']:.0f}, NN-DFL {old['NN-DFL']:.0f}, UL {old['UL']:.0f}")
    shutdown_pools()


if __name__ == "__main__":
    main()
