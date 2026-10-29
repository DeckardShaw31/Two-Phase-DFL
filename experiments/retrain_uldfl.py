"""Retrain UL-DFL starting from the UL_lambda rule re-tuned on the fine grid (retune_ul.py).

UL-DFL starts training exactly at the rule it corrects and keeps that rule if learning does not
lower the closed-loop validation cost. The main study started from the coarse-tuned rule; this
script repeats the training (same seeds, epochs and model selection as run_config.py) from the
rule selected on the fine grid, evaluates the selected network and every seed's network once on
the test episodes, and compares them episode by episode with the re-tuned rule and with the other
policies of the finished run. As in the main study, lifo is not trained: it evaluates the networks
trained for base (under FIFO) under mixed issuing.

Writes results/<config>/uldfl_fine_training.csv, uldfl_fine_policies.joblib and
uldfl_fine_test_episodes.csv, and results/uldfl_fine_summary.csv. Requires retune_ul.py first.

python -m experiments.retrain_uldfl --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd

from experiments.retune_ul import ORDER
from experiments.run_config import CONFIGS, evaluate, log
from pirp.evaluation import paired_test, shutdown_pools
from pirp.rule_dfl import train_ul_dfl
from pirp.setup import EVAL_DAYS, HIST, make_instance, prepare

SEEDS = (0, 1, 2)
OTHERS = ["UL-DFL", "BSP-EW", "NN-DFL", "PTO-Q2", "ScenPO-SS", "DFL"]


def fine_rule(results: str, c: str) -> dict:
    val = pd.read_csv(os.path.join(results, "base" if c == "lifo" else c, "ul_retune_val.csv"))
    return json.loads(val.loc[val.val_cost.idxmin(), "spec"])


def train(c: str, results: str, inst, D, key: str, workers: int, epochs: int) -> dict:
    run = os.path.join(results, c)
    pol_path = os.path.join(run, "uldfl_fine_policies.joblib")
    chosen = joblib.load(pol_path) if os.path.exists(pol_path) else {}
    hist_path = os.path.join(run, "uldfl_fine_training.csv")
    hist_all = pd.read_csv(hist_path).to_dict("records") if os.path.exists(hist_path) else []
    spec = fine_rule(results, c)

    def val_fn(state):
        df = evaluate({"UL-DFL": dict(kind="ul_dfl", dfl_state=state)}, inst, D, "val", key, workers)
        return float(df["cost_total"].mean())

    for sd in SEEDS:
        if f"UL-DFL@s{sd}" in chosen:  # resume after an interruption
            continue
        model, hist = train_ul_dfl(inst, D, spec, val_fn, HIST, EVAL_DAYS, epochs=epochs, seed=sd,
                                   log=lambda s: log(f"{c} {s}"))
        hist_all += [dict(h, seed=sd) for h in hist]
        chosen[f"UL-DFL@s{sd}"] = dict(kind="ul_dfl", dfl_state=model.state())
        chosen[f"val@s{sd}"] = min(h["val_cost"] for h in hist)
        pd.DataFrame(hist_all).to_csv(hist_path, index=False)
        joblib.dump(chosen, pol_path)
    best = min(SEEDS, key=lambda sd: chosen[f"val@s{sd}"])
    chosen["UL-DFL"] = chosen[f"UL-DFL@s{best}"]
    joblib.dump(chosen, pol_path)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=ORDER)
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    sum_path = os.path.join(a.results, "uldfl_fine_summary.csv")
    rows = [r for r in (pd.read_csv(sum_path).to_dict("records") if os.path.exists(sum_path) else [])
            if r["config"] not in a.configs]
    for c in a.configs:
        run = os.path.join(a.results, c)
        cfg = CONFIGS[c]
        D = prepare(cfg)
        inst = make_instance(cfg, D["mean_store"])
        key = f"{cfg.name}-{inst.vehicle_cap}"
        if c == "lifo":  # trained under FIFO in base, as every lifo policy
            chosen = joblib.load(os.path.join(a.results, "base", "uldfl_fine_policies.joblib"))
        else:
            chosen = train(c, a.results, inst, D, key, a.workers, a.epochs)
        test_path = os.path.join(run, "uldfl_fine_test_episodes.csv")
        if os.path.exists(test_path):
            test = pd.read_csv(test_path)
        else:
            specs = {f"UL-DFL-fine{k[6:]}": v for k, v in chosen.items() if k.startswith("UL-DFL")}
            test = evaluate(specs, inst, D, "test", key, a.workers)
            test.to_csv(test_path, index=False)
        old = pd.read_csv(os.path.join(run, "test_episodes.csv"))
        rule = pd.read_csv(os.path.join(run, "ul_retune_test_episodes.csv"))
        piv = pd.concat([old, rule, test]).pivot(index="episode", columns="policy", values="cost_total")
        new = test[test.policy == "UL-DFL-fine"]
        hist_path = os.path.join(a.results, "base" if c == "lifo" else c, "uldfl_fine_training.csv")
        hist = pd.read_csv(hist_path)
        sel = hist.loc[hist.groupby("seed").val_cost.idxmin()]
        row = {"config": c, "rule": json.dumps({k: chosen["UL-DFL"]["dfl_state"][k] for k in ("lam", "alpha", "beta")}),
               "cost": float(new.cost_total.mean()), "fill": float(new.fill_rate.mean()),
               "waste_rate": float(new.waste_rate.mean()), "visits": float(new.visits.mean()),
               "cost_rule": float(piv["UL-fine"].mean()),
               "fill_rule": float(rule.fill_rate.mean()), "waste_rate_rule": float(rule.waste_rate.mean()),
               "selected_epochs": json.dumps(sel.epoch.astype(int).tolist())}
        for other in ["UL-fine"] + OTHERS:
            r = paired_test(piv["UL-DFL-fine"].values, piv[other].values)
            row.update({f"vs_{other}_{k}": v for k, v in r.items()})
        for sd in SEEDS:
            r = paired_test(piv[f"UL-DFL-fine@s{sd}"].values, piv["UL-fine"].values)
            row[f"seed{sd}_vs_rule_rel"] = r["mean_rel_diff"]
        rows.append(row)
        r = row
        log(f"{c}: UL-DFL(fine start) vs fine UL {100 * r['vs_UL-fine_mean_rel_diff']:+.2f}% "
            f"CI [{r['vs_UL-fine_ci_lo']:.0f}, {r['vs_UL-fine_ci_hi']:.0f}] p={r['vs_UL-fine_p_t']:.3f}; "
            f"waste {100 * r['waste_rate']:.2f}% vs {100 * r['waste_rate_rule']:.2f}%; epochs {r['selected_epochs']}")
        pd.DataFrame(rows).to_csv(sum_path, index=False)
    shutdown_pools()


if __name__ == "__main__":
    main()
