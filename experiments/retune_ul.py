"""Re-tune UL_lambda on the fine grid using validation episodes only, and compare it with UL-DFL.

The main study selects UL_lambda on the coarse grid of run_config.GRIDS; the best rule in hindsight
on the test episodes (rule_frontier.py) suggests that part of UL-DFL's advantage reflects that grid.
Here UL_lambda is selected on the fine UL grid of rule_frontier.py (a superset of the coarse grid)
by closed-loop validation cost, evaluated once on the test episodes, and compared episode by
episode with UL-DFL and with the coarse-tuned UL_lambda of the finished run. As in the main study,
lifo is not tuned: it evaluates the rule selected for base (under FIFO) under mixed issuing.

Writes results/<config>/ul_retune_val.csv and ul_retune_test_episodes.csv, and
results/ul_retune_summary.csv.

python -m experiments.retune_ul --results results_v2 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from experiments.rule_frontier import GRID
from experiments.run_config import CONFIGS, evaluate, log
from pirp.evaluation import paired_test, shutdown_pools
from pirp.setup import make_instance, prepare

UL_GRID = [s for s in GRID if s["kind"] == "ul"]
ORDER = ["base", "net_c1", "net_rc1", "route3", "m2", "m4", "pu10", "pu40", "fleet_tight", "fleet_loose",
         "highvar", "lifo", "fresh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=ORDER)
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rows = []
    for c in a.configs:
        run = os.path.join(a.results, c)
        cfg = CONFIGS[c]
        D = prepare(cfg)
        inst = make_instance(cfg, D["mean_store"])
        key = f"{cfg.name}-{inst.vehicle_cap}"
        coarse = json.load(open(os.path.join(run, "manifest.json")))["chosen"]["UL"]
        val_path = os.path.join(a.results, "base" if c == "lifo" else c, "ul_retune_val.csv")
        if c == "lifo" or os.path.exists(val_path):  # lifo: tuned under FIFO in base, as every lifo policy
            val = pd.read_csv(val_path)
        else:
            specs = {json.dumps(s): s for s in UL_GRID}
            ep = evaluate(specs, inst, D, "val", key, a.workers)
            val = ep.groupby("policy")["cost_total"].mean().rename("val_cost").reset_index()
            val = val.rename(columns={"policy": "spec"}).sort_values("val_cost")
            val.to_csv(val_path, index=False)
        best = json.loads(val.loc[val.val_cost.idxmin(), "spec"])
        # check: the coarse selection's validation cost is reproduced
        old_val = pd.read_csv(os.path.join(a.results, "base" if c == "lifo" else c, "validation_tuning.csv"))
        old_val = old_val[old_val.family == "UL"]
        old_best = float(old_val.val_cost.min())
        mine = float(val.set_index("spec").val_cost.get(json.dumps(coarse), float("nan")))
        old = pd.read_csv(os.path.join(run, "test_episodes.csv"))
        test_path = os.path.join(run, "ul_retune_test_episodes.csv")
        if os.path.exists(test_path):
            test = pd.read_csv(test_path)
        else:
            test = evaluate({"UL-fine": best}, inst, D, "test", key, a.workers)
            test.to_csv(test_path, index=False)
        piv = old.pivot(index="episode", columns="policy", values="cost_total")
        fine = test.sort_values("episode").set_index("episode")
        r_dfl = paired_test(piv["UL-DFL"].values, fine.cost_total.reindex(piv.index).values)
        r_old = paired_test(fine.cost_total.reindex(piv.index).values, piv["UL"].values)
        r_ref = paired_test(piv["UL-DFL"].values, piv["UL"].values)
        f = fine.reindex(piv.index)
        g = old[old.policy == "UL-DFL"]
        rows.append({
            "config": c, "coarse": json.dumps({k: coarse[k] for k in ("lam", "alpha", "beta")}),
            "fine": json.dumps({k: best[k] for k in ("lam", "alpha", "beta")}),
            "val_coarse": old_best, "val_coarse_check": mine, "val_fine": float(val.val_cost.min()),
            "test_UL": float(piv["UL"].mean()), "test_UL_fine": float(f.cost_total.mean()),
            "test_UL_DFL": float(piv["UL-DFL"].mean()),
            "fill_UL_fine": float(f.fill_rate.mean()), "waste_rate_UL_fine": float(f.waste_rate.mean()),
            "fill_UL_DFL": float(g.fill_rate.mean()), "waste_rate_UL_DFL": float(g.waste_rate.mean()),
            **{f"dfl_vs_fine_{k}": v for k, v in r_dfl.items()},
            **{f"fine_vs_coarse_{k}": v for k, v in r_old.items()},
            **{f"dfl_vs_coarse_{k}": v for k, v in r_ref.items()},
        })
        log(f"{c}: coarse {rows[-1]['coarse']} -> fine {rows[-1]['fine']}; UL-DFL vs fine UL "
            f"{100 * r_dfl['mean_rel_diff']:+.2f}% CI [{r_dfl['ci_lo']:.0f}, {r_dfl['ci_hi']:.0f}]")
        pd.DataFrame(rows).to_csv(os.path.join(a.results, "ul_retune_summary.csv"), index=False)
    shutdown_pools()


if __name__ == "__main__":
    main()
