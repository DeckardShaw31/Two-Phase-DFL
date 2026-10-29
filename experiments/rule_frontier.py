"""Can re-tuning the rules reproduce what UL-DFL learns? (waste/fill frontier on the test episodes)

Every parameterisation of UL_lambda on a fine grid, and of daily BSP and BSP-EW over a fine
quantile grid, is run on the test episodes of a finished configuration. The script reports
(i) the waste/fill-rate frontier the rules can reach, (ii) whether any parameterisation serves at
least UL-DFL's fill rate with no more waste, and (iii) the best rule in hindsight, i.e. tuned on
the test episodes themselves, compared with UL-DFL episode by episode. (iii) favours the rules:
UL-DFL is selected on validation episodes only.

python -m experiments.rule_frontier --config base --results results_v2 --out revision_1/figures
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from experiments.run_config import CONFIGS, evaluate
from pirp.evaluation import paired_test, shutdown_pools
from pirp.setup import make_instance, prepare

ALPHAS = (0.85, 0.875, 0.9, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99)
GRID = ([dict(kind="ul", lam=1, alpha=a, beta=b) for a in ALPHAS for b in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)]
        + [dict(kind="ul", lam=l, alpha=a, beta=b) for l in (2, 3) for a in (0.9, 0.95, 0.975, 0.99)
           for b in (0.7, 0.9, 0.95)]
        + [dict(kind=k, R=1, alpha=a) for k in ("bsp", "bsp_ew") for a in ALPHAS])


def frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Points not dominated in (higher fill, lower waste rate)."""
    d = df.sort_values(["fill", "waste_rate"], ascending=[False, True])
    keep, best = [], np.inf
    for r in d.itertuples():
        if r.waste_rate < best:
            keep.append(r.Index)
            best = r.waste_rate
    return df.loc[keep].sort_values("fill")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default="revision_1/figures")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    run = os.path.join(a.results, a.config)
    cfg = CONFIGS[a.config]
    D = prepare(cfg)
    inst = make_instance(cfg, D["mean_store"])
    key = f"{cfg.name}-{inst.vehicle_cap}"
    specs = {json.dumps(s): s for s in GRID}
    ep = evaluate(specs, inst, D, "test", key, a.workers)
    shutdown_pools()
    old = pd.read_csv(os.path.join(run, "test_episodes.csv"))
    rows = []
    for name, g in ep.groupby("policy"):
        s = specs[name]
        rows.append({"spec": name, "family": {"ul": "UL", "bsp": "BSP", "bsp_ew": "BSP-EW"}[s["kind"]],
                     "lam": s.get("lam"), "alpha": s["alpha"], "beta": s.get("beta"), "cost": g.cost_total.mean(),
                     "fill": g.fill_rate.mean(), "waste_rate": g.waste_rate.mean(), "hold": g.cost_hold.mean(),
                     "waste": g.cost_waste.mean(), "lost": g.cost_lost.mean(), "route": g.cost_route.mean()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, "rule_frontier.csv"), index=False)
    ul = old[old.policy == "UL-DFL"].sort_values("episode")
    f_ul, w_ul, c_ul = ul.fill_rate.mean(), ul.waste_rate.mean(), ul.cost_total.mean()
    # check: the tuned rule is reproduced exactly
    tuned = old[old.policy == "UL"].cost_total.mean()
    mine = df[(df.family == "UL") & (df.lam == 1) & (df.alpha == 0.95) & (df.beta == 0.9)].cost
    dom = df[(df.fill >= f_ul) & (df.waste_rate <= w_ul)]
    at_fill = df[df.fill >= f_ul]
    best = df.loc[df.cost.idxmin()]
    best_ep = ep[ep.policy == best.spec].sort_values("episode")
    t = paired_test(ul.cost_total.values, best_ep.cost_total.values)
    summary = {"config": a.config, "n_rules": len(df), "tuned_UL_check": [float(tuned), float(mine.iloc[0])],
               "uldfl_fill": f_ul, "uldfl_waste_rate": w_ul, "uldfl_cost": c_ul,
               "n_dominating": len(dom), "min_waste_at_uldfl_fill": float(at_fill.waste_rate.min()) if len(at_fill) else None,
               "best_rule": best.spec, "best_rule_cost": float(best.cost), "best_rule_fill": float(best.fill),
               "best_rule_waste_rate": float(best.waste_rate), "uldfl_vs_best_rel": t["mean_rel_diff"],
               "uldfl_vs_best_ci": [t["ci_lo"], t["ci_hi"]], "uldfl_vs_best_p": t["p_t"], "win_rate": t["win_rate"]}
    with open(os.path.join(run, "rule_frontier_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    if a.config == "base":
        fig_frontier(df, old, a.out)


def fig_frontier(df: pd.DataFrame, old: pd.DataFrame, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for fam, mk, col in (("BSP", "s", "#c9c9c9"), ("BSP-EW", "D", "#9e9e9e"), ("UL", "o", "#b5b5b5")):
        g = df[df.family == fam]
        ax.scatter(100 * g.fill, 100 * g.waste_rate, s=12, marker=mk, color=col, alpha=0.8,
                   label={"UL": r"UL$_\lambda$ (all parameters)", "BSP": "BSP (daily, all quantiles)",
                          "BSP-EW": "BSP-EW (daily, all quantiles)"}[fam])
    fr = frontier(df)
    ax.plot(100 * fr.fill, 100 * fr.waste_rate, color="#6b6b6b", lw=1, label="Frontier of the rules")
    for pol, mk, col, lab in (("UL", "^", "black", r"UL$_\lambda$ (tuned)"), ("UL-DFL", "*", "#4c72b0", "UL-DFL")):
        g = old[old.policy == pol]
        ax.scatter(100 * g.fill_rate.mean(), 100 * g.waste_rate.mean(), s=90 if mk == "*" else 40, marker=mk,
                   color=col, zorder=5, label=lab)
    ax.set_xlabel("Fill rate (%)")
    ax.set_ylabel("Waste rate (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_frontier.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
