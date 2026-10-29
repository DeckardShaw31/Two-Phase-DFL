"""Interpretation of the learned order-up-to corrections of UL-DFL (base configuration).

For the decision states visited by UL-DFL on the test episodes, the script records the learned
correction delta of the order-up-to level (in units of the forecast scale of the lambda-day demand)
and the implied service level Phi(z_alpha + delta), and summarises them over the store-days on
which the rule triggers a delivery (the correction has no effect on the others), by promotion
status, weekday and the age of the store's stock.

python -m experiments.analyze_uldfl --run results_v2/base --out revision_1/figures
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from experiments.run_config import CONFIGS
from pirp.evaluation import get_pool, make_policy, run_episode, shutdown_pools
from pirp.network import SubsetRouting
from pirp.rule_dfl import ULDFLModel, rule_terms, static_features, stock_features
from pirp.setup import EVAL_DAYS, HIST, make_instance, prepare

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _states(args):
    spec, inst, demand, fc, seed = args
    router = SubsetRouting(inst.dist, inst.n_vehicles, inst.vehicle_cap)
    pol = make_policy(spec, inst, router)
    out = run_episode(pol, inst, router, demand, fc, HIST, EVAL_DAYS, seed, record_states=True)
    return out["states"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results_v2/base")
    ap.add_argument("--config", default="base")
    ap.add_argument("--out", default="revision_1/figures")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    D = prepare(cfg)
    inst = make_instance(cfg, D["mean_store"])
    spec = joblib.load(os.path.join(a.run, "chosen_policies.joblib"))["UL-DFL"]
    md = ULDFLModel.from_state(spec["dfl_state"])
    eps, fcs = D["episodes"]["test"], D["forecasts"]["test"]
    args = [(spec, inst, ep["demand"], fc, e) for e, (ep, fc) in enumerate(zip(eps, fcs))]
    states = list(get_pool(a.workers).map(_states, args))
    shutdown_pools()
    H, z = inst.horizon, norm.ppf(md.alpha)
    rows = []
    for e, (fc, st) in enumerate(zip(fcs, states)):
        for s in st:
            day, B = s["day"], s["B"]
            X, mu, qs = fc["X"][day], fc["mu"][day], fc["qs"][day]
            S, scale, thr, _ = rule_terms(mu, qs, md.lam, md.alpha, md.beta)
            feats = np.column_stack([static_features(X, mu, qs, md.mean_store, H), stock_features(B, md.mean_store)])
            delta = md.delta(feats)
            stock = B.sum(1)
            for i in range(inst.N):
                x0 = X[i * H]
                rows.append({"episode": e, "day": day, "store": i, "delta": delta[i],
                             "level": norm.cdf(z + delta[i]), "trigger": bool(stock[i] < thr[i]),
                             "S_change": scale[i] * delta[i] / md.mean_store[i],
                             "promo": bool(x0[4] > 0), "dow": DAYS[int(np.argmax(x0[13:20]))],
                             "expiring": bool(B[i, 0] > 1e-9), "stock_days": stock[i] / md.mean_store[i]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.run, "uldfl_corrections.csv"), index=False)
    v = df[df.trigger].copy()
    v["stock_band"] = pd.cut(v.stock_days, [-0.01, 0.25, 0.5, 10], labels=["<0.25 d", "0.25-0.5 d", ">0.5 d"])
    groups = [("all delivery decisions", v)]
    groups += [("promotion" if p else "no promotion", g) for p, g in v.groupby("promo")]
    groups += [(d, v[v.dow == d]) for d in DAYS if (v.dow == d).any()]
    groups += [(f"stock {b}", g) for b, g in v.groupby("stock_band", observed=True)]
    groups += [("units expiring today" if o else "no expiring units", g) for o, g in v.groupby("expiring")]
    summ = pd.DataFrame([{"group": n, "n": len(g), "delta_mean": g.delta.mean(), "delta_sd": g.delta.std(),
                          "level_mean": g.level.mean(), "S_change": g.S_change.mean()} for n, g in groups])
    summ.to_csv(os.path.join(a.run, "uldfl_corrections_summary.csv"), index=False)
    lines = [f"{r.group} & {r.n:,} & {r.delta_mean:+.2f} ({r.delta_sd:.2f}) & {r.level_mean:.3f} & {r.S_change:+.2f} \\\\"
             for r in summ.itertuples()]
    tex = ("\\begin{tabular}{lrrrr}\n\\toprule\nDelivery decisions & $n$ & Correction $\\delta_\\theta$ (SD) & "
           "Implied service level & Change of $S$ (days) \\\\\n\\midrule\n" + "\n".join(lines)
           + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(a.out, "tab_uldfl_corrections.tex"), "w").write(tex)
    print(f"store-days {len(df)}, delivery decisions {len(v)} ({len(v) / len(df):.1%})")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
