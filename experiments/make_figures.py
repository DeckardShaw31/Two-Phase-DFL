"""Tables (LaTeX) and figures for the revised manuscript from the results folders.

python -m experiments.make_figures --results results --out revision_1/figures
"""

from __future__ import annotations

import argparse
import ast
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["BSP", "BSP-EW", "UL", "PTO-Mean", "PTO-Q", "Robust", "ScenPO-SS", "DFL", "PI"]
LABEL = {"BSP": "BSP", "BSP-EW": "BSP-EW", "UL": r"UL$_\lambda$", "PTO-Mean": "LA-Mean", "PTO-Q": "LA-Quantile",
         "Robust": "LA-Robust", "ScenPO-SS": "ScenPO-SS", "DFL": "LA-DFL", "PI": "PI (reference)"}
TEX = {"BSP": "BSP", "BSP-EW": "BSP-EW", "UL": r"UL$_\lambda$", "PTO-Mean": "LA-Mean", "PTO-Q": "LA-Quantile",
       "Robust": "LA-Robust", "ScenPO-SS": "ScenPO-SS", "DFL": "LA-DFL", "PI": "PI"}
CONFIG_LABEL = {"base": "Base", "m2": r"$m=2$", "m4": r"$m=4$", "pu10": r"$p_u=10$", "pu40": r"$p_u=40$",
                "fleet_tight": r"$\rho=1.10$", "fleet_loose": r"$\rho=1.60$", "highvar": "High var.",
                "route3": r"$c^{\mathrm{r}}=3$",
                "lifo": "30\\% LIFO"}
GREY = "#6b6b6b"
COLORS = {"route": "#4c72b0", "hold": "#dd8452", "waste": "#55a868", "lost": "#c44e52"}


def load(results: str, name: str):
    d = os.path.join(results, name)
    if not os.path.exists(os.path.join(d, "test_episodes.csv")):
        return None
    ep = pd.read_csv(os.path.join(d, "test_episodes.csv"))
    tests = pd.read_csv(os.path.join(d, "paired_tests.csv"))
    summ = pd.read_csv(os.path.join(d, "test_summary.csv"), index_col=0)
    man = json.load(open(os.path.join(d, "manifest.json")))
    return {"ep": ep, "tests": tests, "summ": summ, "man": man, "dir": d}


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def table_base(R: dict, out: str) -> None:
    s = R["summ"]
    rows = []
    for k in ORDER:
        if k not in s.index:
            continue
        r = s.loc[k]
        rows.append(f"{TEX[k]} & {r.cost:,.0f} ({r.cost_sd:,.0f}) & {r.route:,.0f} & {r.hold:,.0f} & {r.waste:,.0f} & "
                    f"{r.lost:,.0f} & {100 * r.fill:.1f} & {100 * r.waste_rate:.2f} & {r.visits / 21:.1f} \\\\")
    body = "\n".join(rows)
    tex = (r"""\begin{tabular}{lrrrrrrrr}
\toprule
Policy & Total cost (SD) & Routing & Holding & Waste & Lost sales & Fill rate (\%) & Waste rate (\%) & Visits/day \\
\midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tab_base.tex"), "w").write(tex)


def table_tests(R: dict, out: str) -> None:
    t = R["tests"].copy()
    rows = []
    for _, r in t.iterrows():
        other = r.comparison.replace("DFL - ", "")
        holm_p = fmt_p(r.p_holm_t) if r.primary else "--"
        rows.append(f"LA-DFL $-$ {TEX.get(other, other)} & {r.mean_diff:,.1f} & [{r.ci_lo:,.1f}, {r.ci_hi:,.1f}] & "
                    f"{100 * r.mean_rel_diff:.1f} & {100 * r.win_rate:.0f} & {fmt_p(r.p_t)} & {fmt_p(r.p_wilcoxon)} & "
                    f"{holm_p} \\\\")
    tex = (r"""\begin{tabular}{lrrrrrrr}
\toprule
Comparison & Mean diff. & 95\% bootstrap CI & Rel. (\%) & DFL wins (\%) & $p$ ($t$) & $p$ (Wilcoxon) & Holm $p$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tab_tests.tex"), "w").write(tex)


def fig_components(R: dict, out: str) -> None:
    s = R["summ"].reindex([k for k in ORDER if k in R["summ"].index])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(s))
    bottom = np.zeros(len(s))
    for comp, lab in (("route", "Routing"), ("hold", "Holding"), ("waste", "Waste"), ("lost", "Lost sales")):
        ax.bar(x, s[comp].values, bottom=bottom, color=COLORS[comp], label=lab, width=0.7)
        bottom += s[comp].values
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in s.index], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cost per 21-day episode")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_components.pdf"))
    plt.close(fig)


def table_sensitivity(results: str, out: str, configs: list[str]) -> pd.DataFrame:
    recs = []
    for c in configs:
        R = load(results, c)
        if R is None:
            continue
        s = R["summ"]
        ref = s.loc["PTO-Q", "cost"]
        best_nonpi = s.drop(index="PI").cost.idxmin()
        t = R["tests"].set_index("comparison")
        for k in ORDER:
            if k in s.index:
                recs.append({"config": c, "policy": k, "cost": s.loc[k, "cost"], "rel_to_q": s.loc[k, "cost"] / ref - 1,
                             "fill": s.loc[k, "fill"], "waste_rate": s.loc[k, "waste_rate"], "best": k == best_nonpi})
        for comp in ("PTO-Q", "ScenPO-SS", "BSP-EW", "UL", "Robust"):
            key = f"DFL - {comp}"
            if key in t.index:
                recs.append({"config": c, "policy": f"diff:{comp}", "cost": t.loc[key, "mean_diff"],
                             "rel_to_q": t.loc[key, "mean_rel_diff"], "fill": t.loc[key, "ci_lo"],
                             "waste_rate": t.loc[key, "ci_hi"], "best": False})
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, "sensitivity_long.csv"), index=False)
    if df.empty:
        return df
    cols = [c for c in configs if c in set(df.config)]
    lines = []
    for k in ORDER:
        if k == "PI":
            continue
        cells = []
        for c in cols:
            r = df[(df.config == c) & (df.policy == k)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            val = f"{r.cost:,.0f}"
            cells.append(r"\textbf{" + val + "}" if r.best else val)
        lines.append(f"{TEX[k]} & " + " & ".join(cells) + r" \\")
    pi = []
    for c in cols:
        r = df[(df.config == c) & (df.policy == "PI")]
        pi.append(f"{r.iloc[0].cost:,.0f}" if not r.empty else "--")
    lines.append(r"\midrule" + "\nPI (reference) & " + " & ".join(pi) + r" \\")
    fill = []
    for k in ("DFL", "ScenPO-SS", "PTO-Q", "BSP-EW"):
        cells = []
        for c in cols:
            r = df[(df.config == c) & (df.policy == k)]
            cells.append(f"{100 * r.iloc[0].fill:.1f}" if not r.empty else "--")
        fill.append(f"Fill rate (\\%), {TEX[k]} & " + " & ".join(cells) + r" \\")
    head = "Policy & " + " & ".join(CONFIG_LABEL[c] for c in cols) + r" \\"
    tex = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines)
           + "\n\\midrule\n" + "\n".join(fill) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tab_sensitivity.tex"), "w").write(tex)
    # DFL paired differences across configurations
    dl = []
    for comp in ("PTO-Q", "ScenPO-SS", "Robust", "BSP-EW", "UL"):
        cells = []
        for c in cols:
            r = df[(df.config == c) & (df.policy == f"diff:{comp}")]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            sig = r.waste_rate < 0 or r.fill > 0
            txt = f"{100 * r.rel_to_q:+.1f}"
            cells.append(r"\textbf{" + txt + "}" if sig else txt)
        dl.append(f"vs {TEX[comp]} & " + " & ".join(cells) + r" \\")
    tex2 = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head.replace("Policy", "LA-DFL")
            + "\n\\midrule\n" + "\n".join(dl) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tab_dfl_diffs.tex"), "w").write(tex2)
    return df


def fig_sensitivity(df: pd.DataFrame, out: str, configs: list[str]) -> None:
    if df.empty:
        return
    cols = [c for c in configs if c in set(df.config)]
    pols = ["BSP-EW", "UL", "PTO-Mean", "Robust", "ScenPO-SS", "DFL"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(cols))
    w = 0.13
    pal = ["#8c8c8c", "#b5b5b5", "#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for j, p in enumerate(pols):
        vals = [df[(df.config == c) & (df.policy == p)].rel_to_q.values for c in cols]
        vals = [100 * v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (j - 2.5) * w, vals, width=w, color=pal[j], label=LABEL[p])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c].replace("$", "").replace("\\", "") for c in cols], fontsize=8)
    ax.set_ylabel("Cost relative to LA-Quantile (%)")
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_sensitivity.pdf"))
    plt.close(fig)


def fig_store_service(R: dict, out: str) -> None:
    ep = R["ep"]
    from pirp.network import build_network
    coords, dist = build_network(R["man"]["config"]["layout"], R["man"]["config"]["n_stores"])
    d0 = dist[0, 1:]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for p, mk, col in (("BSP-EW", "s", "#8c8c8c"), ("UL", "^", "#b5b5b5"), ("PTO-Q", "o", "#4c72b0"),
                       ("ScenPO-SS", "D", "#55a868"), ("DFL", "*", "#c44e52")):
        sub = ep[ep.policy == p]
        if sub.empty:
            continue
        fills = np.array([ast.literal_eval(v) for v in sub.store_fill]).mean(0)
        order = np.argsort(d0)
        ax.plot(d0[order], 100 * fills[order], marker=mk, color=col, lw=0.8, ms=4, label=LABEL[p])
    ax.set_xlabel("Distance from depot")
    ax.set_ylabel("Store fill rate (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_store_service.pdf"))
    plt.close(fig)


def fig_dfl_training(R: dict, out: str) -> None:
    p = os.path.join(R["dir"], "dfl_training.csv")
    if not os.path.exists(p):
        return
    h = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(h.epoch, h.val_cost, marker="o", color="#c44e52")
    ax.set_xlabel("DFL epoch (0 = tuned LA-Quantile)")
    ax.set_ylabel("Validation cost per episode")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dfl_training.pdf"))
    plt.close(fig)


MACRO = {"BSP": "BSP", "BSP-EW": "BSPEW", "UL": "UL", "PTO-Mean": "LAMean", "PTO-Q": "LAQ", "Robust": "LARob",
         "ScenPO-SS": "ScenPO", "DFL": "DFL", "PI": "PI"}
CFG_MACRO = {"base": "Base", "m2": "MTwo", "m4": "MFour", "pu10": "PuTen", "pu40": "PuForty",
             "fleet_tight": "FleetTight", "fleet_loose": "FleetLoose", "highvar": "HighVar", "route3": "RouteThree",
             "lifo": "Lifo"}


def write_numbers(results: str, out: str, configs: list[str]) -> None:
    """LaTeX macros for every number quoted in the text (traceable to result files)."""
    L = []

    def m(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for c in configs:
        R = load(results, c)
        if R is None:
            continue
        cm = CFG_MACRO[c]
        s, t = R["summ"], R["tests"].set_index("comparison")
        for k, mk in MACRO.items():
            if k not in s.index:
                continue
            r = s.loc[k]
            m(f"cost{mk}{cm}", f"{r.cost:,.0f}")
            m(f"fill{mk}{cm}", f"{100 * r.fill:.1f}")
            m(f"waste{mk}{cm}", f"{100 * r.waste_rate:.2f}")
            m(f"visits{mk}{cm}", f"{r.visits / 21:.1f}")
            m(f"route{mk}{cm}", f"{r.route:,.0f}")
            m(f"lost{mk}{cm}", f"{r.lost:,.0f}")
            m(f"hold{mk}{cm}", f"{r.hold:,.0f}")
            m(f"wastecost{mk}{cm}", f"{r.waste:,.0f}")
            m(f"secs{mk}{cm}", f"{r.solver_secs / 21:.2f}")
        for k, mk in MACRO.items():
            key = f"DFL - {k}"
            if key not in t.index:
                continue
            r = t.loc[key]
            m(f"diff{mk}{cm}", f"{r.mean_diff:,.0f}")
            m(f"rel{mk}{cm}", f"{100 * r.mean_rel_diff:.1f}")
            m(f"ci{mk}{cm}", f"[{r.ci_lo:,.0f}, {r.ci_hi:,.0f}]")
            m(f"win{mk}{cm}", f"{100 * r.win_rate:.0f}")
            m(f"pt{mk}{cm}", fmt_p(r.p_t))
            if bool(r.primary) and "p_holm_t" in r and np.isfinite(r.p_holm_t):
                m(f"pholm{mk}{cm}", fmt_p(r.p_holm_t))
        if "PI" in s.index:
            best_rule = s.loc[["BSP", "BSP-EW", "UL"], "cost"].min()
            for k, mk in MACRO.items():
                if k in s.index and k != "PI":
                    closed = (best_rule - s.loc[k, "cost"]) / (best_rule - s.loc["PI", "cost"])
                    m(f"closed{mk}{cm}", f"{100 * closed:.0f}")
        man = R["man"]
        m(f"Q{cm}", f"{man['vehicle_cap']:.0f}")
        m(f"meandemand{cm}", f"{sum(man['mean_store_demand']):.0f}")
        ch = man["chosen"]
        if "PTO-Q" in ch:
            m(f"alphaQ{cm}", f"{ch['PTO-Q']['param']}")
        if "Robust" in ch:
            m(f"gammaRob{cm}", f"{ch['Robust']['param']}")
        ep = R["ep"]
        look = ep[ep.policy.isin(["PTO-Mean", "PTO-Q", "Robust", "ScenPO-SS", "DFL", "PI"])]
        m(f"maxgap{cm}", f"{100 * look.max_gap.max():.2f}")
        m(f"nonopt{cm}", f"{int(look.n_nonoptimal.sum())}")
        m(f"fallbacks{cm}", f"{int(look.n_fallback.sum())}")
        m(f"ndecisions{cm}", f"{len(look) * 21:,}")
    open(os.path.join(out, "numbers_auto.tex"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="revision_1/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    configs = ["base", "m2", "m4", "pu10", "pu40", "fleet_tight", "fleet_loose", "highvar", "route3", "lifo"]
    R = load(a.results, "base")
    if R is not None:
        table_base(R, a.out)
        table_tests(R, a.out)
        fig_components(R, a.out)
        fig_store_service(R, a.out)
        fig_dfl_training(R, a.out)
    df = table_sensitivity(a.results, a.out, configs)
    fig_sensitivity(df, a.out, configs)
    write_numbers(a.results, a.out, configs)
    print("written to", a.out)


if __name__ == "__main__":
    main()
