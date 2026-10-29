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

ORDER = ["BSP", "BSP-EW", "UL", "UL-DFL", "NN-DFL", "PTO-Mean", "PTO-Q", "PTO-Q2", "Robust", "ScenPO-SS", "DFL", "PI"]
LABEL = {"BSP": "BSP", "BSP-EW": "BSP-EW", "UL": r"UL$_\lambda$", "UL-DFL": "UL-DFL", "NN-DFL": "NN-DFL",
         "PTO-Mean": "LA-Mean",
         "PTO-Q": "LA-Quantile", "PTO-Q2": "LA-Quantile2", "Robust": "LA-Robust", "ScenPO-SS": "ScenPO-SS",
         "DFL": "LA-DFL", "PI": "PI (reference)"}
TEX = {"BSP": "BSP", "BSP-EW": "BSP-EW", "UL": r"UL$_\lambda$", "UL-DFL": "UL-DFL", "NN-DFL": "NN-DFL",
       "PTO-Mean": "LA-Mean",
       "PTO-Q": "LA-Quantile", "PTO-Q2": "LA-Quantile2", "Robust": "LA-Robust", "ScenPO-SS": "ScenPO-SS",
       "DFL": "LA-DFL", "PI": "PI"}
CONFIG_LABEL = {"base": "Base", "net_c1": "Clustered", "net_rc1": "Mixed", "m2": r"$m=2$", "m4": r"$m=4$",
                "pu10": r"$p_u=10$", "pu40": r"$p_u=40$", "fleet_tight": r"$\rho=1.10$",
                "fleet_loose": r"$\rho=1.60$", "highvar": "High var.", "route3": r"$c^{\mathrm{r}}=3$",
                "lifo": "30\\% LIFO", "fresh": "Real data", "base_fresh": "Base (fresh test)"}
CONFIGS = ["base", "net_c1", "net_rc1", "m2", "m4", "pu10", "pu40", "fleet_tight", "fleet_loose", "highvar", "route3",
           "lifo", "fresh"]
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
    if name == "base_fresh":  # fresh test episodes of the base case: network data from the base run
        base = json.load(open(os.path.join(results, "base", "manifest.json")))
        man = {**base, "chosen": man["chosen"], "episode_seeds": man["episode_seeds"]}
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


def table_tests(R: dict, out: str, prop: str = "DFL", fname: str = "tab_tests.tex") -> None:
    t = R["tests"].copy()
    t = t[t.comparison.str.startswith(f"{prop} - ")]
    rows = []
    for _, r in t.iterrows():
        other = r.comparison.replace(f"{prop} - ", "")
        if other not in ORDER:  # first versions of UL, UL-DFL and ScenPO-SS: see table_variants
            continue
        holm_p = fmt_p(r.p_holm_t) if r.primary else "--"
        rows.append(f"{TEX[prop]} $-$ {TEX.get(other, other)} & {r.mean_diff:,.1f} & [{r.ci_lo:,.1f}, {r.ci_hi:,.1f}] & "
                    f"{100 * r.mean_rel_diff:.1f} & {100 * r.win_rate:.0f} & {fmt_p(r.p_t)} & {fmt_p(r.p_wilcoxon)} & "
                    f"{holm_p} \\\\")
    tex = (r"""\begin{tabular}{lrrrrrrr}
\toprule
Comparison & Mean diff. & 95\% bootstrap CI & Rel. (\%) & Wins (\%) & $p$ ($t$) & $p$ (Wilcoxon) & Holm $p$ \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, fname), "w").write(tex)


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
        best_nonpi = s.loc[[k for k in ORDER if k in s.index and k != "PI"]].cost.idxmin()
        t = R["tests"].set_index("comparison")
        for k in ORDER:
            if k in s.index:
                recs.append({"config": c, "policy": k, "cost": s.loc[k, "cost"], "rel_to_q": s.loc[k, "cost"] / ref - 1,
                             "fill": s.loc[k, "fill"], "waste_rate": s.loc[k, "waste_rate"], "best": k == best_nonpi})
        for prop, comp in [("DFL", x) for x in ("PTO-Q", "PTO-Q2", "ScenPO-SS", "BSP-EW", "UL", "Robust")] + \
                [("UL-DFL", x) for x in ("UL", "BSP-EW", "ScenPO-SS", "DFL", "PTO-Q2", "NN-DFL")]:
            key = f"{prop} - {comp}"
            if key in t.index:
                recs.append({"config": c, "policy": f"diff:{key}", "cost": t.loc[key, "mean_diff"],
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
    for k in ("UL-DFL", "DFL", "ScenPO-SS", "PTO-Q", "BSP-EW"):
        cells = []
        for c in cols:
            r = df[(df.config == c) & (df.policy == k)]
            cells.append(f"{100 * r.iloc[0].fill:.1f}" if not r.empty else "--")
        fill.append(f"Fill rate (\\%), {TEX[k]} & " + " & ".join(cells) + r" \\")
    head = "Policy & " + " & ".join(CONFIG_LABEL[c] for c in cols) + r" \\"
    tex = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines)
           + "\n\\midrule\n" + "\n".join(fill) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tab_sensitivity.tex"), "w").write(tex)
    # paired differences of the two decision-focused policies across configurations
    for prop, comps, fname in (("DFL", ("PTO-Q", "PTO-Q2", "ScenPO-SS", "Robust", "BSP-EW", "UL"), "tab_dfl_diffs.tex"),
                               ("UL-DFL", ("UL", "BSP-EW", "ScenPO-SS", "DFL", "PTO-Q2", "NN-DFL"), "tab_uldfl_diffs.tex")):
        dl = []
        for comp in comps:
            cells = []
            for c in cols:
                r = df[(df.config == c) & (df.policy == f"diff:{prop} - {comp}")]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                sig = r.waste_rate < 0 or r.fill > 0
                txt = f"{100 * r.rel_to_q:+.1f}"
                cells.append(r"\textbf{" + txt + "}" if sig else txt)
            dl.append(f"vs {TEX[comp]} & " + " & ".join(cells) + r" \\")
        tex2 = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head.replace("Policy", TEX[prop])
                + "\n\\midrule\n" + "\n".join(dl) + "\n\\bottomrule\n\\end{tabular}\n")
        open(os.path.join(out, fname), "w").write(tex2)
    return df


def fig_sensitivity(df: pd.DataFrame, out: str, configs: list[str]) -> None:
    if df.empty:
        return
    cols = [c for c in configs if c in set(df.config)]
    # LA-Mean (+24% to +119%) is omitted: it would compress all other bars
    pols = ["BSP-EW", "UL", "UL-DFL", "Robust", "ScenPO-SS", "DFL"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(cols))
    w = 0.13
    pal = ["#8c8c8c", "#b5b5b5", "#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for j, p in enumerate(pols):
        vals = [df[(df.config == c) & (df.policy == p)].rel_to_q.values for c in cols]
        vals = [100 * v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + (j - 2.5) * w, vals, width=w, color=pal[j], label=LABEL[p])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c] for c in cols], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Cost relative to LA-Quantile (%)")
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_sensitivity.pdf"))
    plt.close(fig)


def store_detour_per_unit(man: dict) -> np.ndarray:
    """Extra travel of inserting each store into a tour through all stores, per unit of its mean demand."""
    from pirp.network import build_network, held_karp_all_subsets
    n = man["config"]["n_stores"]
    _, dist = build_network(man["config"]["layout"], n)
    tour = held_karp_all_subsets(dist)
    full = (1 << n) - 1
    detour = np.array([tour[full] - tour[full & ~(1 << i)] for i in range(n)])
    return detour / np.asarray(man["mean_store_demand"])


def fig_store_service(R: dict, out: str) -> None:
    ep = R["ep"]
    d0 = store_detour_per_unit(R["man"])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for p, mk, col in (("BSP-EW", "s", "#8c8c8c"), ("UL", "^", "#b5b5b5"), ("UL-DFL", "v", "#4c72b0"),
                       ("PTO-Q", "o", "#dd8452"), ("ScenPO-SS", "D", "#55a868"), ("DFL", "*", "#c44e52")):
        sub = ep[ep.policy == p]
        if sub.empty:
            continue
        fills = np.array([ast.literal_eval(v) for v in sub.store_fill]).mean(0)
        order = np.argsort(d0)
        ax.plot(d0[order], 100 * fills[order], marker=mk, color=col, lw=0.8, ms=4, label=LABEL[p])
    ax.set_xlabel("Detour per unit of demand")
    ax.set_ylabel("Store fill rate (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_store_service.pdf"))
    plt.close(fig)


def fig_dfl_training(R: dict, out: str) -> None:
    """Validation cost by epoch and training seed: LA-DFL (left) and UL-DFL (right)."""
    ul_hist = "uldfl_fine_training.csv"  # UL-DFL trained from the rule tuned on the fine grid
    if not os.path.exists(os.path.join(R["dir"], ul_hist)):
        ul_hist = "ul_dfl_training.csv"
    panels = [(f, lab) for f, lab in (("dfl_training.csv", "LA-DFL epoch (0 = tuned LA-Quantile)"),
                                      (ul_hist, r"UL-DFL epoch (0 = tuned UL$_\lambda$)"))
              if os.path.exists(os.path.join(R["dir"], f))]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.8), squeeze=False)
    for ax, (f, lab) in zip(axes[0], panels):
        h = pd.read_csv(os.path.join(R["dir"], f))
        if "seed" not in h:
            h["seed"] = 0
        for j, (sd, g) in enumerate(h.groupby("seed")):
            ax.plot(g.epoch, g.val_cost, marker="o", ms=3, lw=1, color=["#c44e52", "#4c72b0", "#55a868"][j % 3],
                    label=f"seed {sd}")
        ax.set_xlabel(lab)
        ax.set_ylabel("Validation cost per episode")
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dfl_training.pdf"))
    plt.close(fig)


def table_seeds(results: str, out: str, configs: list[str]) -> None:
    """Test cost of each training seed relative to the policy the method starts from."""
    rows = []
    for c in configs:
        R = load(results, c)
        if R is None:
            continue
        piv = R["ep"].pivot(index="episode", columns="policy", values="cost_total")
        for prop, ref in (("UL-DFL", "UL"), ("NN-DFL", "UL"), ("DFL", "PTO-Q")):
            seeds = sorted(k for k in piv.columns if k.startswith(prop + "@s"))
            if not seeds or prop not in piv:
                continue
            rel = [100 * (piv[k].mean() / piv[ref].mean() - 1) for k in seeds]
            sel = 100 * (piv[prop].mean() / piv[ref].mean() - 1)
            cells = [f"{v:+.2f}" for v in rel] + ["--"] * (3 - len(rel))
            rows.append(f"{CONFIG_LABEL[c]} & {TEX[prop]} vs {TEX[ref]} & {sel:+.2f} & " + " & ".join(cells) + r" \\")
    if rows:
        tex = ("\\begin{tabular}{llrrrr}\n\\toprule\n"
               "Setting & Comparison & Selected (\\%) & Seed 0 & Seed 1 & Seed 2 \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        open(os.path.join(out, "tab_seeds.tex"), "w").write(tex)


MACRO = {"BSP": "BSP", "BSP-EW": "BSPEW", "UL": "UL", "UL-DFL": "ULDFL", "NN-DFL": "NNDFL", "PTO-Mean": "LAMean", "PTO-Q": "LAQ",
         "PTO-Q2": "LAQTwo", "Robust": "LARob", "ScenPO-SS": "ScenPO", "DFL": "DFL", "PI": "PI",
         "UL-coarse": "ULcoarse", "UL-DFL-coarse": "ULDFLcoarse", "ScenPO-SS-bands": "ScenPObands"}
CFG_MACRO = {"base": "Base", "net_c1": "NetCOne", "net_rc1": "NetRCOne", "fresh": "Fresh", "m2": "MTwo", "m4": "MFour", "pu10": "PuTen", "pu40": "PuForty",
             "fleet_tight": "FleetTight", "fleet_loose": "FleetLoose", "highvar": "HighVar", "route3": "RouteThree",
             "lifo": "Lifo", "base_fresh": "BaseFresh"}


def table_variants(results: str, out: str, configs: list[str]) -> None:
    """First versions against final versions of the rule, its learner and ScenPO-SS (appendix).

    Paired cost differences in % of the second policy; bold: 95% paired bootstrap CI excludes zero.
    """
    from pirp.evaluation import paired_test
    pairs = [("UL", "UL-coarse", r"UL$_\lambda$, fine vs.\ coarse grid"),
             ("UL-DFL-coarse", "UL-coarse", r"UL-DFL vs.\ UL$_\lambda$, both coarse (first version)"),
             ("UL-DFL-coarse", "UL", r"UL-DFL from coarse rule vs.\ UL$_\lambda$ fine"),
             ("UL-DFL", "UL", r"UL-DFL vs.\ UL$_\lambda$, both fine (reported)"),
             ("ScenPO-SS", "ScenPO-SS-bands", r"ScenPO-SS, Jia et al.\ vs.\ band sampling")]
    cols, cells = [], {p[2]: [] for p in pairs}
    fills = {"ScenPO-SS": [], "ScenPO-SS-bands": []}
    for c in configs:
        R = load(results, c)
        if R is None or "UL-coarse" not in set(R["ep"].policy):
            continue
        cols.append(c)
        piv = R["ep"].pivot(index="episode", columns="policy", values="cost_total")
        for a, b, lab in pairs:
            r = paired_test(piv[a].values, piv[b].values)
            txt = f"{100 * r['mean_rel_diff']:+.1f}"
            cells[lab].append(r"\textbf{" + txt + "}" if (r["ci_hi"] < 0 or r["ci_lo"] > 0) else txt)
        for k in fills:
            fills[k].append(f"{100 * R['summ'].loc[k, 'fill']:.1f}")
    if not cols:
        return
    lines = [f"{lab} & " + " & ".join(v) + r" \\" for lab, v in cells.items()]
    lines.append(r"\midrule")
    lines += [r"Fill rate (\%), ScenPO-SS, Jia et al. & " + " & ".join(fills["ScenPO-SS"]) + r" \\",
              r"Fill rate (\%), ScenPO-SS, band sampling & " + " & ".join(fills["ScenPO-SS-bands"]) + r" \\"]
    head = "Comparison (\\%) & " + " & ".join(CONFIG_LABEL[c] for c in cols) + r" \\"
    tex = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines)
           + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tab_variants.tex"), "w").write(tex)


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
        for prop, pre in (("DFL", ""), ("UL-DFL", "UL")):
            for k, mk in MACRO.items():
                key = f"{prop} - {k}"
                if key not in t.index:
                    continue
                r = t.loc[key]
                m(f"{pre}diff{mk}{cm}", f"{r.mean_diff:,.0f}")
                m(f"{pre}rel{mk}{cm}", f"{100 * r.mean_rel_diff:.1f}")
                m(f"{pre}absrel{mk}{cm}", f"{100 * abs(r.mean_rel_diff):.1f}")  # for "x% less/more" in text
                m(f"{pre}ci{mk}{cm}", f"[{r.ci_lo:,.0f}, {r.ci_hi:,.0f}]")
                m(f"{pre}win{mk}{cm}", f"{100 * r.win_rate:.0f}")
                m(f"{pre}pt{mk}{cm}", fmt_p(r.p_t))
                if bool(r.primary) and "p_holm_t" in r and np.isfinite(r.p_holm_t):
                    m(f"{pre}pholm{mk}{cm}", fmt_p(r.p_holm_t))
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
        if "UL" in ch:
            m(f"alphaUL{cm}", f"{ch['UL']['alpha']}")
            m(f"betaUL{cm}", f"{ch['UL']['beta']}")
        ep = R["ep"]
        look = ep[ep.policy.isin(["PTO-Mean", "PTO-Q", "PTO-Q2", "Robust", "ScenPO-SS", "DFL", "PI"])]
        m(f"maxgap{cm}", f"{100 * look.max_gap.max():.2f}")
        m(f"nonopt{cm}", f"{int(look.n_nonoptimal.sum())}")
        m(f"fallbacks{cm}", f"{int(look.n_fallback.sum())}")
        m(f"ndecisions{cm}", f"{len(look) * 21:,}")
    sens_path = os.path.join(results, "solver_sensitivity.csv")
    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path).set_index("config")
        b = sens.loc["base"]
        m("noiseSDBase", f"{b.sd_rel:.1f}")
        m("noiseMinBase", f"{b.min_rel:+.1f}")
        m("noiseMaxBase", f"{b.max_rel:+.1f}")
        m("noiseMeanBase", f"{b.mean_rel:+.2f}")
        m("noiseSDLo", f"{sens.sd_rel.min():.1f}")
        m("noiseSDHi", f"{sens.sd_rel.max():.1f}")
        m("noiseAbsMeanMax", f"{sens.mean_rel.abs().max():.2f}")
        m("noiseSEHi", f"{sens.se_mean_rel.max():.2f}")
        m("noiseChanged", f"{int(sens.changed.sum())}")
        m("noiseN", f"{int(sens.n.sum()):,}")
    else:  # check not run yet: visible placeholders keep the manuscript compilable
        for k in ("SDBase", "MinBase", "MaxBase", "MeanBase", "SDLo", "SDHi", "AbsMeanMax", "SEHi", "Changed", "N"):
            m(f"noise{k}", r"\textbf{??}")
    open(os.path.join(out, "numbers_auto.tex"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="revision_1/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    configs = CONFIGS
    R = load(a.results, "base")
    F = load(a.results, "base_fresh")  # primary hypotheses: base case on fresh test episodes
    P = F if F is not None else R
    if R is not None:
        table_base(P, a.out)
        table_tests(P, a.out)
        table_tests(P, a.out, "UL-DFL", "tab_tests_uldfl.tex")
        fig_components(P, a.out)
        fig_store_service(P, a.out)
        fig_dfl_training(R, a.out)
        front = os.path.join(R["dir"], "rule_frontier.csv")
        if os.path.exists(front):
            from experiments.rule_frontier import fig_frontier
            fig_frontier(pd.read_csv(front), R["ep"], a.out)
    df = table_sensitivity(a.results, a.out, configs)
    fig_sensitivity(df, a.out, configs)
    table_seeds(a.results, a.out, (["base_fresh"] if F is not None else []) + configs)
    table_variants(a.results, a.out, configs)
    write_numbers(a.results, a.out, configs + (["base_fresh"] if F is not None else []))
    print("written to", a.out)


if __name__ == "__main__":
    main()
