"""Fold the re-tuned rule, the retrained UL-DFL and the faithful ScenPO-SS into the result files.

Afterwards test_episodes.csv of every configuration contains, under the manuscript's names,
  UL           UL_lambda re-tuned on the fine grid               (ul_retune_test_episodes.csv)
  UL-DFL       UL-DFL trained from it, seeds UL-DFL@s*            (uldfl_fine_test_episodes.csv)
  ScenPO-SS    scenario set of Jia et al. (2025), 20 scenarios     (scenpo_jia_test_episodes.csv)
and keeps the first versions as UL-coarse, UL-DFL-coarse (seeds UL-DFL-coarse@s*) and
ScenPO-SS-bands; nothing is dropped. paired_tests.csv and test_summary.csv are recomputed as in
run_config.py, and chosen_policies.joblib and the manifest record the new specifications. The new
rows are always rebuilt from their source files, so folding twice gives the same result.

python -m experiments.fold_results --results results_v2
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd

from experiments.fresh_test import SCENPO
from experiments.retrain_uldfl import fine_rule
from experiments.retune_ul import ORDER as CONFIG_ORDER
from experiments.run_config import log, summarise

MAIN = ["BSP", "BSP-EW", "UL", "UL-DFL", "NN-DFL", "NB-DFL", "PTO-Mean", "PTO-Q", "PTO-Q2", "Robust", "ScenPO-SS", "DFL", "PI"]
FIRST = {"UL": "UL-coarse", "UL-DFL": "UL-DFL-coarse", "ScenPO-SS": "ScenPO-SS-bands"}


def rename_first(name: str) -> str:
    if name.startswith("UL-DFL@"):
        return "UL-DFL-coarse" + name[6:]
    return FIRST.get(name, name)


def fold(c: str, results: str) -> pd.DataFrame:
    run = os.path.join(results, c)
    test = pd.read_csv(os.path.join(run, "test_episodes.csv"))
    if "UL-coarse" not in set(test.policy):
        test["policy"] = test.policy.map(rename_first)
    new = set(FIRST) | {"UL-orig"} | {p for p in test.policy if p.startswith(("UL-DFL@", "NB-DFL"))}
    test = test[~test.policy.isin(new)]
    rule = pd.read_csv(os.path.join(run, "ul_retune_test_episodes.csv")).assign(policy="UL")
    dfl = pd.read_csv(os.path.join(run, "uldfl_fine_test_episodes.csv"))
    dfl["policy"] = dfl.policy.str.replace("UL-DFL-fine", "UL-DFL", regex=False)
    scen = pd.read_csv(os.path.join(run, "scenpo_jia_test_episodes.csv")).assign(policy="ScenPO-SS")
    parts = [test, rule, dfl, scen]
    for extra in ("nbdfl_test_episodes.csv", "ul_orig_test_episodes.csv"):  # run_nbdfl.py, ul_original.py
        if os.path.exists(os.path.join(run, extra)):
            parts.append(pd.read_csv(os.path.join(run, extra)))
    test = pd.concat(parts, ignore_index=True)
    n = test.groupby("policy").size()
    assert n.nunique() == 1, f"{c}: unequal episode counts {n.to_dict()}"
    test.to_csv(os.path.join(run, "test_episodes.csv"), index=False)
    order = [k for k in MAIN if k in set(test.policy)] + list(FIRST.values())
    order += ["UL-orig"] if "UL-orig" in set(test.policy) else []
    order += sorted(k for k in set(test.policy) if "@s" in k)
    summ, _ = summarise(test, order, run)

    rule_spec = fine_rule(results, c)
    src = os.path.join(results, "base" if c == "lifo" else c, "uldfl_fine_policies.joblib")
    fine = joblib.load(src)
    chosen_path = os.path.join(run, "chosen_policies.joblib")
    if os.path.exists(chosen_path):  # lifo evaluates the base policies and has none of its own
        chosen = joblib.load(chosen_path)
        if "UL-coarse" not in chosen:
            for k in [k for k in chosen if k in FIRST or k.startswith("UL-DFL@")]:
                chosen[rename_first(k)] = chosen.pop(k)
        chosen.update({"UL": rule_spec, "UL-DFL": fine["UL-DFL"], "ScenPO-SS": SCENPO})
        chosen.update({k: v for k, v in fine.items() if k.startswith("UL-DFL@")})
        joblib.dump(chosen, chosen_path)
    man_path = os.path.join(run, "manifest.json")
    man = json.load(open(man_path))
    ch = man["chosen"]
    if "UL-coarse" not in ch:
        for k in [k for k in ch if k in FIRST or k.startswith("UL-DFL@")]:
            ch[rename_first(k)] = ch.pop(k)
    ch.update({"UL": rule_spec, "UL-DFL": {"kind": "ul_dfl", "rule": rule_spec}, "ScenPO-SS": SCENPO})
    ch.update({k: {"kind": "ul_dfl", "rule": rule_spec} for k in fine if k.startswith("UL-DFL@")})
    man["folded"] = ("UL re-tuned on the fine grid (retune_ul.py), UL-DFL trained from it (retrain_uldfl.py), "
                     "ScenPO-SS with the scenario set of Jia et al. (rerun_scenpo.py); first versions kept as "
                     "UL-coarse, UL-DFL-coarse and ScenPO-SS-bands")
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=2)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=CONFIG_ORDER)
    ap.add_argument("--results", default="results_v2")
    a = ap.parse_args()
    for c in a.configs:
        summ = fold(c, a.results)
        s = summ["cost"]
        log(f"{c}: UL {s['UL']:.0f}  UL-DFL {s['UL-DFL']:.0f}  ScenPO-SS {s['ScenPO-SS']:.0f}  "
            f"(first versions {s['UL-coarse']:.0f} / {s['UL-DFL-coarse']:.0f} / {s['ScenPO-SS-bands']:.0f})")


if __name__ == "__main__":
    main()
