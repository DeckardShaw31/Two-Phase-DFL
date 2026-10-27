# Claim-to-Evidence Audit — Corrected Revision

**Status:** Primary run and post-hoc robust adaptation are complete and audited. This replaces the legacy audit, which incorrectly marked unsupported claims as verified. Claims below are limited to the manifests, episode-level outputs, and source records listed here.

## 1. Claims withdrawn from the previous manuscript

The following are not valid evidence for the revised submission and must not be copied into the abstract, tables, positioning note, or cover letter:

- “M5-calibrated” or empirically calibrated demand.
- The legacy 81-cell factorial and reported shelf-life/capacity numbers.
- The legacy “13 baselines” comparison and Holm significance claims.
- Regret measured against a heuristic full-information schedule as if it were an optimal lower bound.
- An asserted 5.66x training speedup or validation-regret advantage from the legacy profiling run.
- Claims that custom code faithfully implements SAA, ALNS, MPC, PIL, or published stochastic/robust methods.
- Any claim that all forecast-based plans are optimal MILP solutions.

The old artifacts remain in the workspace as legacy material; they are excluded from the new evidence.

## 2. Current claims and evidence sources

| Claim | Evidence source | Current status |
|---|---|---|
| Demand is hand-specified synthetic and uses no M5 observations | data/generate_dataset.py; data/revision2_final/processed/params_config.json; episode manifest | Verified from code/config |
| Data comprises 400 independent episodes split 280/60/60 | data/revision2_final/processed/episode_manifest.json; results/revision2_final/run_manifest.json | Verified; 60 test episodes and input hashes are recorded in the run manifest |
| Predictors exclude realized within-horizon demand/weather and use one episode-start feature set | src/experiments/run_revision_benchmark.py; generator and feature list in params_config.json | Verified from code; runtime guards enforce exclusion |
| Inventory costs separate expiry spoilage from capacity disposal and track shortages/fill rate/carryover/routing | src/utils/metrics.py; results/revision2_final/episode_metrics.csv | Verified; cost decomposition and spoilage subcomponents reconcile exactly |
| Optimizer may return a feasibility-checked construction fallback and reports solver status/gap | src/optimization/pirp_model.py; results/revision2_final/solver_diagnostics.csv | 300 of 300 PTO/DFL evaluation calls used the construction fallback; 0 evaluation MILPs were optimal |
| Forecast plans match the FIFO simulator under their planning demand | Runner parity assertion; results/revision2_final/solver_diagnostics.csv | PASS; maximum absolute state discrepancy is 0 |
| Crama et al. (2018) UL-lambda and decomposition methods exist for stochastic perishable IRP | Full primary article; src/models/operational_baselines.py | Source policy is dynamic and updates after observed demand; workspace implementation is an open-loop adaptation, not a faithful reproduction |
| Jia et al. (2025) Scenario Predict-then-Optimize is a real online IRP method | Full article text via TU/e repository and INFORMS DOI record | Quantile-RNN demand scenarios and scenario stochastic optimization verified; full formulation reviewed has no finite-life age-cohort state. It is not implemented here; adding FIFO expiry would be an adaptation |
| He et al. (2024) robust perishable PIRP is a real direct operational method | Official AIMS journal article | Robust-counterpart reformulation, Benders, and ILS-BD verified; source-faithful method not implemented here |
| A finite-scenario robust adaptation was evaluated on the same test episodes | `src/models/empirical_robust_pirp.py`; `results/revision2_robust_adaptation_exploratory/` | PASS for this custom, post-hoc restricted-candidate heuristic; mean cost 2712.3 (SD 1148.9), fill rate 0.904; all 300 candidate solves used fallback; FIFO parity max error 0. It is not a reproduction of He et al. |
| The robust adaptation improves on point PTO or DFL | `results/revision2_robust_adaptation_exploratory/exploratory_paired_comparisons.csv` | Exploratory only: -25.8 vs point PTO (95% bootstrap interval [-85.2, 32.7]); -82.7 vs DFL ([-150.0, -14.9]). Added after pilot outcomes were inspected; not confirmatory. |
| Active manuscript citations resolve and BibTeX records include DOI/permanent URLs | `results/citation_audit_revision2.json`; CLI citation audit | PASS: 60 BibTeX entries parsed, 12 manuscript citations resolved, no missing keys, no entries without DOI/URL, LaTeX environments balanced. This parser/metadata check does not independently validate every literature claim. |
| DFL is an exact full-MILP-per-batch SPO+ implementation | src/models/dfl_models.py, src/optimization/pyepo_interface.py | False. The 50-epoch path uses an approximate polyhedral loss on most batches; selected integer-oracle calls may use a feasible fallback |
| The revised method beats a strong state-of-the-art comparator | No source-faithful He/Jia comparator; robust adaptation uses fallbacks | Unsupported; do not claim |
| DFL reduces total cost versus point PTO | results/revision2_final/paired_primary_tests.csv | Not supported: DFL minus point PTO is +56.9 units (95% paired-bootstrap CI -14.2 to 129.1; Holm-adjusted p=.137) |
| DFL reduces total cost versus the current UL-lambda comparator | results/revision2_final/paired_primary_tests.csv | Supported for this specific open-loop nearest-neighbor/2-opt adaptation: difference -263.3 (95% CI -340.0 to -183.4; Holm-adjusted p=2.75e-8). This is not a result against Crama et al.'s source policy. |
| The revised benchmark establishes state-of-the-art performance | No source-faithful He/Jia comparator in this run; 300/300 PTO/DFL plan calls fell back | Unsupported; do not claim |

## 3. Solver diagnostics already observed

The primary benchmark completed on all four methods and all 60 held-out episodes. Across the 300 point-PTO, quantile-PTO, and DFL plan evaluations, every requested MILP selected the feasible construction fallback; none was proven optimal. The UL-lambda adaptation uses its separate nearest-neighbor/2-opt route-packing heuristic. DFL training made 1,534 integer-oracle requests, all of which fell back; the other 936 training solver calls were LP calls reported optimal. The three DFL seeds are averaged within each test episode and are not counted as independent cases. The separate finite-scenario robust adaptation generated five candidates per test episode; all 300 candidate MILP requests used fallback, but all candidate plans passed the FIFO parity and cost-accounting checks.

A separate diagnostic solve with a five-second limit on a representative 10-store, five-period demand instance also selected the construction fallback; the reported MILP gap was about 96%. This is not part of the held-out performance sample.

## 4. Final-run audit checklist

Primary-run checks completed and recorded in `results/revision2_final/audit_report.json`:

1. Manifest contains the expected data hash, split, 60 test IDs, three DFL seeds, solver settings, and software versions.
2. All four methods cover all 60 test episodes; DFL has three training seeds per episode.
3. Primary paired differences, bootstrap intervals, and Holm adjustment are recorded; DFL seeds are averaged within episode.
4. Solver status, fallbacks, and model/simulator parity are reported. No “optimal solution” wording is used.
5. Total cost decomposes into routing + holding + expiry + capacity-disposal + stockout cost; expiry and capacity disposal reconcile with total spoilage cost.
6. The previous manuscript source compiled successfully and all seven rendered pages were visually inspected. The source has since been updated with the exploratory robust adaptation and must be recompiled and visually reviewed again.
7. Keep the state-of-the-art comparator gate open until a source-faithful contemporary robust/scenario method is implemented and evaluated with acceptable solver quality.

## 5. Reproducibility checks

Focused validation command used:

    .\.venv\Scripts\python.exe -m pytest tests\test_optimization.py tests\test_fifo_adversarial.py tests\test_operational_baselines.py tests\test_benchmarks.py -q

The focused suite previously passed 47 tests. That run preceded the final statistical-summary guard change; no newer suite result is claimed in this audit. The primary benchmark and its output audit are complete; the test-suite result remains as previously recorded.
