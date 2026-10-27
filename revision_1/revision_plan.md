# IJPE Rejection Response: Corrected Revision Plan

**Decision letter:** PROECO-D-26-05160  
**Target:** *International Journal of Production Economics*  
**Plan date:** 27 October 2026  
**Status:** Core comparator sources are checked, the primary synthetic benchmark is complete and audited, and a post-hoc finite-scenario robust adaptation has been implemented and evaluated on the same held-out episodes. The revised manuscript now compiles, the seven-page PDF has been visually checked, and the citation-key audit passes. The stronger source-method comparator gate and full source-by-source factual literature audit remain open. This plan is not a claim that the manuscript is ready to resubmit.

## 1. What the editor asked us to fix

The editor gave two reasons for rejection:

1. The paper did not take a clear perishable-inventory-management perspective.
2. The comparisons did not cover strong current operational methods.

The revision must make replenishment, age-specific carryover, expiry, shortage/service, store capacity, shared vehicle capacity, and their cost consequences the organizing story. DFL is one candidate method within that story. More citations alone will not answer either point.

| Rejection reason | Concrete revision response | Evidence required to call it resolved | Current status |
|---|---|---|---|
| The paper lacks a clear perishable inventory-management perspective | Organize the question, model, metrics, and interpretation around shelf life, age-specific stock, replenishment, expiry, service/shortage, carryover, store capacity, and fleet capacity. Keep DFL as one candidate method. | Results and conclusion explain physical inventory trade-offs with measured outcomes and bounded claims; the paper remains understandable without assuming DFL wins. | Draft content and rendered PDF checked; the inventory perspective is clear for this scoped synthetic pilot. |
| Comparisons fall short of state-of-the-art methods | Correct false/misattributed literature; identify direct perishable PIRP work; implement a strong contemporary operational comparator under a fair common evaluator, information schedule, and solver budget. | Source-faithful method or transparently scoped adaptation, shared demand paths, validated inventory transitions, documented solver/fallback behavior, and comparative outcomes. | A post-hoc empirical finite-scenario robust plan selector is evaluated, but all 300 candidate solves used fallback and it is not He et al.'s source method. Comparator gate remains open. |

## 2. Evidence decisions before rewriting

- `main.tex` is now the sole active manuscript source, rewritten around the inventory-management question. It contains audited results but remains a working draft until final PDF visual review and the comparator gap is addressed.
- The checked-in legacy output is not accepted as evidence. Its adjacent blocks were not independent, its features exposed information inconsistent with a fixed episode-start plan, M5 calibration was claimed without a traceable procedure, solver status was overstated, and several comparator names did not match the code.
- The generator is a hand-specified synthetic demand process. It uses illustrative store demand profiles, seasonality, announced promotions/price, hidden weather effects, and autoregressive noise. It does not ingest or calibrate to M5. Solomon-derived coordinates are used without time windows. There are no operational stock, expiry, service, or route observations.
- A fresh dataset now contains 400 independently seeded five-period episodes (280/60/60 train/validation/test), with episode-start features and no realized within-horizon demand or weather exposed to predictors. Test episodes are kept separate from exploratory smoke runs.
- The FIFO evaluator reports expiry separately from capacity disposal and records shortage, fill rate, carryover, deliveries, and routing. The primary test audit found zero maximum state discrepancy between forecast plans and the FIFO simulator under their planning demand; cost and spoilage accounting reconciled exactly.
- The optimizer now distinguishes LP solves, requested MILPs, feasible heuristic fallbacks, and proven optimality. A single diagnostic MILP with a five-second limit still returned the feasible construction fallback at a 96% reported gap; no optimality should be inferred from the production setting's shorter limit.
- Earlier “SAA,” “ALNS,” “MPC,” and “PIL” labels are withdrawn where their code did not implement those published methods. Their outputs are excluded.

## Phase 0 — Freeze the usable evidence

### Completed

- Select `main.tex` as the only manuscript source; archive the conflicting older plans and manuscript artifacts.
- Correct demand provenance and replace the dependent-episode dataset with independently seeded episodes.
- Fix the solver's default status reporting, add optional initial inventory to the optimizer, and separate expiry from capacity disposal in evaluation.
- Freeze an internal, not externally preregistered, primary protocol and keep legacy/smoke outputs separate from the final run.

### Gate 0 — Passed for the primary pilot

Every number in the current results section maps to the completed run files. The manifest records seeds, data hashes, settings, environment, solver outcomes, and fallbacks; the automated result audit passed. This gate closes for the bounded synthetic pilot only.

## Phase 1 — Reframe the research around inventory decisions

### Central question

> How do shelf life, age-structured carryover, and shared fleet capacity shape replenishment, expiry, service, and operating cost under uncertain demand, and what can learning-based methods add relative to operational policies?

### Required manuscript changes

1. Open with a concrete replenishment and service decision, then explain why shelf life and capacity couple periods and stores.
2. Make physical outcomes primary: total cost and components, expiry units, capacity-disposal units, unmet demand/fill rate, age-specific on-hand inventory and carryover, deliveries, routes, and fleet utilization.
3. Treat forecast error, training time, and any decision-regret diagnostic as secondary. Do not use a heuristic full-information plan as a lower bound or as the denominator of headline regret.
4. Present DFL as a candidate method, report where it loses, and do not claim general superiority from a single synthetic configuration.

### Gate 1

The title, abstract, research questions, contribution statements, results, and conclusion must all remain understandable as an inventory-management paper without asserting that DFL wins.

## Phase 2 — Source-audited literature and operational comparators

### Verified source findings

**Phase 2 bibliography gate:** the core source identities and method summaries are verified. **Comparator coverage gate:** still open. The existence of a paper is separate from whether the submitted code faithfully implements it.

- Crama et al. (2018), “Stochastic Inventory Routing for Perishable Products,” is real and directly studies stochastic perishable inventory routing. It includes expected-value, deliver-up-to-level (UL-lambda), decomposition, and decomposition-integration methods. UL-lambda is a dynamic rolling-horizon policy that updates after observed demand. The code in this workspace is an open-loop validation-residual adaptation using simulated inventory propagation and a route-packing heuristic; it is **not** a reproduction of the paper's policy or its stronger decomposition methods.
- He, Liu, and Liu (2024), “Robust Optimization Approaches for the Perishable Product Inventory Routing Problem with Demand Uncertainty,” is real and directly studies robust perishable PIRP using robust-counterpart reformulation, Benders decomposition, and ILS-BD.
- Jia, Schrotenboer, and Chen (2025), “Scenario Predict-then-Optimize for Data-Driven Online Inventory Routing,” *Transportation Science* 59(5), 1032–1056, DOI 10.1287/trsc.2024.0613, is real. The full paper describes multi-horizon quantile-RNN forecasts, sampled demand scenarios, and scenario-based stochastic optimization for online inventory routing. Its reviewed formulation has no finite-life age-cohort state; adding FIFO age tracking and expiry would be an adaptation. Its online decision timing also needs a fair shared-information protocol.
- Alvarez et al. (2020) provide formulations, branch-and-cut, and a hybrid heuristic for deterministic perishable PIRP. Laganà et al. (2024) provide an iterated local search matheuristic for a deterministic, non-perishable multivehicle IRP. Neither supports the old ALNS label.
- Jia's earlier arXiv identifier is 2401.17787. The previously listed 2407.03964 is unrelated and must not be cited for this method.

### Comparator plan

1. Keep LightGBM point PTO and asymmetric-quantile PTO as transparent forecast-based controls; they are not state-of-the-art operational policies.
2. Keep the Crama UL-lambda code only under the explicit “adaptation” label and disclose its open-loop information structure and route-packing heuristic.
3. Before making a state-of-the-art comparison claim, implement a serious direct operational comparator. **Priority: He et al. (2024)**, because it is recent and directly models demand-uncertain perishable PIRP; reproduce its robust counterpart/decomposition approach on the common instances, or label and validate any scoped adaptation. Use the paper's uncertainty-set definition after checking the full source. Record where its deterioration/expiry model differs from the study's FIFO fixed-life simulator. Do not silently change the uncertainty set or call a simplified heuristic the source algorithm.
4. Add Jia et al.'s (2025) scenario PTO as a second, adjacent comparator if resources allow. It is a strong data-driven online IRP method, but a FIFO/expiry extension and a matched online decision schedule are required. If using it, let the other policies replan on the same observations, or report it in a separately labeled online protocol rather than ranking it against episode-start fixed plans.
5. Upgrade the Crama UL-lambda policy to its dynamic observe-and-update form where feasible; this improves the direct operational comparison but does not replace the need to address the recent robust comparator. Keep the current open-loop adaptation labeled as such.
6. For every comparator, publish its information set, re-optimization schedule, inventory state, uncertainty representation, solver limit, and fallback behavior. If a source method cannot be implemented faithfully under the simulator, report the mismatch and do not use it to claim state-of-the-art coverage.

**Work completed after the revision2 pilot:** an empirical finite-scenario robust plan selector now uses validation residual paths, generates five fixed-plan candidates, and selects the candidate with the lowest worst-case FIFO cost over the validation scenario bank. It uses episode-start information and the common evaluator, but it is a custom restricted-candidate heuristic, not He et al.'s RCR/BD/ILS-BD. Its post-hoc 60-episode run is stored in `results/revision2_robust_adaptation_exploratory/`; 300/300 candidate MILP requests selected the construction fallback. Its paired results are descriptive only and do not close Gate 2.

### Gate 2

Pass only after source-faithful or explicitly scoped comparator implementations have been tested under a common physical evaluator, shared demand paths, fair information access, and a stated solver budget. The current UL-lambda adaptation and PTO controls alone do not pass this gate. The source audit corrected the bibliography and attributions; it does not by itself satisfy the editor's comparison request.

## Phase 3 — Run the corrected primary experiment

### Current internal protocol

- One primary setting: C1 network, 10 stores, five periods, shelf life 3, holding cost 1, spoilage/disposal cost 5, shortage cost 20, two vehicles of capacity 100, store capacity 70, and zero starting inventory.
- 400 independent synthetic episodes, split 280/60/60. The episode is the independent unit. All policies share the same held-out paths and FIFO evaluator. Forecasts use only features known at episode start; no test path is used for tuning.
- Primary method set: LightGBM point PTO, LightGBM quantile PTO, the Crama UL-lambda adaptation, and two-phase DFL. An additional post-hoc empirical robust adaptation is reported separately and excluded from the prespecified primary test family.
- The DFL run uses three training seeds, averaged within each test episode. Its 50-epoch path uses a vectorized polyhedral approximation on most batches and requests the integer oracle only on selected batches in the final phase. HiGHS may return a feasible construction fallback. Describe this as an approximate hybrid; do not claim exact full-MILP-per-batch SPO+ training.
- Primary comparisons are paired episode total cost for DFL versus LightGBM point PTO and the UL-lambda adaptation, with a paired bootstrap and Holm correction. Quantile PTO, physical measures, and subgroup summaries are secondary unless separately prespecified.
- Store requested gap and time limit, actual solver status/gap, fallback counts, and FIFO-model parity. A requested MILP is not the same as a proven optimal MILP solution.
- The empirical robust adaptation had mean test cost 2712.3 (SD 1148.9), fill rate 0.904, and zero FIFO parity error. Its exploratory mean paired difference was -25.8 versus point PTO (95% bootstrap CI [-85.2, 32.7]) and -82.7 versus DFL ([-150.0, -14.9]); all 300 candidate MILPs used fallback. Because it was added after the pilot results were examined, these contrasts are not confirmatory.

### Sensitivity work

The former 81-cell factorial is **not** the current primary experiment and its old results are not valid evidence. After a valid base run and strong comparator implementation, choose a documented full or fractional design across topology, shelf life, cost ratio, and capacity. Decide the design before examining those outcomes. Until then, make no cross-topology, shelf-life, cost-ratio, or capacity-interaction claim.

### Gate 3 — Passed for the bounded synthetic pilot

The fresh run completed; data hashes and 280/60/60 split match; all methods cover the 60 test episodes; cost reconciliation and FIFO parity checks pass; fallbacks are reported; and all results are regenerated from `results/revision2_final/`. The audit found 300/300 forecast/DFL plan evaluations used fallback and zero evaluation MILPs were optimal. Passing this gate does not fix the weak comparator or synthetic-data scope.

## Phase 4 — Explain physical mechanisms from measured outcomes

For each tested method report total cost, cost components, expiry and capacity-disposal quantities, shortages and fill rate, inventory by age and period, deliveries, routing distance, routes, and fleet utilization. The primary 60-episode test gives mean cost 2738.1 for point PTO, 3238.9 for quantile PTO, 3058.3 for the UL-lambda adaptation, and 2795.0 for DFL. DFL minus point PTO is +56.9 (95% paired-bootstrap CI [-14.2, 129.1], Holm-adjusted p=.137); DFL minus the UL-lambda adaptation is -263.3 (95% CI [-340.0, -183.4], Holm-adjusted p=2.75e-8). Mean fill rates range from 0.885 to 0.903; average expired units range from 0 to 1.05 per episode, with zero capacity-discarded units for all methods. DFL does not show a statistically detectable cost improvement over point PTO. The separate post-hoc empirical robust adaptation gives mean cost 2712.3 (SD 1148.9), fill rate 0.904, and zero capacity discards; exploratory paired intervals are [-85.2, 32.7] versus point PTO and [-150.0, -14.9] versus DFL. All its 300 candidate MILP requests used fallback. Do not interpret the post-hoc contrasts as confirmatory evidence or extrapolate costs to a real network.

Do not report decision regret against a heuristic full-information schedule as “optimal regret.” If retained, call it a gap to a specified heuristic reference, describe that reference, and keep it secondary. Report both winning and losing test cases and the scale of differences.

## Phase 5 — Rewrite and check the manuscript

1. **Draft and results completed:** replace the mixed-version manuscript with the inventory-centered `main.tex` and insert the audited primary results. Old M5, 81-grid, 13-baseline, clairvoyant-regret, and speedup claims are withdrawn.
2. Organize the paper as: inventory question; source-audited related work; physical PIRP and information timing; synthetic episode design; operational policies and their fidelity; physical evaluator; paired results; limitations and implications.
3. Explicitly distinguish a genuine published method, a faithful reproduction, a disclosed adaptation, and a custom comparator.
4. State that synthetic outcomes are not field validation. Do not describe M5 as an input or calibration source.
5. **Completed for the current internal draft:** after adding the exploratory robust adaptation and its limits, `main.tex` compiled successfully, all seven rendered pages were visually inspected, and the citation-resolution audit passed for 60 bibliography records and 12 active citations. Before submission, finish a source-by-source audit of every factual literature claim in the full matrix and manuscript; the parser checks citation-key resolution and DOI/URL presence, not claim truth.

### Gate 4

The manuscript must make no claim broader than the tested setting, no mathematical optimality claim contradicted by solver logs, and no operational-comparison claim broader than the comparators that were actually run.

## Phase 6 — Decide resubmission readiness

Before resubmission, recheck current IJPE scope and author requirements, have an inventory/operations reader review the framing and comparator fairness, and package the exact code/data/configuration/manifest used for all reported results. A stronger state-of-the-art comparator remains a prerequisite for claiming that the editor's second concern is resolved. If a fair contemporary comparator or operational evidence cannot be supplied, narrow the claims and reassess whether IJPE is the right venue rather than implying the desk rejection has been overcome.

## Current gate ledger

| Gate | Status | Evidence needed to close |
|---|---|---|
| 0. Evidence freeze/reproduction | Passed for the synthetic pilot | None for this run; operational data remain absent |
| 1. Inventory-centered framing | Content and rendered-PDF check passed for the synthetic draft | External inventory/operations reader review |
| 2a. Source identity and core method audit | Core Crama/Jia/He sources checked; citation parser passed (60 BibTeX records, 12 active citations); remaining matrix rows need final source-by-source check | Complete source audit for every cited claim before submission |
| 2b. Credible operational comparison | Not passed; exploratory robust adaptation added, but 300/300 candidate solves fell back | A source-faithful or rigorously scoped contemporary method with fair information timing, physical parity, and acceptable optimization quality |
| 3. Corrected primary experiment | Passed for the bounded pilot; high fallback rate disclosed | None for the pilot; solver quality limits interpretation |
| 4. Claim-to-evidence manuscript | Primary and exploratory result audits pass; current source compiles; seven-page visual QA and citation-key audit pass | Full source-by-source factual citation review before submission |
| 5. IJPE resubmission readiness | Not passed | Gates 0–4, external operations review, current submission checklist |

The revised literature matrix and comparator protocol in this folder is the authoritative source for bibliographic corrections and method scope. The one-page positioning note, hold cover letter, README, result audit, and manuscript have been synchronized to this gate ledger.
