# IJPE Desk Rejection (PROECO-D-26-05160): Revision Plan v2

**Plan date:** 27 October 2026
**Manuscript source:** `revision_1/manuscript.tex` (single source of truth)
**Code:** `pirp/` package + `experiments/` scripts (the former `src/`, `scripts/` and legacy result folders were removed; they remain in git history)

## 0. Editor's two reasons and how this plan answers them

| Editor's reason | What was wrong in the submission | What the revision does |
|---|---|---|
| "does not take a clear perishable inventory management perspective" | The paper was organised around a learning loss (regret, SPO+, speed-ups). Inventory decisions were fixed open-loop for a whole horizon, nobody observed stock or its age before ordering, the fleet was so tight (utilisation 0.94-0.99) that every method had about 90% fill rate, and outcomes were reported as regret against a heuristic "clairvoyant" plan. | The study becomes a **periodic-review perishable inventory problem with a routing coupling**. Every day each policy observes the age-structured stock of every store, decides replenishment, and ships it on capacitated routes; demand is then served FIFO, unmet demand is lost and stock past its shelf life is discarded. Results are expressed in inventory terms (total cost and components, fill rate, waste rate, stock on hand, delivery frequency, store-level service) and analysed across shelf life, cost ratio, fleet tightness, demand variability and issuing policy. |
| "therefore also falls short in comparisons with state-of-the-art methods" | Comparators were forecasting models (ridge, LightGBM, ...) and later a single open-loop adaptation of Crama et al.'s UL rule; every MILP returned a heuristic fallback. | Comparators are the methods an inventory/IRP reviewer expects, all closed-loop with identical information: (i) periodic base-stock (BSP), (ii) waste-anticipating base-stock BSP-EW (Broekmeulen & van Donselaar 2009; Haijema & Minner 2019), (iii) Crama et al. (2018) UL_lambda, (iv) rolling-horizon PIRP with point forecasts, (v) rolling-horizon PIRP with tuned quantile targets (cost-function approximation), (vi) budgeted robust targets (Bertsimas & Thiele 2006; Solyali et al. 2012), (vii) Scenario Predict-then-Optimize ScenPO-SS (Jia et al. 2025), (viii) the proposed DFL, plus (ix) a perfect-information reference. All lookahead MILPs are solved to a proven gap (<=0.5% in evaluation); no heuristic fallback is used for any reported decision. |

## 1. Assessment of the previous plan (v1)

The v1 plan was careful about evidence integrity (it withdrew false M5, regret and speed-up claims and corrected misattributed citations), and those corrections are retained. As a resubmission strategy it was not adequate:

1. **Open-loop design.** A five-day plan fixed at episode start without observing stock is not how perishable inventory is managed; it also made the published UL_lambda rule (a state-dependent rule) impossible to implement faithfully.
2. **Capacity-starved instance.** Fleet utilisation of 0.94-0.99 forced ~10% lost sales for every method, so method differences were dominated by the fleet bottleneck.
3. **Solver failure.** 100% of evaluation MILPs (MTZ, vehicle-indexed, 0.5 s limit) returned the construction fallback, so no reported plan was an optimised plan.
4. **No contemporary comparator.** The plan itself recorded Gate 2 (state-of-the-art comparison) as not passed.
5. **Resulting manuscript** was a seven-page negative report that states it cannot address the editor's concern; it would be desk-rejected again.

Decision: replace v1 by the design below; keep v1's integrity rules (no claim beyond evidence, synthetic data disclosed, paired episode-level inference, every number traceable to a result file).

## 2. Research questions

* **RQ1 (inventory policies under routing coupling).** How do classical perishable inventory policies (BSP, BSP-EW), a published stochastic PIRP rule (UL_lambda) and integrated rolling-horizon PIRP policies differ in cost, waste and service when deliveries share a capacitated fleet?
* **RQ2 (how to use forecasts).** Given the same contextual forecasts, does it matter whether they enter the lookahead as means, quantiles, robust budgets, scenarios (ScenPO-SS), or decision-focused targets?
* **RQ3 (when).** How do shelf life, lost-sales/waste cost ratio, fleet tightness, demand variability and the issuing policy (FIFO vs partial LIFO) change these answers?

## 3. Model and methods (implemented in `pirp/`)

* **System:** depot + 10 stores (Solomon R101 geometry), K = 2 vehicles of capacity Q = fleet ratio x mean daily demand / K, shelf capacity = 3 days of mean store demand, shelf life m in {2,3,4}, lead time zero, daily review over 21 days per episode, lost sales, FIFO issuing (partial LIFO as robustness).
* **Costs:** holding h = 1 per unit-night, waste p_s = 5 per expired unit, lost sales p_u = 20 (10, 40 in sensitivity), routing 1 per distance unit.
* **Demand:** contextual negative binomial with day-of-week, announced promotions (higher mean and variance), temperature (forecast error grows with lead time), holidays and a latent AR(1) store level. Hand-specified; disclosed as synthetic. Train/validation/test episodes are independent (100/20/40).
* **Forecasts:** one LightGBM mean model and 11 LightGBM quantile models shared by all forecast-based policies.
* **Exact routing:** subset TSP costs (Held-Karp) and exact CVRP packing of any delivery vector; lookahead MILP uses a vehicle-assignment formulation with exact subset route costs (only assignment binaries are branched).
* **Proposed DFL:** planning targets = tuned quantile + learned, context- and stock-age-dependent correction. Trained by backpropagating realised FIFO cost through a differentiable coverage layer with routes fixed by the MILP oracle (solution caching between re-solves); model selection by closed-loop validation cost, starting from the tuned quantile policy.

## 4. Theory retained from the rejected manuscript (corrected) and verified in code

| Original claim | Status | Revised statement | Verification (`experiments/verify_theory.py`) |
|---|---|---|---|
| Lemma 1: age-increasing holding costs make every optimal solution FIFO | Incorrect as stated (counterexample: an old unit expiring today and a fresh unit nobody will buy; selling the fresh one saves holding) | Theorem 2: with an early-disposal option the recourse LP has an optimal greedy-FIFO solution and V_LP = C_FIFO - h * Omega (Omega = unit-nights of units that expire unsold); hence V_LP <= C_FIFO with equality whenever no doomed unit is carried | 3,000 random instances |
| Theorem 1: complete recourse, compactness, solver safety | Correct but trivial; retained briefly | Theorem 1 for both lookahead models (disposal slack in the scenario model) | MILP status over extreme demands |
| Proposition: recourse duals in [-p_u, p_s] | Correct only with a disposal option | Proposition 3: duals are subgradients and lie in [-p_u, p_s]; FIFO-simulator derivatives lie in [-p_u, p_s + (m-1)h] | Same 3,000 instances, subgradient inequality tested |
| SPO+ convexity / upper bound | Known result (Elmachtoub & Grigas 2022), not used by the revised method | Cited, not re-proved | n/a |
| Stop-gradient surrogate | Heuristic | Replaced by Proposition 4: stationary points of the DFL objective satisfy the stochastic first-order condition for fixed visits; with one-day shelf life they are the perishable newsvendor quantile p_u/(p_u+p_s) | SGD experiment |
| 5.66x solver speed-up, 22% vs 35% regret, 81-cell factorial, M5 calibration | Withdrawn (not reproducible / false provenance) | Replaced by closed-loop experiments | n/a |

## 5. Experimental protocol

* **Base configuration** (m = 3, p_u = 20, fleet ratio 1.25, NB size 8, FIFO) plus one-factor-at-a-time variants: m in {2, 4}; p_u in {10, 40}; fleet ratio in {1.1, 1.6}; high variability (NB size 3); partial LIFO (30% of demand takes the freshest units, evaluation only).
* **Tuning on validation episodes only:** BSP/BSP-EW (R, alpha), UL (lambda, alpha, beta), quantile lookahead (alpha), robust (Gamma); DFL early stopping. Test episodes are touched once.
* **Primary inference (base configuration):** paired episode-level differences in total cost, DFL vs each of {quantile lookahead, ScenPO-SS, robust, BSP-EW, UL}; 10,000-replicate paired bootstrap CIs, paired t-test and Wilcoxon, Holm adjustment over the five comparisons. Variants are reported descriptively with paired CIs.
* **Solver audit:** status, proven gap, time for every lookahead decision; any decision without a proven solution is reported.

## 6. Phases and gates

| Phase | Content | Gate | Status |
|---|---|---|---|
| 0 | Repository clean-up; remove legacy code/results; single manuscript source | Reproducible from `README.md` commands | Done |
| 1 | Rebuild model, simulator, exact routing, MILPs, policies | Unit/smoke tests; solver proves gap on all evaluation solves | Done (0 fallbacks, 0 time-limit stops so far) |
| 2 | Theory verification | All checks pass or statements are corrected | Done (original FIFO lemma corrected) |
| 3 | Experiments (base + variants) | Every manuscript number generated by `experiments/` scripts | Done: all 10 configurations, 0 fallbacks, 0 time-limit stops, max gap 0.5% |
| 4 | Manuscript rewrite (`revision_1/manuscript.tex`) | Inventory-centred framing; claims bounded by results; verified bibliography | Done (text covers all 10 configurations) |
| 5 | Compile, visual check, cover letter / response to editor | Clean compile; references resolve | Clean compile (30 pages, abstract 223 words); author's visual read-through pending |

**Outcome (honest reading):** stock-age-based rules with daily delivery (UL, BSP, BSP-EW) are the cheapest policies in 9 of 10 settings (2.4-8.9% below the best integrated policy); with three times higher routing cost ScenPO-SS is best overall. Among integrated policies none dominates: DFL is best in 5 settings, ScenPO-SS in 4, LA-Robust in 1. DFL improves the quantile lookahead in 9 of 10 settings (significant after Holm in 5) and is never significantly worse. The manuscript was retitled and framed around these findings.

## 7. Honesty rules carried over

* Synthetic demand is described as such; no M5 or field-calibration claims.
* A comparator is called a reproduction only where it follows the published rule; adaptations (perishable FIFO in ScenPO-SS, LightGBM quantiles instead of MQ-RNN, BSP-EW parameterisation) are stated.
* Negative or null results for DFL are reported as they come out.
