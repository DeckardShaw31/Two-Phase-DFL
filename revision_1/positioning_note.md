# Positioning Note: Perishable Inventory-Routing under Demand Uncertainty

**Target journal:** *International Journal of Production Economics*  
**Decision letter:** PROECO-D-26-05160  
**Status:** Revised framing and primary synthetic run audited; post-hoc finite-scenario robust adaptation evaluated; source-method comparator gate not passed.

## Editorial problem

The editor said the submission did not present a clear perishable inventory-management perspective and did not compare against strong state-of-the-art methods. The paper is therefore organized around replenishment, age-specific carryover, expiry, service, and shared fleet capacity. DFL is one candidate decision method within that operational question.

## Central question

How do point-forecast PTO, cost-sensitive quantile PTO, a perishable inventory policy adaptation, and an approximate DFL hybrid compare on operating cost, service, expiry, carryover, and routing under the same held-out demand episodes?

The current experiment tests only one synthetic configuration: ten stores on a clustered Solomon-derived layout, a five-period horizon, shelf life three, two vehicles with capacity 100, store capacity 70, zero initial stock, holding cost 1, expiry/disposal penalty 5, and unmet-demand penalty 20.

## What the evidence can support

- The demand process is hand-specified synthetic data. It is neither M5-based nor empirically calibrated, and there are no operational stock, expiry, service, or route records.
- The rebuilt dataset has 400 independently generated episodes split 280/60/60. Predictors use episode-start features; realized demand and weather inside the five-period horizon are excluded.
- The primary comparison evaluates LightGBM point PTO, LightGBM quantile PTO, an open-loop validation-residual adaptation inspired by Crama et al.'s UL-lambda policy, and an approximate two-phase DFL hybrid.
- A separate post-hoc finite-scenario robust plan selector uses whole validation residual paths, chooses one fixed schedule by worst-case validation-scenario FIFO cost, and evaluates it on the same 60 test episodes. It is a custom restricted-candidate adaptation, not He et al.'s source algorithm, and all 300 candidate MILP calls used fallback.
- Physical outcomes are evaluated with a common FIFO simulator, separating expiry spoilage from capacity disposal and recording shortages, fill rate, inventory by age, deliveries, and routing.
- The optimizer requests time-limited HiGHS MILPs but can select a feasibility-checked construction fallback. The benchmark reports solver status and does not assume optimality.

The audited run has mean episode cost 2795.0 (SD 1212.1) for DFL and 2738.1 (SD 1176.6) for point PTO. The paired DFL-minus-point difference is +56.9 (95% episode-bootstrap CI [-14.2, 129.1], Holm-adjusted p=.137), so the primary test does not show lower cost for DFL. DFL is lower-cost than the open-loop UL-lambda-inspired adaptation by 263.3 units (95% CI [-340.0, -183.4], Holm-adjusted p=2.75e-8), but that adaptation is not the published dynamic Crama policy. All 300 point-PTO, quantile-PTO, and DFL evaluation plans used the feasible construction fallback; no evaluation MILP was optimal. The former 81-cell factorial, 13-baseline comparison, M5-calibration statement, clairvoyant-regret results, and solver-speedup claim are withdrawn from the revised evidence.

The exploratory robust adaptation has mean episode cost 2712.3 (SD 1148.9), fill rate 0.904, and zero capacity discards. Its post-hoc paired cost differences are -25.8 versus point PTO (95% bootstrap CI [-85.2, 32.7]) and -82.7 versus DFL ([-150.0, -14.9]). These are not confirmatory tests and do not establish the performance of He et al.'s method; the fallback-based candidate plans also limit the operational comparison.

## Literature and comparison boundary

The source audit confirms that Crama et al. (2018) is a direct stochastic perishable inventory-routing paper. Its UL-lambda policy updates after observed demand each period; the code here instead fixes an open-loop plan and propagates predicted inventory. This comparator is an adaptation, not a faithful reproduction.

He et al. (2024) is a real robust perishable PIRP study. Its source methods are not reproduced; the added finite-scenario plan selector is custom and does not implement its robust counterpart, Benders decomposition, or ILS-BD. Jia et al. (2025), *Scenario Predict-then-Optimize for Data-Driven Online Inventory Routing*, is a real recent online inventory-routing method using multi-horizon quantile-RNN forecasts and sampled demand scenarios in stochastic optimization. The full formulation reviewed has no finite-life age-cohort state; adding FIFO expiry would be an adaptation. Its online decision timing also needs to be reconciled with this study's fixed-plan protocol.

## Resubmission decision

The completed, audited primary run plus exploratory adaptation provide a corrected synthetic pilot and an inventory-centered manuscript draft. They do not satisfy the editor's request for strong current operational comparisons. Before claiming that concern is resolved, implement and validate a direct robust perishable-PIRP comparator, preferably He et al. (2024) after full-source access to its formulation, uncertainty set, and solution procedure. A scenario-based comparator such as Jia et al. (2025) would require an explicit FIFO/expiry adaptation and matched online decision timing. A wider shelf-life/capacity design and operational data would strengthen the paper further. Do not describe the current study as state of the art or field validated.

