> Historical draft retained for provenance. Superseded by `revision_1/revision_plan.md` and the revised evidence package. Earlier M5-calibration, solver-optimality, comparator, or result claims in this file are not current evidence.

# Phase 1 Positioning Note (Extended Reference Archive)

**Document:** `revision_1/positioning_note_detailed_archive.md`  
**Target Journal:** *International Journal of Production Economics* (IJPE)  
**Manuscript Title:** Perishable Inventory-Routing under Demand Uncertainty: Operational Value and Limits of Decision-Focused Learning  
**Author:** Nguyễn Nhật Thanh  
**Date:** October 2026 (Revision 1)

---

## 1. Executive Summary & Editorial Pivot

### 1.1 Context of the Editorial Decision
The initial submission (Ms. Ref. No. PROECO-D-26-05160) received an immediate desk rejection from Principal Editor Prof. Stefan Minner:
> *"Unfortunately, the manuscript does not take a clear perishable inventory management perspective and therefore also falls short in comparisons with state-of-the-art methods."*

A rigorous audit of the original manuscript confirms the editor's assessment:
1. **Misaligned Core Narrative:** The paper was framed predominantly around Computer Science and Machine Learning methodology (Decision-Focused Learning [DFL], bilevel programming, dual gradient derivation, surrogate losses, and solver profiling). The central research questions and all four primary contributions focused on DFL algorithm mechanics rather than the physical inventory-routing problem.
2. **Missing Operational Inventory Perspective:** The fundamental trade-offs in perishable supply chains—namely, how discrete shelf life ($\tau$), inventory carried forward between planning periods, shared vehicle payload constraints ($Q$), and asymmetric spoilage ($p_s$) versus stockout penalties ($p_u$) jointly dictate replenishment timing and service—were treated merely as a computational testbed rather than the core scientific inquiry.
3. **Restricted Comparator Horizon:** The benchmark compared DFL solely against standard machine learning predictors plugged into a static MILP (Predict-then-Optimize [PTO]), lacking operational inventory-control benchmarks.

### 1.2 The Phase 1 Strategic Pivot
Phase 1 re-anchors the manuscript firmly into **production economics and perishable inventory management**. Decision-Focused Learning is repositioned as **one candidate solution technology** investigated to determine where, how, and why aligning replenishment decisions with operational economics yields value—and crucially, where its physical boundary conditions lie.

---

## 2. Revised Title and Article Metadata

- **Previous Title:**  
  *End-to-End Decision-Focused Learning for Perishable Inventory-Routing under Asymmetric Spoilage and Stockout Costs*
- **Revised Working Title:**  
  **Perishable Inventory-Routing under Demand Uncertainty: Operational Value and Limits of Decision-Focused Learning**
- **Target Audience:** Production economists, logistics planners, and operations researchers interested in perishable inventory control, multi-period replenishment, vehicle routing, and data-driven prescriptive decision-making.

---

## 3. The Central Research Question

> **Central Question:**  
> *How do product shelf life and shared fleet capacity shape the trade-off between replenishment, carryover, spoilage, and service under uncertain demand, and under what conditions does decision-focused learning improve those outcomes relative to standard forecast-driven replenishment policies?*

### Three Recast Inventory-Centered Research Questions (RQs)

1. **RQ1 (Shelf-Life & Carryover Dynamics):**  
   *How do finite shelf life ($\tau \in \{2, 3, 5\}$) and age-structured carryover inventory govern the physical trade-off between delivery frequency, routing consolidation, and expiry waste across multi-period planning horizons under uncertain customer demand?*
2. **RQ2 (Fleet Capacity & Asymmetric Cost Structures):**  
   *How do shared vehicle fleet capacity constraints ($\gamma \in \{0.8, 1.0, 1.5\}$) and punitive stockout-to-spoilage cost ratios ($p_u/p_s \in \{2, 4, 10\}$) dictate the feasibility and value of tactical store-level inventory buffering, fleet utilization, and spatial service equity across retail networks?*
3. **RQ3 (Operational Value & Structural Boundary Conditions of DFL):**  
   *Under what operational conditions (fleet tightness regimes, spatial network topologies, and perishability horizons) does end-to-end decision-focused learning generate meaningful operational cost and service improvements over conventional predict-then-optimize workflows, and where do physical delivery bottlenecks render downstream-aware training ineffective or inferior to point forecast precision?*

*(Note on Comparator Scope: The current benchmark evaluates 14 forecast-driven replenishment policies spanning statistical, regularized, tree-based, deep sequence, pinball, quantile, safety stock, and DFL models. Comparing against specialized operational inventory-control policies, such as age-aware base-stock policies, rolling-horizon model predictive control, and adaptive PIRP heuristics, is the designated task of Phase 2.)*

---

## 4. The 4-Layer Decision Stack

To prevent conflating statistical estimation with physical logistics outcomes, the revised manuscript explicitly decouples the decision problem into four operational layers:

```
[Layer 1: Predictive Engine]
Exogenous Features (x_{it})  -->  Predicted Demands (\hat{d}_{it})
  Evaluated by Statistical Accuracy: MSE, MAE

                |  (Fed into PIRP Decision Formulation)
                v

[Layer 2: Planned Replenishment & Routing Schedule]
First-Stage Decisions: Vehicle Tours (x_{ijt}^k) & Delivery Quantities (q_{it}^k)
  Constrained by Fleet Payload (K * Q) and Store Storage (C_i)

                |  (Realized under True Physical Demand d_{it})
                v

[Layer 3: Physical Inventory & Age Dynamics]
FIFO Cohort Consumption (w_{ita}), End-of-Period Carryover (I_{ita}),
Unmet Demand Stockout (y_{it}^+), Expiry & Excess Spoilage (y_{it}^-)

                |  (Monetized via Economic Parameter Hierarchy)
                v

[Layer 4: Realized Operating Costs & Decision Regret]
Total Cost: C_{total} = Routing + Holding (h_a) + Spoilage (p_s) + Stockout (p_u)
Evaluated by Decision Regret against Full-Information Clairvoyant Benchmark
```

---

## 5. Revised Abstract

> **Abstract:**  
> Multi-period distribution of perishable goods requires coordinating retail replenishment and vehicle fleet routing under uncertain demand and finite product shelf lives ($\tau \in \{2, 3, 5\}$). In the cost structure studied here, the unit stockout penalty ($p_u$) is 2--10 times the unit spoilage penalty ($p_s = 5h$) and 10--50 times the per-period unit holding cost ($h$). Conventional logistics planning often separates forecasting from optimization; statistical forecasts are then passed to a mixed-integer linear program (MILP) to schedule routes and delivery quantities. This separation can produce a mismatch between forecast accuracy and inventory-routing performance because forecast losses do not directly encode downstream operational costs.
> 
> This paper investigates operational trade-offs in the multi-period perishable inventory-routing problem (PIRP) and evaluates decision-focused learning (DFL) as one solution approach. An age-structured recourse model represents FIFO consumption, inventory age, and recourse. A synthetic benchmark calibrated against M5 retail time series spans 81 configurations of topology, shelf life, asymmetric cost, and fleet capacity.
> 
> In the factorial results, reported spoilage costs range from $\$9.23$ to $\$18.80$ at $\tau=2$, from $\$1.14$ to $\$5.87$ at $\tau=3$, and are $\$0.00$ at $\tau=5$. The capacity summary averages 27 configurations per regime for Ridge, LightGBM, and Two-Phase DFL. Stockout costs range from $\$7,581$ to $\$7,651$ under tight capacity ($\gamma=0.8$, $Q=80$), accounting for over $88\%$ of total costs for these three methods; under balanced capacity ($\gamma=1.0$, $Q=100$), the reported range is $\$3,068$--$\$3,296$. On the separate primary benchmark (clustered topology, $\tau=3$, $p_u/h=20$, $\gamma=1.0$), Two-Phase DFL has mean regret of $22.28\%$ versus $35.51\%$ for LightGBM, with lower regret than all 13 baselines after Holm adjustment. In the capacity summary, mean regret for Two-Phase DFL versus LightGBM is $6.91\%$ versus $5.60\%$ at $\gamma=0.8$, $15.64\%$ versus $21.54\%$ at $\gamma=1.0$, and $85.34\%$ versus $88.79\%$ at $\gamma=1.5$. Results are specific to the synthetic benchmark; a separate 20-epoch solver-profiling experiment records a $5.66\times$ speedup over direct MILP training.

---

## 6. Three Recast Primary Operational Contributions

The revised contributions place operational and inventory economics at the center, keeping mathematical formulation and training algorithms strictly as supporting methodology:

### Primary Contribution 1: Operational Characterization of Shelf-Life and Capacity Interactions
The 81-configuration factorial analysis reports spoilage costs across shelf-life levels and a separate capacity summary over 27 configurations per regime for Ridge, LightGBM, and Two-Phase DFL:
- **Shelf-life and spoilage:** Reported spoilage costs range from $\$9.23$ to $\$18.80$ at $\tau=2$, from $\$1.14$ to $\$5.87$ at $\tau=3$, and are $\$0.00$ at $\tau=5$. The manuscript shows period-level carryover for the primary benchmark, but does not quantify cross-shelf-life carryover differences in the factorial summary.
- **Capacity and stockout cost:** Under tight capacity ($\gamma=0.8$, $Q=80$), stockout costs range from $\$7,581$ to $\$7,651$ and exceed $88\%$ of total cost for the three methods in the summary. At balanced capacity ($\gamma=1.0$, $Q=100$), the reported range is $\$3,068$--$\$3,296$.

### Primary Contribution 2: Empirical Boundary Conditions of Decision-Focused Learning
Across an 81-configuration factorial grid, we evaluate the operational value of aligning replenishment decisions with downstream costs:
- **Capacity-grid comparison:** In the capacity summary, mean regret for Two-Phase DFL versus LightGBM is $6.91\%$ versus $5.60\%$ at $\gamma=0.8$, $15.64\%$ versus $21.54\%$ at $\gamma=1.0$, and $85.34\%$ versus $88.79\%$ at $\gamma=1.5$. The DFL advantage is not uniform and remains modest at slack capacity.
- **Separate primary benchmark:** On clustered topology at the baseline setting ($\tau=3$, $p_u/h=20$, $\gamma=1.0$), Two-Phase DFL has mean regret of $22.28\%$ versus $35.51\%$ for LightGBM. This primary-benchmark result is distinct from the capacity-grid averages above.

### Primary Contribution 3: Spatial Service Equity and Multi-Metric Operational Diagnostics
The manuscript compares 14 predictive models on regret and forecast metrics and reports physical inventory outcomes for the methods and settings shown in the corresponding tables and figures:
- **Spatial rationing:** A store-level diagnostic reports that peripheral stores (4, 6, and 10) account for $87.7\%$ of shortages in the evaluated setting, with Peripheral Rationing Indices up to $4.47$.
- **Statistical inference:** Holm-adjusted paired tests are reported for the primary benchmark comparisons; these significance results are not generalized to every capacity-grid average.

### Supporting Methodological and Computational Formulation
To evaluate these operational questions within an integrated framework, we formulate an age-structured two-stage recourse PIRP that preserves strict FIFO depletion (\textbf{Lemma 1}) and complete recourse (\textbf{Theorem 1}). Computationally, we employ a two-phase LP-relaxation warmup to MILP fine-tuning schedule (\textbf{Algorithm 1}) with a custom stop-gradient surrogate, achieving a $5.66\times$ solver speedup in our profiling experiments.

---

## 7. Language De-escalation & Nuanced Framing

To adhere to IJPE's scientific standards, all categorical and hyperbolic assertions from the initial draft have been revised:

| Original Phrasing | Revised Nuanced Phrasing | Rationale |
| :--- | :--- | :--- |
| *"The Methodological Failure of Two-Stage PTO in Perishable Operations"* | *"Operational Trade-offs in Perishable Inventory-Routing under Demand Uncertainty"* | Avoids dismissive claims; PTO remains standard and competitive under tight capacity. |
| *"Standard loss functions systematically fail in perishable logistics"* | *"Symmetric forecasting losses are economically misaligned with downstream cost asymmetries"* | Accurately describes economic trade-offs without overstating universal failure. |
| *"Two-Phase DFL solves the perishable IRP"* | *"Two-Phase DFL mitigates operational regret in flexible fleet regimes"* | Acknowledges that DFL is an integrated heuristic approach with operational limits. |
| *"Zero tactical slack makes pre-positioning physically infeasible"* | *"In the tested tight-capacity setting ($\gamma = 0.8, Q = 80$), delivery limits constrain forward buffering across all models"* | Replaces universal physical impossibility with an empirical finding for the tested setup. |
| *"Catastrophic stockouts across all baselines"* | *"Costly retail stockouts and uneven service dispersion"* | Uses professional operations research terminology. |
| *"Eliminates the computational bottleneck"* | *"Accelerates training convergence by $5.66\times$ in the tested solver profiling setup"* | Grounds speedup claims within the specific experimental profiling setup. |

---

## 8. Gate 1 Compliance Audit

> **Gate 1 Criterion:**  
> *If the paper has no inventory-management implication that remains interesting when the DFL algorithm is removed, the IJPE reframing is not yet substantive enough.*

### Why the Revised Paper Has an Inventory-Management Contribution:
Even without the DFL method, the tested simulation provides descriptive inventory findings for the evaluated configurations:

1. **Fleet Capacity and Shortages:**  
   In the tested tight fleet regime ($\gamma = 0.8, Q = 80$), cumulative transport capacity ($800$ units) is below aggregate demand ($>900$ units). In the capacity summary, stockout penalties constitute over $88\%$ of total costs ($\$7,581$--$\$7,651$) for the three reported methods. This is a finding for the simulated setup, not a general service guarantee or causal estimate.
2. **Shelf-Life and Spoilage Outcomes:**  
   Reported spoilage costs range from $\$9.23$ to $\$18.80$ for $\tau=2$, from $\$1.14$ to $\$5.87$ for $\tau=3$, and are $\$0.00$ for $\tau=5$. The manuscript reports period-level carryover for its primary benchmark, but the factorial summary does not quantify carryover differences across shelf lives.
3. **Spatial Service Dispersion in the Evaluated Network:**  
   Network diagnostics show that under tight fleet capacity, reported shortages are heavily concentrated on peripheral retail stores (stores 4, 6, 10), which exhibit Peripheral Rationing Indices up to $4.47$ and account for $87.7\%$ of total network stockouts in the tested setup.
4. **Economic Penalties of Symmetric Forecasting under Asymmetric Costs:**  
   The simulation results quantify the cost consequences of symmetric forecast estimation when stockout penalties dominate holding and disposal costs ($p_u/h \in \{10, 20, 50\}$).

---

*End of Extended Reference Archive.*

