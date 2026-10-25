"""
Empirical Adversarial Stress Test Suite for Milestone 1 Theoretical Claims
Executed by Challenger 1 (Theory & FIFO Stress Testing Challenger).

Target Audits:
1. Lemma 1 & Remark 1:
   - Construct adversarial LP / MILP instances where uniform holding costs (h_a = h)
     cause non-FIFO simplex degeneracy (underdetermined reduced costs permitting consumption
     of young inventory while leaving older inventory unconsumed).
   - Verify that age-differentiated holding costs h_a = h + (a-1)*eps (eps = 0.01) strictly
     restore 100% FIFO consumption discipline (older cohorts 100% depleted before younger cohorts).
2. Theorem 1 Complete Recourse with Retail Storage Capacity:
   - Construct boundary test cases where deliveries plus carryover exceed Ci = 70.0 and demand
     is zero (d_tilde = 0).
   - Verify that without proactive disposal slacks (delta_{i,t,a} for a < tau), the recourse
     problem is PRIMAL INFEASIBLE.
   - Verify that WITH proactive disposal slacks delta_{i,t,a} >= 0 (as defined in Definition 2
     and Theorem 1), the recourse problem maintains complete recourse, is strictly feasible,
     bounded (0 <= Q < infty), and caps retail inventory at Ci = 70.0.
3. Master Recourse Balance Identity:
   - Verify algebraically and empirically that demand non-negativity y_{it}^+ >= 0 retains
     customer demand d_tilde on the right-hand side of the recourse constraints.
   - Verify that relaxing y_{it}^+ >= 0 produces unbounded or false negative-cost solutions,
     proving that demand cannot be unconditionally eliminated from constraints.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from src.optimization.pirp_model import PIRPModel

# =====================================================================
# PART 1: LEMMA 1 & REMARK 1 — SIMPLEX DEGENERACY VS FIFO RESTORATION
# =====================================================================

class TestLemma1SimplexDegeneracyAndFIFORestoration:
    """
    Adversarial verification of Lemma 1 and Remark 1:
    - Demonstrates that uniform holding costs induce degenerate LP faces where non-FIFO
      consumption incurs identical operational cost.
    - Demonstrates that age-differentiated holding costs strictly penalize non-FIFO solutions
      and restore unique FIFO consumption.
    """

    def test_lp_flat_objective_face_under_uniform_holding_costs(self):
        """
        Construct a 2-period perishable LP instance where uniform holding costs
        produce a flat objective face between FIFO and non-FIFO policies.
        """
        # S_1 = 20.0, S_2 = 10.0.
        # Demand d_0 = 8.0, d_1 = 22.0.
        # Policy A (FIFO): w_2 = 8.0, w_1 = 0.0.
        # Carryover: I_1 = 20.0, I_2 = 2.0.
        # Holding cost at t=0 under uniform h: 1.0 * 20.0 + 1.0 * 2.0 = 22.0.

        # Policy B (Non-FIFO): w_1 = 8.0, w_2 = 0.0.
        # Carryover: I_1 = 12.0, I_2 = 10.0.
        # Holding cost at t=0 under uniform h: 1.0 * 12.0 + 1.0 * 10.0 = 22.0.

        cost_fifo_uniform = 1.0 * 20.0 + 1.0 * 2.0
        cost_non_fifo_uniform = 1.0 * 12.0 + 1.0 * 10.0
        assert np.isclose(cost_fifo_uniform, cost_non_fifo_uniform), (
            "Under uniform holding costs, FIFO and non-FIFO policies must have identical cost!"
        )

        # Now verify age-differentiated holding costs: h_a = h + (a-1)*eps, eps = 0.01:
        eps = 0.01
        h1 = 1.0
        h2 = 1.0 + eps

        cost_fifo_diff = h1 * 20.0 + h2 * 2.0  # 20.0 + 2.02 = 22.02
        cost_non_fifo_diff = h1 * 12.0 + h2 * 10.0  # 12.0 + 10.10 = 22.10

        diff = cost_non_fifo_diff - cost_fifo_diff
        expected_diff = (h2 - h1) * 8.0  # 0.01 * 8.0 = 0.08
        assert np.isclose(diff, expected_diff), f"Cost difference mismatch: {diff} vs {expected_diff}"
        assert diff > 0.0, "Age-differentiated costs must strictly penalize non-FIFO consumption!"

    def test_simplex_degeneracy_empirical_lp_solver(self):
        """
        Formulate this exact multi-period inventory LP in SciPy linprog and prove:
        1. With uniform costs, an adversarial tie-break direction allows non-FIFO optimal solutions.
        2. With age-differentiated costs (eps = 0.01), the unique LP optimal solution strictly enforces FIFO.
        """
        # Variables: z = [w_1, w_2, I_1, I_2, w_1_next, w_2_next]
        # Constraints:
        # w_1 + I_1 = 20.0           (Fresh age 1 balance at t=0)
        # w_2 + I_2 = 10.0           (Carryover age 2 balance at t=0)
        # w_1 + w_2 = 8.0            (Demand at t=0)
        # w_1_next - I_1 = 0         (Consumption of age 1 carryover at t=1)
        # w_2_next - I_2 = 0         (Consumption of age 2 carryover at t=1)
        # Bounds: all variables >= 0.

        A_eq = np.array([
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        ])
        b_eq = np.array([20.0, 10.0, 8.0, 0.0, 0.0])

        # 1. Uniform holding cost vector:
        # Cost is h * I_1 + h * I_2.
        # Add an adversarial infinitesimal perturbation (-1e-7 on w_1) representing simplex
        # tie-break or roundoff favoring younger stock:
        c_uniform_adversarial = np.array([-1e-7, 0.0, 1.0, 1.0, 0.0, 0.0])

        res_uniform = linprog(
            c=c_uniform_adversarial,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * 6,
            method="highs",
        )
        assert res_uniform.success is True

        w1_u, w2_u = res_uniform.x[0], res_uniform.x[1]
        # Under adversarial tie-break, uniform cost fails FIFO:
        # Younger stock is consumed (w1 > 0) while older stock is left unconsumed (w2 < 8.0)
        assert w1_u > 0.0, f"Expected non-FIFO consumption under adversarial tie-break, got w1={w1_u}"
        assert w2_u < 8.0, f"Expected incomplete older stock depletion, got w2={w2_u}"

        # 2. Age-differentiated cost vector:
        # Cost is h_1 * I_1 + h_2 * I_2, where h_1 = 1.0, h_2 = 1.0 + 0.01 = 1.01.
        # Retain the exact same adversarial perturbation on w_1:
        c_differentiated = np.array([-1e-7, 0.0, 1.0, 1.01, 0.0, 0.0])

        res_diff = linprog(
            c=c_differentiated,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * 6,
            method="highs",
        )
        assert res_diff.success is True

        w1_d, w2_d = res_diff.x[0], res_diff.x[1]
        # Under age-differentiated holding costs:
        # Older stock MUST be 100% consumed before any fresh stock is consumed!
        assert np.isclose(w2_d, 8.0, atol=1e-5), f"Older stock must be fully consumed: {w2_d}"
        assert np.isclose(w1_d, 0.0, atol=1e-5), f"Younger stock must not be consumed: {w1_d}"
        # Complementarity w_1 * (S_2 - w_2) == 0
        comp_slack = w1_d * (10.0 - w2_d)
        assert np.isclose(comp_slack, 0.0, atol=1e-7), f"Complementarity violated: {comp_slack}"

    @pytest.mark.parametrize("tau", [2, 3, 5])
    def test_lemma1_full_pirp_model_restoration(self, tau):
        """
        Verifies on full PIRPModel instances across shelf lives tau in {2, 3, 5} that
        aging_surcharge_eps = 0.01 strictly enforces FIFO across all stores and periods.
        """
        N, T = 3, 4
        model = PIRPModel(
            num_stores=N,
            horizon_T=T,
            shelf_life_tau=tau,
            aging_surcharge_eps=0.01,
            continuous_relaxation=True,
        )

        # Adversarial demand pattern forcing multi-cohort carryovers
        rng = np.random.default_rng(42)
        demand = rng.uniform(5.0, 25.0, size=(N, T))
        demand[:, 0] = 5.0    # Low early demand creates carryover
        demand[:, 1] = 20.0   # Higher demand consumes mixed cohorts

        sol = model.solve(demand)
        assert sol.success is True

        # Check FIFO complementarity across all stores, periods, and cohort pairs:
        # w[a_young] * (S[a_old] - w[a_old]) == 0 for all a_young < a_old
        for t in range(T):
            for i in range(N):
                # Reconstruct available stock S for this period:
                S = np.zeros(tau)
                S[0] = sol.q_total[i, t]
                if t > 0:
                    for a in range(1, tau):
                        S[a] = sol.I_inventory[i, t - 1, a - 1]

                w = sol.w_consumed[i, t, :]
                for a_y in range(tau):
                    for a_o in range(a_y + 1, tau):
                        if w[a_y] > 1e-4:
                            assert np.isclose(w[a_o], S[a_o], atol=1e-3), (
                                f"FIFO VIOLATION in PIRPModel at store {i}, period {t}, tau={tau}: "
                                f"Younger cohort {a_y+1} consumed {w[a_y]:.2f} while older cohort {a_o+1} "
                                f"had unconsumed stock: {w[a_o]:.2f} / {S[a_o]:.2f}"
                            )


# =====================================================================
# PART 2: THEOREM 1 — COMPLETE RECOURSE WITH RETAIL STORAGE CAPACITY
# =====================================================================

class TestTheorem1CompleteRecourseRetailCapacity:
    """
    Adversarial verification of Theorem 1:
    - Verifies that when deliveries + carryover exceed Ci = 70.0 and demand is zero (d_tilde = 0),
      proactive disposal slacks delta_{i,t,a} are MATHEMATICALLY ESSENTIAL for complete recourse.
    - Demonstrates that without delta (delta=0), the recourse problem is PRIMAL INFEASIBLE.
    - Demonstrates that WITH delta >= 0, the recourse problem is GUARANTEED PRIMAL FEASIBLE,
      bounded, and properly penalizes excess inventory at unit rate p_s.
    """

    def test_recourse_infeasibility_without_proactive_disposal(self):
        """
        Boundary Case: q + I > Ci and d = 0.
        Without proactive disposal slacks delta_{i,t,a} for intermediate cohorts,
        the second stage recourse LP is strictly infeasible.
        """
        Ci = 70.0

        # Second-Stage Formulation WITHOUT proactive disposal slacks:
        # Variables: z = [w_1, w_2, I_1, I_2, y_pos, y_neg]
        # Constraints:
        # w_1 + w_2 + y_pos = 0.0          (Demand satisfaction, d=0)
        # I_1 + w_1 = q_delivered = 50.0   (Fresh inventory balance, delta_1 = 0)
        # I_2 + w_2 = carryover = 50.0     (Intermediate inventory balance, delta_2 = 0)
        # I_1 + I_2 <= Ci = 70.0           (Retail storage capacity)
        # Bounds: all variables >= 0.

        A_eq = np.array([
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ])
        b_eq = np.array([0.0, 50.0, 50.0])

        A_ub = np.array([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ])
        b_ub = np.array([Ci])

        c = np.array([0.0, 0.0, 1.0, 1.01, 20.0, 5.0])

        res_no_slack = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0, None)] * 6,
            method="highs",
        )

        # MUST FAIL with primal infeasibility!
        assert res_no_slack.success is False, (
            "Expected recourse to be primal infeasible without proactive disposal slacks!"
        )
        assert res_no_slack.status == 2, (
            f"Expected HiGHS status 2 (primal infeasible), got {res_no_slack.status} ({res_no_slack.message})"
        )

    def test_recourse_feasibility_with_proactive_disposal_slacks(self):
        """
        Boundary Case: q + I > Ci and d = 0.
        WITH proactive disposal slacks delta_{i,t,a} >= 0 (Theorem 1 Definition 2):
        The recourse problem is GUARANTEED PRIMAL FEASIBLE, bounded, and disposes
        exactly the excess inventory (30.0 units).
        """
        Ci = 70.0
        q_delivered = 50.0
        carryover_incoming = 50.0  # Total available = 100.0 > Ci

        # Variables: z = [w_1, w_2, I_1, I_2, delta_1, delta_2, y_pos, y_neg]
        # Constraints:
        # 1. Demand: w_1 + w_2 + y_pos = 0.0
        # 2. Fresh balance: I_1 + w_1 + delta_1 = 50.0
        # 3. Intermediate balance: I_2 + w_2 + delta_2 = 50.0
        # 4. Spoilage/Disposal linking: y_neg - delta_1 - delta_2 = 0.0
        # 5. Store capacity: I_1 + I_2 <= 70.0
        # Objective: min h_1*I_1 + h_2*I_2 + p_s*y_neg + p_u*y_pos

        A_eq = np.array([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0],
        ])
        b_eq = np.array([0.0, 50.0, 50.0, 0.0])

        A_ub = np.array([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        b_ub = np.array([Ci])

        h1, h2 = 1.0, 1.01
        ps, pu = 5.0, 20.0
        c = np.array([0.0, 0.0, h1, h2, 0.0, 0.0, pu, ps])

        res_with_slack = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0, None)] * 8,
            method="highs",
        )

        assert res_with_slack.success is True, (
            f"Recourse with proactive slacks must be feasible: {res_with_slack.message}"
        )
        assert res_with_slack.status == 0, "HiGHS status must be 0 (optimal)"

        z = res_with_slack.x
        w1, w2 = z[0], z[1]
        I1, I2 = z[2], z[3]
        d1, d2 = z[4], z[5]
        y_pos, y_neg = z[6], z[7]

        # Invariants:
        # Zero demand implies zero consumption and zero stockout
        assert np.isclose(w1, 0.0)
        assert np.isclose(w2, 0.0)
        assert np.isclose(y_pos, 0.0)

        # Store capacity is strictly respected:
        assert np.isclose(I1 + I2, Ci), f"Carryover must be capped at Ci={Ci}, got {I1+I2}"

        # Proactive disposal absorbs exact excess: 100 - 70 = 30
        assert np.isclose(y_neg, 30.0), f"Excess disposal must equal 30.0, got {y_neg}"
        assert np.isclose(d1 + d2, 30.0)

        # Because h2 > h1 (older stock has higher holding cost), the optimal disposal
        # greedily discards older stock (d2 = 30.0, d1 = 0.0) to minimize holding costs!
        assert np.isclose(d2, 30.0, atol=1e-5), f"Expected older stock to be proactively discarded first: d2={d2}"
        assert np.isclose(d1, 0.0, atol=1e-5), f"Expected younger stock to be retained: d1={d1}"

        # Recourse cost is finite and non-negative:
        recourse_cost = res_with_slack.fun
        expected_cost = h1 * 50.0 + h2 * 20.0 + ps * 30.0  # 50(1.0) + 20(1.01) + 5(30) = 50 + 20.2 + 150 = 220.2
        assert np.isclose(recourse_cost, expected_cost, atol=1e-4)

        # Theoretical upper bound from Theorem 1 Part 3:
        # Q <= sum p_u * d + (h_tau_1 * tau + p_s) * q
        upper_bound = pu * 0.0 + (h2 * 3 + ps) * (q_delivered + carryover_incoming)
        assert recourse_cost <= upper_bound, f"Recourse cost {recourse_cost} exceeded upper bound {upper_bound}"

    def test_theorem1_random_boundary_fuzzing(self):
        """
        Fuzz 50 randomized extreme boundary instances where deliveries + carryover >> Ci
        and demands are near zero or sparse, confirming Complete Recourse non-emptiness.
        """
        rng = np.random.default_rng(2026)
        Ci = 70.0
        h1, h2 = 1.0, 1.01
        ps, pu = 5.0, 20.0

        for trial in range(50):
            q_val = rng.uniform(40.0, 70.0)
            carryover_val = rng.uniform(40.0, 70.0)  # Sum 80..140 > Ci (70)
            d_val = rng.choice([0.0, 1.0, 5.0])      # Very small demand

            A_eq = np.array([
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0],
            ])
            b_eq = np.array([d_val, q_val, carryover_val, 0.0])

            A_ub = np.array([
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ])
            b_ub = np.array([Ci])

            c = np.array([0.0, 0.0, h1, h2, 0.0, 0.0, pu, ps])

            res = linprog(
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=[(0, None)] * 8,
                method="highs",
            )

            assert res.success is True, f"Trial {trial} failed: {res.message}"
            z = res.x
            # Capacity satisfied
            assert z[2] + z[3] <= Ci + 1e-6
            # Non-negativity
            assert np.all(z >= -1e-6)
            # Finiteness
            assert np.isfinite(res.fun)
            assert res.fun >= 0.0


# =====================================================================
# PART 3: MASTER RECOURSE BALANCE IDENTITY & CONSTRAINT RHS RETENTION
# =====================================================================

class TestMasterRecourseBalanceIdentityRHS:
    """
    Adversarial verification of the Master Recourse Balance Identity:
    - Demonstrates algebraically and empirically that demand non-negativity y_{it}^+ >= 0
      actively retains demand on the right-hand side of recourse constraints.
    - Demonstrates that omitting y_{it}^+ >= 0 renders the linear surrogate objective
      unbounded below (-infty) or operationally invalid.
    """

    def test_algebraic_reconstruction_of_master_identity(self):
        """
        Verify the step-by-step telescoping algebra of the Master Recourse Balance Identity:
        I_{i,t} - I_{i,t-1} - q_{it} + y_{it}^- - y_{it}^+ = -d_{i,t}.
        """
        # For an arbitrary period trace with N=1, T=3:
        q = np.array([30.0, 25.0, 20.0])
        d = np.array([15.0, 35.0, 10.0])

        # Simulate exact transitions:
        # t=0: Avail = [q=30, I_prev=10]. Demand = 15.
        # FIFO: consume 10 from I_prev (age 2), consume 5 from fresh (age 1).
        # w_consumed = [5.0, 10.0]. Total w = 15.0.
        # Stockout y_pos = 0.0. Spoilage y_neg = 0.0.
        # End inventory: I[0] = 25.0 (age 1). Total I_0 = 25.0.
        # Check Master Identity at t=0:
        # I_0 - I_init - q_0 + y_neg_0 - y_pos_0 = 25 - 10 - 30 + 0 - 0 = -15 == -d_0!
        diff_0 = 25.0 - 10.0 - 30.0 + 0.0 - 0.0
        assert diff_0 == -15.0 == -d[0]

        # t=1: Avail = [q=25, I_prev=25]. Demand = 35.
        # FIFO: consume 25 from I_prev (age 2), consume 10 from fresh (age 1).
        # Total w = 35.0. Stockout y_pos = 0.0. Spoilage y_neg = 0.0.
        # End inventory: I[1] = 15.0 (age 1). Total I_1 = 15.0.
        # Check Master Identity at t=1:
        # I_1 - I_0 - q_1 + y_neg_1 - y_pos_1 = 15 - 25 - 25 + 0 - 0 = -35 == -d_1!
        diff_1 = 15.0 - 25.0 - 25.0 + 0.0 - 0.0
        assert diff_1 == -35.0 == -d[1]

        # t=2: Avail = [q=20, I_prev=15]. Demand = 10.
        # FIFO: consume 10 from I_prev (age 2, now age 3). 5 units of age 3 unconsumed -> SPOILS!
        # y_neg_2 = 5.0. Fresh q=20 unconsumed -> I_2 = 20.0.
        # Stockout y_pos = 0.0.
        # Check Master Identity at t=2:
        # I_2 - I_1 - q_2 + y_neg_2 - y_pos_2 = 20 - 15 - 20 + 5.0 - 0.0 = -10 == -d_2!
        diff_2 = 20.0 - 15.0 - 20.0 + 5.0 - 0.0
        assert diff_2 == -10.0 == -d[2]

        # Telescoping sum over t=0..2:
        # sum_{t=0}^2 (I_t - I_{t-1}) - sum q_t + sum y_neg_t - sum y_pos_t = - sum d_t
        # (I_2 - I_init) - sum q_t + sum y_neg_t - sum y_pos_t = - sum d_t
        lhs_sum = (20.0 - 10.0) - np.sum(q) + 5.0 - 0.0
        rhs_sum = -np.sum(d)
        assert np.isclose(lhs_sum, rhs_sum), f"Telescoping sum broken: {lhs_sum} != {rhs_sum}"

    def test_omission_of_nonnegativity_produces_unbounded_solver_exploit(self):
        """
        Verify what happens if one assumes demand was 'eliminated' from constraints
        by dropping the explicit demand satisfaction constraint and relaxing y^+ to be unconstrained:
        The linear surrogate objective min c(d)^T w assigns -p_u coefficient to q.
        Without y^+ >= 0, delivering infinite q earns infinite negative penalty (-p_u * q -> -infty),
        causing complete optimization breakdown.
        """
        # Objective: min -p_u * q + h * I + (p_s + p_u) * y_neg
        # If delivery bounds are relaxed or large (e.g. up to warehouse capacity):
        pu = 20.0

        # Case 1: Unconstrained LP without demand satisfaction constraint on RHS
        # Variable: q in [0, 1000]
        # Objective coefficient: -pu = -20.0
        # Optimal solution trivially hits upper bound (q = 1000) earning -20,000,
        # despite customer demand being only 5.0!
        d_true = 5.0
        c = np.array([-pu])
        A_ub = np.array([[1.0]])
        b_ub = np.array([1000.0])

        res_unconstrained = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)])
        assert res_unconstrained.success is True
        assert np.isclose(res_unconstrained.x[0], 1000.0)
        assert np.isclose(res_unconstrained.fun, -20000.0), (
            "Without demand constraint, solver exploits -p_u linear credit to deliver infinite stock!"
        )

        # Case 2: When demand non-negativity constraint y^+ >= 0 is retained:
        # y^+ = d - q + I + y_neg >= 0  <=>  q - I - y_neg <= d
        # For d = 5.0, q is bounded by d (plus carryover/spoilage):
        A_demand_constraint = np.array([[1.0]])
        b_demand_constraint = np.array([d_true])

        res_constrained = linprog(c=c, A_ub=A_demand_constraint, b_ub=b_demand_constraint, bounds=[(0, None)])
        assert res_constrained.success is True
        assert np.isclose(res_constrained.x[0], 5.0), (
            "Retaining y^+ >= 0 bounds delivery by customer demand d_true!"
        )
        assert np.isclose(res_constrained.fun, -100.0)

        # Conclusion: Customer demand d_true MUST remain on the RHS of the constraint!
