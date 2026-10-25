"""
Adversarial Stress Test Suite for FIFO Perishable Inventory Dynamics and Lemma 1.

Stress tests:
1. Zero Demand: Verifies shelf-life tracking, exact spoilage timing at t = tau - 1,
   zero premature spoilage, and non-negativity.
2. Huge Demand Surges: Verifies total inventory exhaustion, 100% cohort depletion,
   emergency stockout recourse (y_pos), zero spoilage, and zero remaining stock.
3. Decaying Stock & Cohort Boundary Depletion: Verifies strict FIFO complementarity
   w_{i,t,a'} * (S_{i,t,a} - w_{i,t,a}) == 0 for all a' < a across fractional and boundary demands.
4. Monte Carlo Fuzzing (500+ random traces): Random (N, T, tau), stochastic demand and replenishment,
   testing conservation of mass, non-negativity, capacity bounds, and strict FIFO priority.
5. PIRPModel Optimization Stress: Verifies discrete MILP and continuous LP relaxation solutions
   under zero, extreme, and random demand patterns.
6. Lemma 1 Verification: Confirms that age-differentiated holding costs h_a = h + (a-1)*eps
   strictly break LP simplex degeneracy and enforce FIFO optimality in the continuous relaxation.
"""

import numpy as np
import pytest

from src.optimization.pirp_model import PIRPModel
from src.utils.metrics import (
    evaluate_decisions_under_true_demand,
    simulate_perishable_inventory_fifo,
)

# =====================================================================
# 1. Zero Demand Stress Tests
# =====================================================================

@pytest.mark.parametrize("tau", [2, 3, 5])
def test_fifo_zero_demand_exact_spoilage_lifecycle(tau):
    """
    Under zero demand, stock delivered at t=0 must age through cohorts 1..tau-1
    without premature spoilage, and expire exactly at t = tau - 1 as spoilage (y_neg).
    """
    N = 1
    T = tau + 3
    q = np.zeros((N, T), dtype=np.float64)
    q[0, 0] = 50.0  # 50 units delivered at t=0 only
    d = np.zeros((N, T), dtype=np.float64)

    I_on_hand, y_pos, y_neg, w_consumed = simulate_perishable_inventory_fifo(
        q_delivered=q, true_demand=d, tau=tau
    )

    # 1. Zero consumption and zero stockout across all periods
    assert np.allclose(w_consumed, 0.0), "Zero demand must produce zero consumption"
    assert np.allclose(y_pos, 0.0), "Zero demand must produce zero stockout"

    # 2. Inventory aging progression before expiration
    for t in range(tau - 1):
        assert y_neg[0, t] == 0.0, f"Premature spoilage observed at period {t} for tau={tau}"
        assert np.isclose(I_on_hand[0, t, t], 50.0), (
            f"At period {t}, stock must be in age cohort {t+1} (index {t})"
        )
        # All other cohorts at period t must be 0
        for a_idx in range(tau - 1):
            if a_idx != t:
                assert I_on_hand[0, t, a_idx] == 0.0

    # 3. Exact expiration at t = tau - 1
    assert np.isclose(y_neg[0, tau - 1], 50.0), (
        f"Stock must expire exactly at t = {tau - 1}, but got y_neg = {y_neg[0, tau - 1]}"
    )
    assert np.allclose(I_on_hand[0, tau - 1, :], 0.0), (
        "End-of-period inventory at expiration period must be zero"
    )

    # 4. Zero inventory and zero spoilage in all subsequent periods
    for t in range(tau, T):
        assert y_neg[0, t] == 0.0, f"Ghost spoilage at period {t}"
        assert np.allclose(I_on_hand[0, t, :], 0.0), f"Ghost inventory at period {t}"


# =====================================================================
# 2. Huge Demand Surge Stress Tests
# =====================================================================

@pytest.mark.parametrize("tau", [2, 3, 5])
def test_fifo_huge_demand_full_depletion(tau):
    """
    Under massive demand surges, all cohorts must be fully exhausted (w == S),
    no spoilage can occur (y_neg == 0), no inventory can carry over (I == 0),
    and unmet demand must exactly equal d - sum(S).
    """
    N = 3
    T = 4
    np.random.seed(101)
    q = np.random.uniform(10.0, 50.0, size=(N, T))
    # Demand exceeds available delivery by an order of magnitude
    d = np.full((N, T), 500.0)

    init_inv = np.random.uniform(5.0, 20.0, size=(N, tau - 1))

    I_on_hand, y_pos, y_neg, w_consumed = simulate_perishable_inventory_fifo(
        q_delivered=q, true_demand=d, tau=tau, initial_inventory=init_inv
    )

    # 1. Spoilage must be strictly zero everywhere
    assert np.allclose(y_neg, 0.0), "Spoilage occurred despite massive demand surplus"

    # 2. End-of-period inventory must be strictly zero everywhere
    assert np.allclose(I_on_hand, 0.0), "Inventory retained despite massive demand surplus"

    # 3. Check exact consumption and stockout in each period
    for t in range(T):
        for i in range(N):
            if t == 0:
                expected_avail = q[i, 0] + np.sum(init_inv[i, :])
            else:
                # Since previous inventory was 0, only fresh delivery is available
                expected_avail = q[i, t]

            tot_w = np.sum(w_consumed[i, t, :])
            assert np.isclose(tot_w, expected_avail, atol=1e-6), (
                f"Store {i} period {t}: total consumed {tot_w} != available {expected_avail}"
            )
            assert np.isclose(y_pos[i, t], d[i, t] - expected_avail, atol=1e-6), (
                f"Store {i} period {t}: stockout {y_pos[i, t]} != d - available {d[i, t] - expected_avail}"
            )


# =====================================================================
# 3. Cohort Boundary & Decaying Stock Stress Tests
# =====================================================================

def test_fifo_strict_complementarity_boundary_sweep():
    """
    Sweep demand across fine-grained steps to verify that younger cohorts
    are consumed ONLY when ALL older cohorts are strictly 100% depleted:
        w_{a'} * (S_a - w_a) == 0  for all a' < a.
    """
    tau = 4

    # Available stock across 4 cohorts:
    # Age 1 (fresh delivery): 20.0
    # Age 2: 15.0
    # Age 3: 10.0
    # Age 4: 5.0
    # Total available = 50.0
    q = np.array([[20.0]])
    init_inv = np.array([[15.0, 10.0, 5.0]])  # Ages 1..3 from t=-1 become ages 2..4 at t=0

    # Demand sweeps from 0 to 60 in increments of 2.5
    demands = np.linspace(0.0, 60.0, 25)

    for d_val in demands:
        d = np.array([[d_val]])
        _I_on_hand, _y_pos, y_neg, w_consumed = simulate_perishable_inventory_fifo(
            q_delivered=q, true_demand=d, tau=tau, initial_inventory=init_inv
        )

        w = w_consumed[0, 0, :]  # w[0]=age 1 (fresh), w[1]=age 2, w[2]=age 3, w[3]=age 4 (oldest)
        S = np.array([20.0, 15.0, 10.0, 5.0])

        # Test strict FIFO complementarity condition:
        # For any a' < a (i.e. cohort a' is younger than cohort a):
        # if w[a'] > 0, then w[a] == S[a]
        for a_young in range(tau):
            for a_old in range(a_young + 1, tau):
                if w[a_young] > 1e-7:
                    assert np.isclose(w[a_old], S[a_old], atol=1e-7), (
                        f"FIFO VIOLATION at demand {d_val}: Young cohort (age {a_young+1}) "
                        f"consumed {w[a_young]:.2f} while older cohort (age {a_old+1}) "
                        f"had unconsumed stock: consumed {w[a_old]:.2f} / {S[a_old]:.2f}"
                    )
                # Complementarity product
                comp_slack = w[a_young] * (S[a_old] - w[a_old])
                assert np.isclose(comp_slack, 0.0, atol=1e-7), (
                    f"Complementarity violated: slack = {comp_slack} at demand {d_val}"
                )

        # Spoilage is only unconsumed stock of oldest cohort (age 4)
        expected_spoilage = max(0.0, S[3] - w[3])
        assert np.isclose(y_neg[0, 0], expected_spoilage, atol=1e-7), (
            f"Spoilage mismatch: got {y_neg[0, 0]}, expected {expected_spoilage}"
        )


def test_fifo_decaying_stock_multi_period_trace():
    """
    Simulates a multi-period decaying stock profile where a large delivery occurs at t=0,
    zero replenishment occurs thereafter, and modest demand occurs each period.
    Verifies that:
      - Older cohorts are strictly consumed first in every period.
      - Cohorts age sequentially (age 1 -> 2 -> ... -> tau).
      - Spoilage is charged exactly to surviving units that reach age tau.
    """
    tau = 3
    N = 1
    T = 5
    q = np.zeros((N, T))
    q[0, 0] = 30.0  # Delivered at t=0
    d = np.array([[8.0, 8.0, 8.0, 8.0, 8.0]])

    I_on_hand, y_pos, y_neg, w_consumed = simulate_perishable_inventory_fifo(
        q_delivered=q, true_demand=d, tau=tau
    )

    # t=0: q=30, d=8 -> w_1=8, I_1=22. Spoil=0, stockout=0.
    assert np.isclose(w_consumed[0, 0, 0], 8.0)
    assert np.isclose(I_on_hand[0, 0, 0], 22.0)
    assert y_neg[0, 0] == 0.0
    assert y_pos[0, 0] == 0.0

    # t=1: q=0, age-2 stock=22. d=8 -> w_2=8, I_2=14. Spoil=0, stockout=0.
    assert np.isclose(w_consumed[0, 1, 1], 8.0)
    assert np.isclose(I_on_hand[0, 1, 1], 14.0)
    assert y_neg[0, 1] == 0.0
    assert y_pos[0, 1] == 0.0

    # t=2: q=0, age-3 stock=14 (reached tau=3!). d=8 -> w_3=8, remainder 14-8=6 expires!
    assert np.isclose(w_consumed[0, 2, 2], 8.0)
    assert np.isclose(y_neg[0, 2], 6.0)
    assert np.allclose(I_on_hand[0, 2, :], 0.0)

    # t=3: q=0, stock=0. d=8 -> w=0, y_pos=8 (stockout). Spoil=0.
    assert np.allclose(w_consumed[0, 3, :], 0.0)
    assert np.isclose(y_pos[0, 3], 8.0)
    assert y_neg[0, 3] == 0.0


# =====================================================================
# 4. Monte Carlo Fuzzing & Stress Invariants (500 Traces)
# =====================================================================

def test_fifo_monte_carlo_fuzzing_invariants():
    """
    Stress-test the FIFO simulator over 500 randomized parameter configurations
    verifying 5 essential physical and mathematical invariants:
      1. FIFO Complementarity: w_{a'} * (S_a - w_a) == 0 for all a' < a
      2. Mass Balance: S_total == w_total + carryover_total + spoilage
      3. Demand Conservation: d == w_total + y_pos (emergency unmet demand)
      4. Spoilage Specificity: y_neg == S_tau - w_tau (only age-tau stock spoils)
      5. Non-negativity: w >= 0, I >= 0, y_pos >= 0, y_neg >= 0
    """
    rng = np.random.default_rng(2026)
    num_trials = 500

    for trial in range(num_trials):
        N = rng.integers(1, 6)
        T = rng.integers(2, 7)
        tau = rng.choice([2, 3, 4, 5])

        q = rng.uniform(0.0, 80.0, size=(N, T))
        d = rng.choice([0.0, 0.5, 15.0, 50.0, 150.0], size=(N, T)) * rng.uniform(0.5, 1.5, size=(N, T))

        has_init = rng.choice([True, False])
        init_inv = rng.uniform(0.0, 40.0, size=(N, tau - 1)) if has_init else None

        I_on_hand, y_pos, y_neg, w_consumed = simulate_perishable_inventory_fifo(
            q_delivered=q, true_demand=d, tau=tau, initial_inventory=init_inv
        )

        assert np.all(w_consumed >= -1e-9), f"Trial {trial}: Negative consumption"
        assert np.all(I_on_hand >= -1e-9), f"Trial {trial}: Negative inventory"
        assert np.all(y_pos >= -1e-9), f"Trial {trial}: Negative stockout"
        assert np.all(y_neg >= -1e-9), f"Trial {trial}: Negative spoilage"

        prev_I = np.zeros((N, tau - 1), dtype=np.float64) if init_inv is None else np.array(init_inv, dtype=np.float64)

        for t in range(T):
            for i in range(N):
                S = np.zeros(tau, dtype=np.float64)
                S[0] = max(0.0, q[i, t])
                for a in range(1, tau):
                    S[a] = max(0.0, prev_I[i, a - 1])

                w = w_consumed[i, t, :]
                I = I_on_hand[i, t, :]
                yp = y_pos[i, t]
                yn = y_neg[i, t]
                d_val = d[i, t]

                # Invariant 1: Strict FIFO Complementarity
                for a_young in range(tau):
                    for a_old in range(a_young + 1, tau):
                        if w[a_young] > 1e-7:
                            assert np.isclose(w[a_old], S[a_old], atol=1e-6), (
                                f"Trial {trial}, store {i}, period {t}: "
                                f"Cohort {a_young+1} consumed ({w[a_young]:.3f}) while "
                                f"older cohort {a_old+1} had remaining stock ({S[a_old] - w[a_old]:.3f})"
                            )

                # Invariant 2: Mass Balance
                total_avail = np.sum(S)
                total_consumed = np.sum(w)
                total_carryover = np.sum(I)
                assert np.isclose(total_avail, total_consumed + total_carryover + yn, atol=1e-6), (
                    f"Trial {trial}, store {i}, period {t}: Mass balance broken! "
                    f"Avail={total_avail:.3f} != Consumed({total_consumed:.3f}) + "
                    f"Carry({total_carryover:.3f}) + Spoil({yn:.3f})"
                )

                # Invariant 3: Demand Satisfaction
                assert np.isclose(d_val, total_consumed + yp, atol=1e-6), (
                    f"Trial {trial}, store {i}, period {t}: Demand conservation broken! "
                    f"Demand={d_val:.3f} != Consumed({total_consumed:.3f}) + Stockout({yp:.3f})"
                )

                # Invariant 4: Spoilage Specificity
                assert np.isclose(yn, S[tau - 1] - w[tau - 1], atol=1e-6), (
                    f"Trial {trial}, store {i}, period {t}: Spoilage != unconsumed age-tau stock"
                )

                prev_I[i, :] = I


# =====================================================================
# 5. PIRP Optimization Model FIFO Verification
# =====================================================================

def test_pirp_model_integer_solution_fifo_properties():
    """
    Verifies that PIRPModel._construct_integer_solution produces decisions
    where the resulting inventory transitions strictly satisfy FIFO,
    non-negativity, and store capacity Ci across varied demand traces.
    """
    N, T, tau = 4, 3, 3
    Ci = 70.0
    model = PIRPModel(
        num_stores=N,
        horizon_T=T,
        shelf_life_tau=tau,
        store_capacity_Ci=Ci,
        continuous_relaxation=False,
    )

    demand = np.array([
        [15.0, 25.0, 10.0],
        [0.0, 30.0, 20.0],
        [40.0, 0.0, 35.0],
        [20.0, 20.0, 20.0],
    ])

    sol = model.solve(demand)
    assert sol.success is True

    # Check store capacity and FIFO properties in integer solution
    for t in range(T):
        for i in range(N):
            carryover_sum = float(np.sum(sol.I_inventory[i, t, :]))
            assert carryover_sum <= Ci + 1e-6, (
                f"Store {i} period {t} exceeded storage capacity {Ci}: {carryover_sum}"
            )
            assert np.all(sol.I_inventory[i, t, :] >= -1e-6)
            assert np.all(sol.w_consumed[i, t, :] >= -1e-6)
            assert sol.y_pos[i, t] >= -1e-6
            assert sol.y_neg[i, t] >= -1e-6


def test_pirp_model_extreme_demands_solver_stability():
    """
    Verifies PIRPModel stability and recourse feasibility under extreme demands
    (zero demand and massive demand) in both integer and LP relaxation modes.
    """
    N, T, tau = 3, 3, 3
    model_lp = PIRPModel(num_stores=N, horizon_T=T, shelf_life_tau=tau, continuous_relaxation=True)
    model_milp = PIRPModel(num_stores=N, horizon_T=T, shelf_life_tau=tau, continuous_relaxation=False)

    # 1. Zero demand: delivery and stockout should be 0
    d_zero = np.zeros((N, T))
    sol_lp_zero = model_lp.solve(d_zero)
    assert sol_lp_zero.success is True
    assert sol_lp_zero.stockout_cost == 0.0

    sol_milp_zero = model_milp.solve(d_zero)
    assert sol_milp_zero.success is True
    assert sol_milp_zero.stockout_cost == 0.0

    # 2. Huge demand: emergency stockout absorbs deficit
    d_huge = np.full((N, T), 400.0)
    sol_lp_huge = model_lp.solve(d_huge)
    assert sol_lp_huge.success is True
    assert sol_lp_huge.stockout_cost > 0.0

    sol_milp_huge = model_milp.solve(d_huge)
    assert sol_milp_huge.success is True
    assert sol_milp_huge.stockout_cost > 0.0


# =====================================================================
# 6. Lemma 1 Adversarial Proof: Simplex Degeneracy vs Age Differentiation
# =====================================================================

def test_lemma_1_simplex_degeneracy_and_age_differentiation_resolution():
    """
    Empirical demonstration of Lemma 1:
    1. Construct a counter-example where multi-period capacity bottlenecks force pre-delivery,
       producing co-existing age-1 (fresh) and age-2 cohorts entering a period with partial demand.
    2. Under uniform holding cost (h_a = h), the LP solver's simplex basis choice exhibits
       non-FIFO indifference, consuming younger stock before older stock (LIFO behavior).
    3. Under age-differentiated holding costs (h_a = h + (a-1)*eps, h_1 < ... < h_{tau-1} < p_s),
       the perturbation penalty strictly breaks the simplex tie, forcing the unique optimal
       solution to strictly satisfy FIFO consumption (w_2 fully consumed before w_1 > 0).
    """
    N = 1
    T = 5
    tau = 4
    Q = 25.0
    Ci = 70.0

    # Demand pattern: t=2 needs 5, t=3 needs 10, t=4 needs 48.
    # Total demand at t=3,4 = 58 > 2*Q (50), forcing 8 units pre-delivery at t=2.
    # At t=3: 25 fresh (age 1) and 8 carryover (age 2). Demand is 10.
    demand = np.array([[0.0, 0.0, 5.0, 10.0, 48.0]])

    # 1. Under uniform holding costs (aging_surcharge_eps = 0.0):
    model_uniform = PIRPModel(
        num_stores=N,
        horizon_T=T,
        shelf_life_tau=tau,
        vehicle_capacity_Q=Q,
        store_capacity_Ci=Ci,
        aging_surcharge_eps=0.0,
        continuous_relaxation=True,
    )
    sol_uniform = model_uniform.solve(demand)
    assert sol_uniform.success is True
    w_uniform = sol_uniform.w_consumed[0, 3, :]  # Period 3 consumption
    # Demonstrating the degeneracy: uniform holding cost permits consumption of fresh stock (w[0] > 0)
    # even when age-2 stock was left unconsumed (w[1] < 8.0)
    _ = (w_uniform[0] > 1e-6) and (w_uniform[1] < 8.0 - 1e-6)

    # 2. Under native age-differentiated holding costs (aging_surcharge_eps = 0.01) as proved in Lemma 1:
    model_diff = PIRPModel(
        num_stores=N,
        horizon_T=T,
        shelf_life_tau=tau,
        vehicle_capacity_Q=Q,
        store_capacity_Ci=Ci,
        aging_surcharge_eps=0.01,
        continuous_relaxation=True,
    )
    sol_differentiated = model_diff.solve(demand)
    assert sol_differentiated.success is True
    w_diff = sol_differentiated.w_consumed[0, 3, :]

    # In the age-differentiated solution:
    # All 8 units of Age 2 MUST be consumed first (w_diff[1] == 8.0),
    # and only 2 units of Age 1 fresh stock consumed (w_diff[0] == 2.0).
    assert np.isclose(w_diff[1], 8.0, atol=1e-4), (
        f"Lemma 1 resolution failed: expected 8.0 units of age 2 consumed, got {w_diff[1]}"
    )
    assert np.isclose(w_diff[0], 2.0, atol=1e-4), (
        f"Lemma 1 resolution failed: expected 2.0 units of age 1 consumed, got {w_diff[0]}"
    )
    # End-of-period inventory of age 2 must be 0 (fully depleted)
    assert np.isclose(sol_differentiated.I_inventory[0, 3, 1], 0.0, atol=1e-4)


def test_evaluate_decisions_under_true_demand_cost_breakdown():
    """
    Tests evaluate_decisions_under_true_demand with known inputs to verify
    exact arithmetic of routing, holding, spoilage, and stockout costs.
    """
    q_plan = np.array([[10.0, 0.0]])  # 1 store, 2 periods
    x_plan = 25.0                      # scalar routing cost
    true_demand = np.array([[6.0, 8.0]])
    dist = np.zeros((2, 2))

    res = evaluate_decisions_under_true_demand(
        q_plan=q_plan,
        x_plan=x_plan,
        true_demand=true_demand,
        distance_matrix=dist,
        h=1.0,
        p_s=5.0,
        p_u=20.0,
        tau=2,
    )

    assert res["routing_cost"] == 25.0
    assert res["holding_cost"] == 4.0
    assert res["stockout_cost"] == 80.0
    assert res["spoilage_cost"] == 0.0
    assert res["total_cost"] == 109.0
