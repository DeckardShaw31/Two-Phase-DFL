"""
Adversarial Stress Test Suite for Milestone 1 Mathematical Rigor and Baselines:
1. Two-Stage Complete Recourse (Theorem 1)
   - First-stage set W_1 demand independence, non-emptiness, compactness
   - Extreme demand bounds: zero demand (zero stockout, finite spoilage)
   - Extreme demand bounds: massive surge demand d >> K*Q (stockout slack y^+ absorbs demand with finite penalty)
   - Theoretical recourse bound verification: Q(q, d) <= p_u * d + (h*tau + p_s)*q
2. Critical Fractiles Baselines in src/models/baselines.py
   - LightGBMQuantilePTO under myopic (alpha=0.800) and tau-amortized (beta=0.8824) fractiles
   - SafetyStockPTO under myopic (alpha=0.800) and tau-amortized (beta=0.8824) fractiles
   - Non-negativity, finiteness, monotonicity, multidimensional input shape handling
   - Downstream capacity binding / externality under inflated fractiles
"""

import numpy as np
import pytest

from src.models.baselines import LightGBMQuantilePTO, SafetyStockPTO
from src.optimization.pirp_model import PIRPModel, solve_pirp


@pytest.fixture
def standard_setup():
    N, T = 6, 4
    np.random.seed(123)
    coords = np.random.uniform(0, 100, size=(N + 1, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return N, T, dist


# =============================================================================
# PART 1: TWO-STAGE COMPLETE RECOURSE & COMPACTNESS (THEOREM 1)
# =============================================================================

class TestTwoStageCompleteRecourseAdversarial:
    """Adversarial stress-testing of Theorem 1 complete recourse and demand independence."""


    def test_first_stage_w1_demand_independence(self, standard_setup):
        """
        Verify that W_1 constraints (flow, capacity, MTZ, visit linking)
        have exactly zero dependence on customer demand realization d.
        """
        N, T, dist = standard_setup
        model = PIRPModel(
            num_stores=N,
            horizon_T=T,
            fleet_size_K=2,
            vehicle_capacity_Q=100.0,
            store_capacity_Ci=70.0,
            distance_matrix=dist,
            shelf_life_tau=3,
        )

        # Inspect static constraints
        # Static eq and static ub constraints must be defined prior to demand
        assert model.static_eq_count > 0
        assert model.static_ub_count > 0

        # Change demand from zeros to huge numbers and verify that static A, b_ub,
        # and first-stage constraint bounds remain strictly bitwise identical.
        demand_zero = np.zeros((N, T))
        demand_huge = np.full((N, T), 10000.0)

        A1, bl1, bu1, eq1 = model.build_system(demand_zero)
        A2, bl2, bu2, eq2 = model.build_system(demand_huge)

        # Constraint matrices A must be identical
        assert (A1 != A2).nnz == 0
        # Total equality constraint count must be identical
        assert eq1 == eq2
        # Static equality bounds (indices 0..static_eq_count) must be identical
        assert np.array_equal(bl1[:model.static_eq_count], bl2[:model.static_eq_count])
        # Inequality bounds (indices eq1..) must be identical
        assert np.array_equal(bu1[eq1:], bu2[eq2:])

        # The ONLY modified values between demand_zero and demand_huge are the RHS of demand rows
        assert not np.array_equal(bl1[model.static_eq_count:eq1], bl2[model.static_eq_count:eq2])
        assert np.all(bl1[model.static_eq_count:eq1] == 0.0)
        assert np.all(bl2[model.static_eq_count:eq2] == 10000.0)

    def test_null_dispatch_membership_in_w1(self, standard_setup):
        """
        Verify Part 1 of Theorem 1: The candidate null dispatch z1^(0) = (x=0, u=1, q=0)
        is unconditionally feasible in W_1.
        """
        N, T, dist = standard_setup
        model = PIRPModel(
            num_stores=N,
            horizon_T=T,
            fleet_size_K=2,
            vehicle_capacity_Q=100.0,
            store_capacity_Ci=70.0,
            distance_matrix=dist,
        )

        z0 = np.zeros(model.total_vars, dtype=np.float64)
        for u_idx in model.u_indices.values():
            z0[u_idx] = 1.0  # u_it^k = 1

        # Check bounds
        assert np.all(z0 >= model.lb - 1e-9)
        assert np.all(z0 <= model.ub + 1e-9)

        # Check static inequalities: A_ub * z0 <= b_ub
        slack_ub = model._b_ub - model._A_ub.dot(z0)
        assert np.all(slack_ub >= -1e-9), f"Violated inequalities: {np.where(slack_ub < -1e-9)[0]}"

        # Check static equalities: A_eq[:static_eq_count] * z0 == b_eq_static
        eq_diff = model._b_eq_template[:model.static_eq_count] - model._A_eq[:model.static_eq_count, :].dot(z0)
        assert np.all(np.abs(eq_diff) <= 1e-9), f"Violated equalities: {np.where(np.abs(eq_diff) > 1e-9)[0]}"

    def test_zero_demand_extreme_recourse(self, standard_setup):
        """
        Verify extreme lower bound d = 0:
        1. Stockout cost is strictly 0.0 (zero unmet demand).
        2. Deliveries, if dispatched, spoil at age tau with strictly finite penalty p_s.
        3. Solver succeeds and returns finite objective.
        """
        N, T, dist = standard_setup
        zero_demand = np.zeros((N, T))

        # Solve continuous LP
        sol_lp = solve_pirp(
            demand=zero_demand,
            distance_matrix=dist,
            continuous_relaxation=True,
            shelf_life_tau=3,
            stockout_penalty_pu=20.0,
            spoilage_penalty_ps=5.0,
        )
        assert sol_lp.success is True
        assert sol_lp.stockout_cost == 0.0
        assert np.all(sol_lp.y_pos == 0.0)
        assert sol_lp.total_cost >= 0.0
        assert np.isfinite(sol_lp.total_cost)

        # Solve discrete MILP
        sol_milp = solve_pirp(
            demand=zero_demand,
            distance_matrix=dist,
            continuous_relaxation=False,
            shelf_life_tau=3,
            stockout_penalty_pu=20.0,
            spoilage_penalty_ps=5.0,
        )
        assert sol_milp.success is True
        assert sol_milp.stockout_cost == 0.0
        assert np.all(sol_milp.y_pos == 0.0)
        assert np.isfinite(sol_milp.total_cost)

    def test_massive_surge_demand_extreme_recourse(self, standard_setup):
        """
        Verify extreme upper bound d >> K*Q:
        Fleet capacity K*Q = 200, demand = 20,000 per store-period (aggregate 480,000 units).
        1. Stockout slack y^+ absorbs all unmet demand.
        2. Stockout penalty p_u is strictly finite and non-negative.
        3. Spoilage is strictly 0.0 (all delivered stock is consumed immediately).
        4. Solver succeeds without numerical overflow or infeasibility.
        """
        N, T, dist = standard_setup
        surge_demand = np.full((N, T), 20000.0)

        # Continuous LP
        sol_lp = solve_pirp(
            demand=surge_demand,
            distance_matrix=dist,
            continuous_relaxation=True,
            shelf_life_tau=3,
            vehicle_capacity_Q=100.0,
            stockout_penalty_pu=20.0,
            spoilage_penalty_ps=5.0,
        )
        assert sol_lp.success is True
        assert sol_lp.stockout_cost > 0.0
        assert np.all(sol_lp.y_pos > 0.0)
        # All available stock consumed -> zero spoilage
        assert np.all(sol_lp.y_neg <= 1e-6)
        assert np.isclose(sol_lp.spoilage_cost, 0.0, atol=1e-5)
        assert np.isfinite(sol_lp.total_cost)

        # Discrete MILP
        sol_milp = solve_pirp(
            demand=surge_demand,
            distance_matrix=dist,
            continuous_relaxation=False,
            shelf_life_tau=3,
            vehicle_capacity_Q=100.0,
            stockout_penalty_pu=20.0,
            spoilage_penalty_ps=5.0,
        )
        assert sol_milp.success is True
        assert sol_milp.stockout_cost > 0.0
        assert np.all(sol_milp.y_pos > 0.0)
        assert np.all(sol_milp.y_neg <= 1e-6)
        assert np.isclose(sol_milp.spoilage_cost, 0.0, atol=1e-5)
        assert np.isfinite(sol_milp.total_cost)

    def test_recourse_cost_upper_bound_theorem1_part3(self, standard_setup):
        """
        Verify Theorem 1 Part 3:
        Q(q, d) <= sum_{t,i} [ p_u * d_{it} + (h_{tau-1}*tau + p_s) * q_{it} ]
        across 20 randomized instances.
        """
        N, T, dist = standard_setup
        rng = np.random.default_rng(42)

        for _ in range(20):
            tau = rng.choice([2, 3, 5])
            h = rng.uniform(0.5, 2.0)
            pu = rng.uniform(10.0, 50.0) * h
            ps = 5.0 * h
            d = rng.exponential(scale=25.0, size=(N, T))

            sol = solve_pirp(
                demand=d,
                distance_matrix=dist,
                continuous_relaxation=True,
                shelf_life_tau=tau,
                stockout_penalty_pu=pu,
                spoilage_penalty_ps=ps,
            )
            assert sol.success is True

            # Recourse cost = holding + spoilage + stockout
            recourse_cost = sol.holding_cost + sol.spoilage_cost + sol.stockout_cost
            assert recourse_cost >= 0.0

            # Theorem 1 theoretical upper bound
            h_tau_1 = h  # baseline holding
            upper_bound = np.sum(pu * d) + (h_tau_1 * tau + ps) * np.sum(sol.q_total)
            assert recourse_cost <= upper_bound + 1e-4


# =============================================================================
# PART 2: CRITICAL FRACTILES & BASELINE REALISM
# =============================================================================

class TestCriticalFractilesBaselinesAdversarial:
    """Adversarial stress-testing of LightGBMQuantilePTO and SafetyStockPTO."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(2026)
        n_samples = 150
        n_features = 14
        X = rng.normal(size=(n_samples, n_features))
        # Non-negative skewed demand target
        y = np.maximum(0.0, 15.0 + 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.exponential(scale=5.0, size=n_samples))
        return X, y

    @pytest.mark.parametrize("tau", [2, 3, 5])
    @pytest.mark.parametrize("pu_ratio", [10.0, 20.0, 50.0])
    def test_fractile_formulas_exact_reconciliation(self, tau, pu_ratio):
        """
        Verify exact theoretical fractile values for both myopic and tau-amortized policies:
        alpha_myopic = pu / (pu + ps)
        beta_tau* = pu / (pu + h + ps / tau)
        """
        h = 1.0
        ps = 5.0 * h
        pu = pu_ratio * h

        expected_myopic = pu / (pu + ps)
        expected_amortized = pu / (pu + h + ps / tau)

        # LightGBMQuantilePTO
        m_myopic = LightGBMQuantilePTO(
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            holding_cost_h=h,
            shelf_life_tau=tau,
            fractile_type="myopic",
        )
        assert pytest.approx(m_myopic.critical_fractile, 1e-6) == expected_myopic
        assert pytest.approx(m_myopic.model.alpha, 1e-6) == expected_myopic

        m_amortized = LightGBMQuantilePTO(
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            holding_cost_h=h,
            shelf_life_tau=tau,
            fractile_type="tau_amortized",
        )
        assert pytest.approx(m_amortized.critical_fractile, 1e-6) == expected_amortized
        assert pytest.approx(m_amortized.model.alpha, 1e-6) == expected_amortized

        # SafetyStockPTO
        ss_myopic = SafetyStockPTO(
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            holding_cost_h=h,
            shelf_life_tau=tau,
            fractile_type="myopic",
        )
        assert pytest.approx(ss_myopic.critical_fractile, 1e-6) == expected_myopic

        ss_amortized = SafetyStockPTO(
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            holding_cost_h=h,
            shelf_life_tau=tau,
            fractile_type="tau_amortized",
        )
        assert pytest.approx(ss_amortized.critical_fractile, 1e-6) == expected_amortized

        # In all cases, beta_tau* must strictly exceed alpha_myopic because
        # holding cost h + ps/tau < ps for all tau >= 2!
        # ps / tau <= ps / 2 = 2.5h, so h + ps/tau <= 3.5h < 5.0h = ps!
        assert expected_amortized > expected_myopic

    def test_baseline_predictions_non_negative_and_finite(self, synthetic_data):
        """
        Verify that both models output strictly non-negative and finite predictions
        across both myopic and tau-amortized fractiles.
        """
        X, y = synthetic_data
        X_train, y_train = X[:100], y[:100]
        X_test = X[100:]

        for fractile in ["myopic", "tau_amortized"]:
            # 1. LightGBM Quantile
            lgb_q = LightGBMQuantilePTO(
                stockout_penalty_pu=20.0,
                spoilage_penalty_ps=5.0,
                shelf_life_tau=3,
                fractile_type=fractile,
                n_estimators=30,
            ).fit(X_train, y_train)
            preds_lgb = lgb_q.predict(X_test)
            assert preds_lgb.shape == (len(X_test),)
            assert np.all(preds_lgb >= 0.0), f"Negative predictions found in LightGBM ({fractile})"
            assert np.all(np.isfinite(preds_lgb)), f"Non-finite predictions in LightGBM ({fractile})"

            # 2. Safety Stock
            ss = SafetyStockPTO(
                stockout_penalty_pu=20.0,
                spoilage_penalty_ps=5.0,
                shelf_life_tau=3,
                fractile_type=fractile,
            ).fit(X_train, y_train)
            preds_ss = ss.predict(X_test)
            assert preds_ss.shape == (len(X_test),)
            assert np.all(preds_ss >= 0.0), f"Negative predictions found in SafetyStock ({fractile})"
            assert np.all(np.isfinite(preds_ss)), f"Non-finite predictions in SafetyStock ({fractile})"

    def test_quantile_monotonicity_in_predictions_and_buffer(self, synthetic_data):
        """
        Since beta_tau* = 0.8824 > alpha_myopic = 0.8000:
        1. SafetyStock safety buffer z * sigma must be strictly greater under beta_tau*.
        2. Average predictions must be higher under beta_tau* than under alpha_myopic.
        """
        X, y = synthetic_data
        X_train, y_train = X[:100], y[:100]
        X_test = X[100:]

        # SafetyStock comparison
        ss_myopic = SafetyStockPTO(fractile_type="myopic", shelf_life_tau=3).fit(X_train, y_train)
        ss_amort = SafetyStockPTO(fractile_type="tau_amortized", shelf_life_tau=3).fit(X_train, y_train)

        assert ss_amort.safety_buffer > ss_myopic.safety_buffer
        preds_ss_myopic = ss_myopic.predict(X_test)
        preds_ss_amort = ss_amort.predict(X_test)
        assert np.all(preds_ss_amort >= preds_ss_myopic)

        # LightGBM Quantile comparison
        lgb_myopic = LightGBMQuantilePTO(fractile_type="myopic", shelf_life_tau=3, n_estimators=50).fit(X_train, y_train)
        lgb_amort = LightGBMQuantilePTO(fractile_type="tau_amortized", shelf_life_tau=3, n_estimators=50).fit(X_train, y_train)

        preds_lgb_myopic = lgb_myopic.predict(X_test)
        preds_lgb_amort = lgb_amort.predict(X_test)
        # On average, higher quantile pinball regression produces higher predictions
        assert np.mean(preds_lgb_amort) > np.mean(preds_lgb_myopic)

    def test_multidimensional_input_shape_preservation(self, synthetic_data):
        """
        Test that both models correctly preserve multi-dimensional input shapes:
        - 3D instance: (N, T, F) -> (N, T)
        - 2D flat: (M, F) -> (M,)
        """
        X, y = synthetic_data
        X_train, y_train = X[:100], y[:100]

        N, T, F = 10, 5, 14
        X_3d = np.random.randn(N, T, F)

        lgb_q = LightGBMQuantilePTO(shelf_life_tau=3, n_estimators=10).fit(X_train, y_train)
        preds_3d_lgb = lgb_q.predict(X_3d)
        assert preds_3d_lgb.shape == (N, T)

        ss = SafetyStockPTO(shelf_life_tau=3).fit(X_train, y_train)
        preds_3d_ss = ss.predict(X_3d)
        assert preds_3d_ss.shape == (N, T)

    def test_downstream_capacity_binding_paradox(self, standard_setup):
        """
        Verify Theorem 2 (Spatial-Temporal Capacity Externality):
        When uncoordinated baselines inflate forecasts to beta_tau* = 0.8824,
        nominal demand exceeds fleet capacity K*Q, inducing vehicle capacity
        binding and store rationing.
        """
        N, T, dist = standard_setup
        K, Q = 2, 70.0
        fleet_capacity = K * Q  # 140.0

        # For N=6 stores:
        # Myopic forecast 20/store -> sum = 120 <= 140 (not binding).
        # Amortized forecast 26/store -> sum = 156 > 140 (severely binding).
        d_myopic = np.full((N, T), 20.0)
        d_amortized = np.full((N, T), 26.0)

        assert np.sum(d_myopic, axis=0)[0] <= fleet_capacity
        assert np.sum(d_amortized, axis=0)[0] > fleet_capacity

        sol_myopic = solve_pirp(d_myopic, distance_matrix=dist, vehicle_capacity_Q=Q, store_capacity_Ci=70.0)
        sol_amort = solve_pirp(d_amortized, distance_matrix=dist, vehicle_capacity_Q=Q, store_capacity_Ci=70.0)

        assert sol_myopic.success is True
        assert sol_amort.success is True

        # Total deliveries cannot exceed fleet capacity in either case
        for t in range(T):
            assert np.sum(sol_myopic.q_total[:, t]) <= fleet_capacity + 1e-4
            assert np.sum(sol_amort.q_total[:, t]) <= fleet_capacity + 1e-4

        # Under the amortized forecast, demand (156) exceeds fleet capacity (140).
        # Due to vehicle capacity Q=70 and discrete tour partitioning,
        # total delivered is strictly capped below demand, rationing stores and inducing stockout.
        for t in range(T):
            assert np.sum(sol_amort.q_total[:, t]) < np.sum(d_amortized[:, t])
            # At least one store experiences unmet demand (rationing)
            assert np.any(sol_amort.y_pos[:, t] > 0.0)
            # Total delivered is within vehicle fleet capacity
            assert np.sum(sol_amort.q_total[:, t]) <= fleet_capacity + 1e-4

        assert sol_amort.stockout_cost > 0.0


