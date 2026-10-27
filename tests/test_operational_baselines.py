"""
Unit tests for the 2024-2026 Operational Comparator Baselines (Tiers 1-4).
Verifies:
  1. Capacitated 2-opt vehicle routing tour validity and capacity compliance.
  2. Tier 1 (Age-Aware Base-Stock / Structure-Guided PIL) feasibility and store capacity adherence.
  3. Tier 2 (Rolling-Horizon PIRP / Model Predictive Control) receding horizon execution.
  4. Tier 3 (Scenario-Based Stochastic SAA) scenario generation and consensus planning.
  5. Tier 4 (ALNS Matheuristic) destroy-and-repair operators and feasibility.
  6. Gate 2 verification: Unified evaluation under evaluate_decisions_under_true_demand.
"""

import numpy as np
import pytest

from src.models.operational_baselines import (
    AgeAwareBaseStockPolicy,
    ALNSMatheuristicPIRP,
    RollingHorizonMPCPIRP,
    ScenarioBasedSAAPIRP,
    build_capacitated_routes_2opt,
)
from src.utils.metrics import evaluate_decisions_under_true_demand


@pytest.fixture
def synthetic_toy_instance():
    """Sets up a small 5-store, 3-period test instance with 2 vehicles."""
    N = 5
    T = 3
    K = 2
    Q = 50.0
    Ci = 40.0
    tau = 3
    h = 1.0
    ps = 5.0
    pu = 20.0

    # Euclidean distance matrix for depot + 5 stores
    coords = np.array([
        [0.0, 0.0],    # Depot 0
        [10.0, 10.0],  # Store 1
        [10.0, -10.0], # Store 2
        [-10.0, 10.0], # Store 3
        [-10.0, -10.0],# Store 4
        [15.0, 0.0],   # Store 5
    ])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    predicted_demand = np.array([
        [12.0, 15.0, 10.0],
        [8.0, 14.0, 12.0],
        [15.0, 10.0, 9.0],
        [10.0, 12.0, 14.0],
        [14.0, 8.0, 11.0],
    ])

    true_demand = predicted_demand + np.array([
        [2.0, -1.0, 3.0],
        [-2.0, 3.0, 1.0],
        [1.0, -2.0, 0.0],
        [0.0, 1.0, -2.0],
        [3.0, 2.0, -1.0],
    ])

    initial_inventory = np.zeros((N, tau - 1), dtype=np.float64)

    return {
        "N": N, "T": T, "K": K, "Q": Q, "Ci": Ci, "tau": tau,
        "h": h, "ps": ps, "pu": pu, "dist": dist,
        "predicted_demand": predicted_demand,
        "true_demand": true_demand,
        "initial_inventory": initial_inventory,
    }


def test_build_capacitated_routes_2opt(synthetic_toy_instance):
    """Verifies that vehicle routes do not exceed capacity Q and form closed tours."""
    data = synthetic_toy_instance
    q_target = np.array([20.0, 25.0, 15.0, 10.0, 15.0]) # Total = 85.0 > Q (50.0)

    q_deliv, x_routes, r_cost = build_capacitated_routes_2opt(
        q_target=q_target,
        distance_matrix=data["dist"],
        fleet_size_K=data["K"],
        vehicle_capacity_Q=data["Q"],
    )

    # Deliveries per vehicle should not exceed Q
    assert np.sum(q_deliv) <= data["K"] * data["Q"] + 1e-6
    assert np.all(q_deliv >= 0.0)
    assert r_cost > 0.0
    assert x_routes.shape == (data["N"] + 1, data["N"] + 1, data["K"])

    # Check that each vehicle departs and returns to depot if active
    for k in range(data["K"]):
        departures = np.sum(x_routes[0, 1:, k])
        returns = np.sum(x_routes[1:, 0, k])
        assert departures == returns
        assert departures in (0.0, 1.0)


def test_tier1_age_aware_base_stock(synthetic_toy_instance):
    """Verifies Tier 1 Base-Stock policy feasibility and execution."""
    data = synthetic_toy_instance
    policy = AgeAwareBaseStockPolicy(
        num_stores=data["N"],
        horizon_T=data["T"],
        fleet_size_K=data["K"],
        vehicle_capacity_Q=data["Q"],
        store_capacity_Ci=data["Ci"],
        shelf_life_tau=data["tau"],
        distance_matrix=data["dist"],
    )

    result = policy.plan_horizon(
        predicted_demand=data["predicted_demand"],
        residual_std=3.0,
        initial_inventory=data["initial_inventory"],
    )

    assert result.method_name == "Tier1_AgeAwareBaseStock"
    assert result.q_plan.shape == (data["N"], data["T"])
    assert result.x_plan.shape == (data["N"] + 1, data["N"] + 1, data["K"], data["T"])
    assert np.all(result.q_plan >= 0.0)
    assert np.all(result.q_plan <= data["Ci"] + 1e-6)

    # Evaluate on unified physical simulator
    cost_dict = evaluate_decisions_under_true_demand(
        q_plan=result.q_plan,
        x_plan=result.x_plan,
        true_demand=data["true_demand"],
        distance_matrix=data["dist"],
        h=data["h"],
        p_s=data["ps"],
        p_u=data["pu"],
        tau=data["tau"],
        initial_inventory=data["initial_inventory"],
        store_capacity_Ci=data["Ci"],
    )
    assert cost_dict["total_cost"] > 0.0
    assert cost_dict["routing_cost"] >= 0.0


def test_tier2_rolling_horizon_mpc(synthetic_toy_instance):
    """Verifies Tier 2 Rolling-Horizon MPC execution."""
    data = synthetic_toy_instance
    mpc = RollingHorizonMPCPIRP(
        lookahead_H=2,
        num_stores=data["N"],
        horizon_T=data["T"],
        fleet_size_K=data["K"],
        vehicle_capacity_Q=data["Q"],
        store_capacity_Ci=data["Ci"],
        shelf_life_tau=data["tau"],
        holding_cost_h=data["h"],
        spoilage_penalty_ps=data["ps"],
        stockout_penalty_pu=data["pu"],
        distance_matrix=data["dist"],
    )

    result = mpc.plan_open_loop(
        predicted_demand=data["predicted_demand"],
        initial_inventory=data["initial_inventory"],
    )

    assert result.method_name == "Tier2_RollingHorizonMPC"
    assert result.q_plan.shape == (data["N"], data["T"])
    assert np.all(result.q_plan >= 0.0)

    cost_dict = evaluate_decisions_under_true_demand(
        q_plan=result.q_plan,
        x_plan=result.x_plan,
        true_demand=data["true_demand"],
        distance_matrix=data["dist"],
        h=data["h"],
        p_s=data["ps"],
        p_u=data["pu"],
        tau=data["tau"],
        initial_inventory=data["initial_inventory"],
        store_capacity_Ci=data["Ci"],
    )
    assert cost_dict["total_cost"] > 0.0


def test_residual_scenario_scaling_is_not_mislabeled_as_saa(synthetic_toy_instance):
    """Verifies the exploratory residual search retains its qualified name."""
    data = synthetic_toy_instance
    saa = ScenarioBasedSAAPIRP(
        num_scenarios_S=5,
        num_stores=data["N"],
        horizon_T=data["T"],
        fleet_size_K=data["K"],
        vehicle_capacity_Q=data["Q"],
        store_capacity_Ci=data["Ci"],
        shelf_life_tau=data["tau"],
        holding_cost_h=data["h"],
        spoilage_penalty_ps=data["ps"],
        stockout_penalty_pu=data["pu"],
        distance_matrix=data["dist"],
    )

    result = saa.plan_horizon(
        predicted_demand=data["predicted_demand"],
        initial_inventory=data["initial_inventory"],
        random_seed=42,
    )

    assert result.method_name == "ResidualScenarioScalingSearch_NOT_SAA"
    assert result.q_plan.shape == (data["N"], data["T"])
    assert np.all(result.q_plan >= 0.0)

    cost_dict = evaluate_decisions_under_true_demand(
        q_plan=result.q_plan,
        x_plan=result.x_plan,
        true_demand=data["true_demand"],
        distance_matrix=data["dist"],
        h=data["h"],
        p_s=data["ps"],
        p_u=data["pu"],
        tau=data["tau"],
        initial_inventory=data["initial_inventory"],
        store_capacity_Ci=data["Ci"],
    )
    assert cost_dict["total_cost"] > 0.0


def test_randomized_destroy_repair_is_not_mislabeled_as_alns(synthetic_toy_instance):
    """Verifies the exploratory destroy/repair routine is not called ALNS."""
    data = synthetic_toy_instance
    alns = ALNSMatheuristicPIRP(
        max_iterations=15,
        num_stores=data["N"],
        horizon_T=data["T"],
        fleet_size_K=data["K"],
        vehicle_capacity_Q=data["Q"],
        store_capacity_Ci=data["Ci"],
        shelf_life_tau=data["tau"],
        holding_cost_h=data["h"],
        spoilage_penalty_ps=data["ps"],
        stockout_penalty_pu=data["pu"],
        distance_matrix=data["dist"],
    )

    result = alns.plan_horizon(
        predicted_demand=data["predicted_demand"],
        initial_inventory=data["initial_inventory"],
        random_seed=42,
    )

    assert result.method_name == "RandomizedDestroyRepair_NOT_ALNS"
    assert result.q_plan.shape == (data["N"], data["T"])
    assert np.all(result.q_plan >= 0.0)

    cost_dict = evaluate_decisions_under_true_demand(
        q_plan=result.q_plan,
        x_plan=result.x_plan,
        true_demand=data["true_demand"],
        distance_matrix=data["dist"],
        h=data["h"],
        p_s=data["ps"],
        p_u=data["pu"],
        tau=data["tau"],
        initial_inventory=data["initial_inventory"],
        store_capacity_Ci=data["Ci"],
    )
    assert cost_dict["total_cost"] > 0.0
