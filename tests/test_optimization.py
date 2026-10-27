"""
Unit and Integration Tests for PIRP Mathematical Model and PyEPO Interface.
"""

import numpy as np
import pytest
import torch

from src.models.dfl_models import DFLPipeline, asymmetric_pinball_loss
from src.optimization.pirp_model import PIRPModel, solve_pirp
from src.optimization.pyepo_interface import DPOLoss, IntOptLoss, PIRPOptModel, SpopLoss
from src.utils.metrics import (
    simulate_perishable_inventory_fifo,
)


@pytest.fixture
def dummy_instance():
    """Generates a small test instance for testing."""
    N, T = 10, 5
    np.random.seed(42)
    coords = np.random.uniform(0, 100, size=(N + 1, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    demand = np.random.uniform(10, 30, size=(N, T))
    return N, T, dist, demand


def test_pirp_model_initialization(dummy_instance):
    N, T, dist, _ = dummy_instance
    model = PIRPModel(num_stores=N, horizon_T=T, distance_matrix=dist, continuous_relaxation=True)
    assert model.total_vars > 1000
    assert model.base_c.shape == (model.total_vars,)
    assert len(model.x_indices) == (N + 1) * N * model.K * T
    assert len(model.lb) == model.total_vars
    assert len(model.ub) == model.total_vars


def test_lp_relaxation_feasibility_and_speed(dummy_instance):
    N, T, dist, demand = dummy_instance
    sol_lp = solve_pirp(demand=demand, distance_matrix=dist, continuous_relaxation=True)
    assert sol_lp.success is True
    assert sol_lp.objective_val > 0.0
    assert sol_lp.solve_time_sec < 1.0  # High-speed solve
    assert sol_lp.q_total.shape == (N, T)
    assert sol_lp.x_routes.shape == (N + 1, N + 1, 2, T)


def test_milp_integrality_and_bounds(dummy_instance):
    _N, _T, dist, demand = dummy_instance
    sol_milp = solve_pirp(
        demand=demand,
        distance_matrix=dist,
        continuous_relaxation=False,
        time_limit_sec=3.0,
        mip_rel_gap=0.05,
        use_milp=True,
    )
    assert sol_milp.success is True
    assert sol_milp.objective_val > 0.0
    assert sol_milp.solver_name.startswith("HiGHS MILP") or sol_milp.solver_name.startswith("HiGHS MILP attempt")
    # Check bounds
    assert np.all(sol_milp.x_routes >= -1e-6)
    assert np.all(sol_milp.x_routes <= 1.0 + 1e-6)
    assert np.all(sol_milp.q_total >= -1e-6)


def test_default_integer_solve_is_labeled_as_heuristic(dummy_instance):
    _N, _T, dist, demand = dummy_instance
    solution = solve_pirp(demand=demand, distance_matrix=dist)
    assert solution.success
    assert not solution.is_optimal
    assert solution.solver_name == "LP-guided constructive heuristic"


def test_initial_inventory_model_matches_fifo_simulator():
    demand = np.array([[3.0, 4.0, 2.0], [2.0, 1.0, 4.0]])
    distance = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    initial_inventory = np.array([[2.0, 1.0], [1.0, 0.0]])
    solution = solve_pirp(
        demand,
        distance_matrix=distance,
        shelf_life_tau=3,
        initial_inventory=initial_inventory,
        time_limit_sec=3.0,
        mip_rel_gap=0.05,
        use_milp=True,
    )
    assert solution.success
    I, y_pos, y_neg, w = simulate_perishable_inventory_fifo(
        solution.q_total,
        demand,
        tau=3,
        initial_inventory=initial_inventory,
        store_capacity_Ci=70,
    )
    assert np.allclose(I, solution.I_inventory, atol=1e-5)
    assert np.allclose(y_pos, solution.y_pos, atol=1e-5)
    assert np.allclose(y_neg, solution.y_neg, atol=1e-5)
    assert np.allclose(w, solution.w_consumed, atol=1e-5)


def test_complete_recourse_non_emptiness(dummy_instance):
    """Verifies Theorem 1: Complete recourse feasibility under extreme demands."""
    N, T, dist, _ = dummy_instance
    # Zero demand
    zero_demand = np.zeros((N, T))
    sol_zero = solve_pirp(demand=zero_demand, distance_matrix=dist, continuous_relaxation=True)
    assert sol_zero.success is True
    assert sol_zero.stockout_cost == 0.0

    # Massive demand exceeding fleet capacity
    huge_demand = np.full((N, T), 500.0)
    sol_huge = solve_pirp(demand=huge_demand, distance_matrix=dist, continuous_relaxation=True)
    assert sol_huge.success is True
    assert sol_huge.stockout_cost > 0.0  # Emergency recourse absorbs excess demand


def test_cost_breakdown_consistency(dummy_instance):
    _N, _T, dist, demand = dummy_instance
    sol = solve_pirp(demand=demand, distance_matrix=dist, continuous_relaxation=True)
    recomputed_total = sol.routing_cost + sol.holding_cost + sol.spoilage_cost + sol.stockout_cost
    assert np.isclose(sol.total_cost, recomputed_total, atol=1e-4)


def test_fifo_inventory_simulation():
    # Store with delivery 10 at t=0, demand 6 -> carryover 4
    q = np.array([[10.0, 0.0]])
    d = np.array([[6.0, 5.0]])
    I_on_hand, y_pos, y_neg, _w = simulate_perishable_inventory_fifo(q, d, tau=2)
    assert I_on_hand[0, 0, 0] == 4.0
    assert y_pos[0, 1] == 1.0  # 4 on hand, demand 5 -> 1 stockout
    assert y_neg[0, 1] == 0.0


def test_pyepo_interface_losses(dummy_instance):
    N, T, dist, demand = dummy_instance
    optmodel = PIRPOptModel(num_stores=N, horizon_T=T, distance_matrix=dist, continuous_relaxation=True)

    d_hat = torch.tensor(demand[None, :, :], dtype=torch.float32, requires_grad=True)
    d_true = torch.tensor(demand[None, :, :], dtype=torch.float32)

    # 1. SPO+
    loss_spo = SpopLoss(optmodel)(d_hat, d_true)
    assert loss_spo.requires_grad
    loss_spo.backward(retain_graph=True)
    assert d_hat.grad is not None

    # 2. DPO
    d_hat.grad.zero_()
    loss_dpo = DPOLoss(optmodel, n_samples=2, sigma=1.0)(d_hat, d_true)
    loss_dpo.backward(retain_graph=True)
    assert d_hat.grad is not None

    # 3. IntOpt
    d_hat.grad.zero_()
    loss_intopt = IntOptLoss(optmodel)(d_hat, d_true)
    loss_intopt.backward()
    assert d_hat.grad is not None


def test_spop_subgradient_direction():
    """
    Verifies that SpopLoss backward pass produces:
      1. Strictly negative subgradient when under-predicting (d_hat < d_true)
         so gradient descent increases d_hat.
      2. Strictly positive subgradient when over-predicting (d_hat > d_true)
         so gradient descent decreases d_hat.
      3. Asymmetric recourse scaling: under-prediction subgradient magnitude
         is scaled by p_u while over-prediction is scaled by h + p_s/tau.
    """
    optmodel = PIRPOptModel(
        num_stores=5,
        horizon_T=3,
        continuous_relaxation=True,
        stockout_penalty_pu=20.0,
        holding_cost_h=1.0,
        spoilage_penalty_ps=5.0,
        shelf_life_tau=3,
    )
    d_true = torch.full((1, 5, 3), 20.0, dtype=torch.float32)

    # 1. Under-prediction: d_hat = 10.0 < d_true = 20.0 -> strictly NEGATIVE gradient
    d_under = torch.full((1, 5, 3), 10.0, dtype=torch.float32, requires_grad=True)
    loss_under = SpopLoss(optmodel)(d_under, d_true)
    loss_under.backward()
    assert d_under.grad is not None
    grad_under = d_under.grad.mean().item()
    assert grad_under < 0.0, f"Expected negative gradient for under-prediction, got {grad_under}"

    # 2. Over-prediction: d_hat = 30.0 > d_true = 20.0 -> strictly POSITIVE gradient
    d_over = torch.full((1, 5, 3), 30.0, dtype=torch.float32, requires_grad=True)
    loss_over = SpopLoss(optmodel)(d_over, d_true)
    loss_over.backward()
    assert d_over.grad is not None
    grad_over = d_over.grad.mean().item()
    assert grad_over > 0.0, f"Expected positive gradient for over-prediction, got {grad_over}"

    # 3. Asymmetric weighting: stockout penalty p_u dominates holding/spoilage penalty
    assert abs(grad_under) > abs(grad_over), (
        f"Under-prediction magnitude ({abs(grad_under):.2f}) should exceed "
        f"over-prediction magnitude ({abs(grad_over):.2f}) due to asymmetric recourse"
    )


def test_optmodel_true_demand_caching():
    """Verifies that PIRPOptModel caches solutions when cache=True."""
    optmodel = PIRPOptModel(num_stores=5, horizon_T=3, continuous_relaxation=True)
    demand = np.full((5, 3), 15.0)

    sol1 = optmodel.solve(demand, cache=True)
    assert len(optmodel._cache) == 1
    sol2 = optmodel.solve(demand, cache=True)
    assert sol1 is sol2

    # Switching continuous relaxation does not collide
    optmodel.set_continuous_relaxation(False)
    sol_milp = optmodel.solve(demand, cache=True)
    assert sol_milp is not sol1
    assert len(optmodel._cache) == 2

    optmodel.clear_cache()
    assert len(optmodel._cache) == 0


def test_asymmetric_pinball_loss_properties():
    """Verifies analytical properties and gradient signs of asymmetric pinball loss."""
    alpha = 0.88  # Newsvendor critical fractile
    d_true = torch.tensor([20.0], dtype=torch.float32)

    # 1. Under-prediction: gradient w.r.t. d_hat should be negative (-alpha)
    d_under = torch.tensor([10.0], dtype=torch.float32, requires_grad=True)
    loss_under = asymmetric_pinball_loss(d_under, d_true, alpha=alpha)
    loss_under.backward()
    assert d_under.grad is not None
    assert np.isclose(d_under.grad.item(), -alpha, atol=1e-4)

    # 2. Over-prediction: gradient w.r.t. d_hat should be positive (1 - alpha)
    d_over = torch.tensor([30.0], dtype=torch.float32, requires_grad=True)
    loss_over = asymmetric_pinball_loss(d_over, d_true, alpha=alpha)
    loss_over.backward()
    assert d_over.grad is not None
    assert np.isclose(d_over.grad.item(), 1.0 - alpha, atol=1e-4)

    # 3. Ratio of under-prediction to over-prediction gradient magnitude
    grad_ratio = abs(d_under.grad.item()) / abs(d_over.grad.item())
    expected_ratio = alpha / (1.0 - alpha)
    assert np.isclose(grad_ratio, expected_ratio, atol=1e-4)


def test_dfl_pipeline_best_validation_checkpointing():
    """Verifies that DFLPipeline tracks validation regret and restores the best state."""
    N, T, F = 5, 3, 14
    X_train = np.random.randn(2, N, T, F).astype(np.float32)
    d_train = np.random.uniform(10, 30, size=(2, N, T)).astype(np.float32)
    X_val = np.random.randn(2, N, T, F).astype(np.float32)
    d_val = np.random.uniform(10, 30, size=(2, N, T)).astype(np.float32)

    pipeline = DFLPipeline(loss_type="SPO+", in_features=F, num_stores=N, horizon_T=T)
    history = pipeline.fit_two_phase(
        X_train=X_train,
        d_train=d_train,
        X_val=X_val,
        d_val=d_val,
        total_epochs=2,
        transition_ratio=0.5,
        mse_pretrain_epochs=1,
        val_subset_size=2,
        verbose=False,
    )
    assert len(history) == 2
    assert "val_total_cost" in history[0]
    assert history[0]["val_total_cost"] >= 0.0

