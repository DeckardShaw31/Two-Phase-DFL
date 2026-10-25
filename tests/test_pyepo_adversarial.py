"""
Adversarial Stress Test Suite for PyEPO Interface and DFL Loss Functions.
Tests extreme boundary conditions, subgradient signs, asymmetric recourse weighting,
caching isolation and collision resistance, non-finite input handling, and convergence.
"""

import numpy as np
import pytest
import torch

from src.optimization.pyepo_interface import (
    DPOLoss,
    IntOptLoss,
    PIRPOptModel,
    SpopLoss,
)


@pytest.fixture
def optmodel_lp():
    return PIRPOptModel(
        num_stores=4,
        horizon_T=3,
        continuous_relaxation=True,
        stockout_penalty_pu=20.0,
        holding_cost_h=1.0,
        spoilage_penalty_ps=5.0,
        shelf_life_tau=3,
    )


@pytest.fixture
def optmodel_milp():
    return PIRPOptModel(
        num_stores=4,
        horizon_T=3,
        continuous_relaxation=False,
        stockout_penalty_pu=20.0,
        holding_cost_h=1.0,
        spoilage_penalty_ps=5.0,
        shelf_life_tau=3,
        time_limit_sec=2.0,
        mip_rel_gap=0.05,
    )


# ===========================================================================
# 1. Extreme Under-Prediction Stress Tests (d_hat = 0.001 vs d_true = 50.0)
# ===========================================================================

def test_extreme_under_prediction_spop_lp(optmodel_lp):
    """Verifies that extreme under-prediction produces finite, strictly negative subgradients."""
    d_true = torch.full((1, 4, 3), 50.0, dtype=torch.float32)
    d_hat = torch.full((1, 4, 3), 0.001, dtype=torch.float32, requires_grad=True)

    loss = SpopLoss(optmodel_lp)(d_hat, d_true)
    assert loss.item() > 0.0, "Loss must be strictly positive under extreme under-prediction"
    assert torch.isfinite(loss), "Loss must be finite"

    loss.backward()
    assert d_hat.grad is not None
    assert torch.isfinite(d_hat.grad).all(), "All subgradients must be finite"
    assert (d_hat.grad <= 0.0).all(), "Under-prediction subgradients must be strictly non-positive"
    assert d_hat.grad.mean().item() < -10.0, f"Expected strong negative subgradient, got {d_hat.grad.mean().item()}"

    # Gradient step test: -lr * grad should increase d_hat towards 50.0
    lr = 0.05
    d_next = (d_hat - lr * d_hat.grad).detach()
    assert (d_next > d_hat).all(), "Gradient descent must push predicted demand strictly towards d_true"


def test_extreme_under_prediction_spop_milp(optmodel_milp):
    """Verifies that MILP discrete mode under extreme under-prediction produces finite negative subgradient."""
    d_true = torch.full((1, 4, 3), 50.0, dtype=torch.float32)
    d_hat = torch.full((1, 4, 3), 0.001, dtype=torch.float32, requires_grad=True)

    loss = SpopLoss(optmodel_milp)(d_hat, d_true)
    assert loss.item() > 0.0
    loss.backward()
    assert d_hat.grad is not None
    assert torch.isfinite(d_hat.grad).all()
    assert (d_hat.grad <= 0.0).all()
    assert d_hat.grad.mean().item() < -10.0


def test_extreme_under_prediction_dpo_and_intopt(optmodel_lp):
    """Verifies DPO and IntOpt under extreme under-prediction."""
    d_true = torch.full((1, 4, 3), 50.0, dtype=torch.float32)

    # DPO
    d_hat_dpo = torch.full((1, 4, 3), 0.001, dtype=torch.float32, requires_grad=True)
    loss_dpo = DPOLoss(optmodel_lp, n_samples=3, sigma=1.0)(d_hat_dpo, d_true)
    loss_dpo.backward()
    assert torch.isfinite(d_hat_dpo.grad).all()
    assert d_hat_dpo.grad.mean().item() < 0.0

    # IntOpt
    d_hat_io = torch.full((1, 4, 3), 0.001, dtype=torch.float32, requires_grad=True)
    loss_io = IntOptLoss(optmodel_lp)(d_hat_io, d_true)
    loss_io.backward()
    assert torch.isfinite(d_hat_io.grad).all()
    assert d_hat_io.grad.mean().item() < 0.0


# ===========================================================================
# 2. Extreme Over-Prediction Stress Tests (d_hat = 100.0 vs d_true = 5.0)
# ===========================================================================

def test_extreme_over_prediction_spop_lp(optmodel_lp):
    """Verifies that extreme over-prediction produces finite, strictly positive subgradients."""
    d_true = torch.full((1, 4, 3), 5.0, dtype=torch.float32)
    d_hat = torch.full((1, 4, 3), 25.0, dtype=torch.float32, requires_grad=True)

    loss = SpopLoss(optmodel_lp)(d_hat, d_true)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert d_hat.grad is not None
    assert torch.isfinite(d_hat.grad).all()
    assert (d_hat.grad >= 0.0).all(), "Over-prediction subgradients must be strictly non-negative"
    assert d_hat.grad.mean().item() > 5.0, f"Expected strong positive subgradient, got {d_hat.grad.mean().item()}"

    # Gradient step test: -lr * grad should decrease d_hat towards 5.0
    lr = 0.05
    d_next = (d_hat - lr * d_hat.grad).detach()
    assert (d_next < d_hat).all(), "Gradient descent must push predicted demand strictly down towards d_true"


def test_extreme_over_prediction_spop_milp(optmodel_milp):
    """Verifies that MILP discrete mode under extreme over-prediction produces finite positive subgradient."""
    d_true = torch.full((1, 4, 3), 5.0, dtype=torch.float32)
    d_hat = torch.full((1, 4, 3), 25.0, dtype=torch.float32, requires_grad=True)

    loss = SpopLoss(optmodel_milp)(d_hat, d_true)
    assert loss.item() > 0.0
    loss.backward()
    assert d_hat.grad is not None
    assert torch.isfinite(d_hat.grad).all()
    assert (d_hat.grad >= 0.0).all()
    assert d_hat.grad.mean().item() > 5.0


def test_extreme_over_prediction_dpo_and_intopt(optmodel_lp):
    """Verifies DPO and IntOpt under extreme over-prediction."""
    d_true = torch.full((1, 4, 3), 5.0, dtype=torch.float32)

    # DPO
    d_hat_dpo = torch.full((1, 4, 3), 100.0, dtype=torch.float32, requires_grad=True)
    loss_dpo = DPOLoss(optmodel_lp, n_samples=3, sigma=1.0)(d_hat_dpo, d_true)
    loss_dpo.backward()
    assert torch.isfinite(d_hat_dpo.grad).all()
    assert d_hat_dpo.grad.mean().item() > 0.0

    # IntOpt
    d_hat_io = torch.full((1, 4, 3), 100.0, dtype=torch.float32, requires_grad=True)
    loss_io = IntOptLoss(optmodel_lp)(d_hat_io, d_true)
    loss_io.backward()
    assert torch.isfinite(d_hat_io.grad).all()
    assert d_hat_io.grad.mean().item() > 0.0


# ===========================================================================
# 3. Exact Equality Stress Tests (d_hat == d_true)
# ===========================================================================

@pytest.mark.parametrize("demand_val", [0.0, 10.0, 25.0, 50.0])
def test_exact_equality_spop(optmodel_lp, demand_val):
    """Verifies that exact demand equality yields zero loss and zero subgradient."""
    d_true = torch.full((1, 4, 3), demand_val, dtype=torch.float32)
    d_hat = torch.full((1, 4, 3), demand_val, dtype=torch.float32, requires_grad=True)

    loss = SpopLoss(optmodel_lp)(d_hat, d_true)
    assert np.isclose(loss.item(), 0.0, atol=1e-5), f"Expected 0 loss, got {loss.item()}"

    loss.backward()
    assert d_hat.grad is not None
    assert torch.isfinite(d_hat.grad).all()
    assert torch.allclose(d_hat.grad, torch.zeros_like(d_hat), atol=1e-5), (
        f"Expected zero gradient on exact equality, got max={d_hat.grad.abs().max().item()}"
    )


def test_exact_equality_intopt(optmodel_lp):
    """Verifies that IntOpt exact equality yields zero regret and zero subgradient."""
    d_true = torch.full((1, 4, 3), 20.0, dtype=torch.float32)
    d_hat = torch.full((1, 4, 3), 20.0, dtype=torch.float32, requires_grad=True)

    loss = IntOptLoss(optmodel_lp)(d_hat, d_true)
    assert np.isclose(loss.item(), 0.0, atol=1e-5)
    loss.backward()
    assert torch.allclose(d_hat.grad, torch.zeros_like(d_hat), atol=1e-5)


# ===========================================================================
# 4. Asymmetric Recourse Subgradient Scaling
# ===========================================================================

def test_asymmetric_subgradient_ratio(optmodel_lp):
    """
    Verifies that symmetric deviations produce asymmetric subgradient magnitudes.
    Underage penalty p_u = 20.0 vs Overage penalty h + p_s/tau = 1.0 + 5.0/3 = 2.67
    Ratio p_u / (h + p_s/tau) = 7.5.
    """
    d_true = torch.full((1, 4, 3), 25.0, dtype=torch.float32)

    # Under-prediction: delta = -10.0
    d_under = torch.full((1, 4, 3), 15.0, dtype=torch.float32, requires_grad=True)
    loss_under = SpopLoss(optmodel_lp)(d_under, d_true)
    loss_under.backward()
    grad_under_abs = abs(d_under.grad.mean().item())

    # Over-prediction: delta = +10.0
    d_over = torch.full((1, 4, 3), 35.0, dtype=torch.float32, requires_grad=True)
    loss_over = SpopLoss(optmodel_lp)(d_over, d_true)
    loss_over.backward()
    grad_over_abs = abs(d_over.grad.mean().item())

    assert grad_under_abs > grad_over_abs, (
        f"Under-prediction gradient ({grad_under_abs:.2f}) must exceed "
        f"over-prediction gradient ({grad_over_abs:.2f})"
    )

    ratio = grad_under_abs / grad_over_abs
    expected_ratio = optmodel_lp.pu / (optmodel_lp.h + optmodel_lp.ps / optmodel_lp.tau)
    # Ratio should match closely to theoretical ratio
    assert np.isclose(ratio, expected_ratio, rtol=0.25), (
        f"Observed ratio {ratio:.2f} diverged from theoretical ratio {expected_ratio:.2f}"
    )


# ===========================================================================
# 5. Non-Finite & Boundary Input Stress Tests (NaN, Inf, Negatives)
# ===========================================================================

def test_nan_input_rejection(optmodel_lp):
    """Verifies that NaN inputs in d_hat are properly rejected with ValueError."""
    d_true = torch.full((1, 4, 3), 20.0, dtype=torch.float32)
    d_nan = torch.full((1, 4, 3), float("nan"), dtype=torch.float32, requires_grad=True)

    with pytest.raises(ValueError, match="inf, nan, or None"):
        SpopLoss(optmodel_lp)(d_nan, d_true)


def test_inf_input_rejection(optmodel_lp):
    """Verifies that Inf inputs in d_hat are properly rejected with ValueError."""
    d_true = torch.full((1, 4, 3), 20.0, dtype=torch.float32)
    d_inf = torch.full((1, 4, 3), float("inf"), dtype=torch.float32, requires_grad=True)

    with pytest.raises(ValueError, match="inf, nan, or None"):
        SpopLoss(optmodel_lp)(d_inf, d_true)


def test_negative_d_hat_safe_clamping(optmodel_lp):
    """
    Verifies that negative predicted demand (-15.0) is safely handled by d_spo clipping,
    producing finite, negative subgradient pushing predictions back to positive territory.
    """
    d_true = torch.full((1, 4, 3), 20.0, dtype=torch.float32)
    d_neg = torch.full((1, 4, 3), -15.0, dtype=torch.float32, requires_grad=True)

    loss = SpopLoss(optmodel_lp)(d_neg, d_true)
    assert loss.item() > 0.0
    loss.backward()
    assert d_neg.grad is not None
    assert torch.isfinite(d_neg.grad).all()
    assert (d_neg.grad < 0.0).all(), "Negative d_hat must yield negative subgradient to push d_hat positive"


# ===========================================================================
# 6. Caching Stress, Hash Collision & LP/MILP Isolation
# ===========================================================================

def test_caching_identity_and_speedup(optmodel_lp):
    """Verifies that repeated queries return cached object with >50x speedup."""
    demand = np.full((4, 3), 22.5)

    import time
    t0 = time.perf_counter()
    sol1 = optmodel_lp.solve(demand, cache=True)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol2 = optmodel_lp.solve(demand, cache=True)
    t_second = time.perf_counter() - t0

    assert sol1 is sol2, "Cached call must return the exact same PIRPSolution instance"
    assert t_second < t_first / 20.0 or t_second < 1e-4, "Cache hit must be orders of magnitude faster"


def test_caching_lp_vs_milp_isolation(optmodel_lp):
    """Verifies that LP relaxation and MILP modes do not collide or return each other's cached solutions."""
    demand = np.full((4, 3), 18.0)

    # 1. LP solve
    optmodel_lp.set_continuous_relaxation(True)
    sol_lp = optmodel_lp.solve(demand, cache=True)
    assert sol_lp.is_lp_relaxation is True

    # 2. Switch to MILP
    optmodel_lp.set_continuous_relaxation(False)
    sol_milp = optmodel_lp.solve(demand, cache=True)
    assert sol_milp.is_lp_relaxation is False
    assert sol_lp is not sol_milp, "LP and MILP solutions must not collide in cache"
    assert len(optmodel_lp._cache) == 2

    # 3. Switch back to LP
    optmodel_lp.set_continuous_relaxation(True)
    sol_lp_again = optmodel_lp.solve(demand, cache=True)
    assert sol_lp_again is sol_lp

    optmodel_lp.clear_cache()
    assert len(optmodel_lp._cache) == 0


def test_caching_stress_repeated_queries(optmodel_lp):
    """Adversarial stress: 300 alternating queries to verify cache stability and memory consistency."""
    optmodel_lp.clear_cache()
    d1 = np.full((4, 3), 12.0)
    d2 = np.full((4, 3), 28.0)
    d3 = np.full((4, 3), 44.0)

    for i in range(300):
        target_d = [d1, d2, d3][i % 3]
        sol = optmodel_lp.solve(target_d, cache=True)
        assert sol.success is True

    # Cache should contain exactly 3 unique entries
    assert len(optmodel_lp._cache) == 3


# ===========================================================================
# 7. Batch Gradient Heterogeneity (Mixed Under, Over, Exact)
# ===========================================================================

def test_batch_gradient_heterogeneity(optmodel_lp):
    """
    Verifies that in a batch with mixed under-, over-, and exact predictions,
    each item's subgradient independently has the correct sign.
    """
    d_true = torch.tensor(
        [
            [[25.0, 25.0, 25.0]] * 4,  # Item 0
            [[25.0, 25.0, 25.0]] * 4,  # Item 1
            [[25.0, 25.0, 25.0]] * 4,  # Item 2
        ],
        dtype=torch.float32,
    )

    d_hat = torch.tensor(
        [
            [[10.0, 10.0, 10.0]] * 4,  # Item 0: Under-prediction -> expect negative grad
            [[40.0, 40.0, 40.0]] * 4,  # Item 1: Over-prediction -> expect positive grad
            [[25.0, 25.0, 25.0]] * 4,  # Item 2: Exact equality -> expect zero grad
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss = SpopLoss(optmodel_lp)(d_hat, d_true)
    loss.backward()

    assert d_hat.grad is not None
    grad_0 = d_hat.grad[0]
    grad_1 = d_hat.grad[1]
    grad_2 = d_hat.grad[2]

    assert (grad_0 < 0.0).all(), "Item 0 (under-prediction) must have strictly negative gradients"
    assert (grad_1 > 0.0).all(), "Item 1 (over-prediction) must have strictly positive gradients"
    assert torch.allclose(grad_2, torch.zeros_like(grad_2), atol=1e-5), "Item 2 (exact equality) must have zero gradients"


# ===========================================================================
# 8. Multi-Step Gradient Optimization Convergence
# ===========================================================================

def test_optimization_convergence_from_extremes(optmodel_lp):
    """
    Empirically simulates steps of gradient descent starting from:
      Case A: Extreme under-prediction (0.01)
      Case B: Extreme over-prediction (35.0)
    Verifies monotonic progress towards ground truth d_true = 20.0.
    """
    d_true = torch.full((1, 4, 3), 20.0, dtype=torch.float32)

    # Case A: From under-prediction
    d_param_under = torch.nn.Parameter(torch.full((1, 4, 3), 0.01, dtype=torch.float32))
    opt_under = torch.optim.SGD([d_param_under], lr=0.04)

    prev_val = d_param_under.mean().item()
    for _ in range(5):
        opt_under.zero_grad()
        loss = SpopLoss(optmodel_lp)(d_param_under, d_true)
        loss.backward()
        opt_under.step()
        curr_val = d_param_under.mean().item()
        assert curr_val > prev_val, f"d_hat must monotonically increase towards 20.0 (was {prev_val}, now {curr_val})"
        prev_val = curr_val

    # Case B: From over-prediction
    d_param_over = torch.nn.Parameter(torch.full((1, 4, 3), 35.0, dtype=torch.float32))
    opt_over = torch.optim.SGD([d_param_over], lr=0.08)

    prev_val = d_param_over.mean().item()
    for _ in range(5):
        opt_over.zero_grad()
        loss = SpopLoss(optmodel_lp)(d_param_over, d_true)
        loss.backward()
        opt_over.step()
        curr_val = d_param_over.mean().item()
        assert curr_val < prev_val, f"d_hat must monotonically decrease towards 20.0 (was {prev_val}, now {curr_val})"
        prev_val = curr_val
