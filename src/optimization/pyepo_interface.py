"""
PyEPO-compatible Differentiable Optimization Interface and Loss Functions.

Implements:
  1. PIRPOptModel: PyEPO-style optimization model wrapper.
  2. SPO+ (Smart Predict-then-Optimize) autograd function and loss module.
  3. DPO (Differentiable Perturbed Optimizer) autograd function and loss module.
  4. IntOpt (Interior Point Differentiation) autograd function and loss module.
Supports both Continuous LP Relaxation (for fast warmup) and Discrete MILP.
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from .pirp_model import PIRPModel, PIRPSolution


class PIRPOptModel:
    """
    PyEPO-style optimization model interface for the Multi-Period PIRP.
    Maintains internal solver and caches oracle solutions.
    """

    def __init__(
        self,
        num_stores: int = 10,
        horizon_T: int = 5,
        fleet_size_K: int = 2,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        shelf_life_tau: int = 3,
        holding_cost_h: float = 1.0,
        aging_surcharge_eps: float = 0.01,
        spoilage_penalty_ps: float = 5.0,
        stockout_penalty_pu: float = 20.0,
        distance_matrix: np.ndarray | None = None,
        continuous_relaxation: bool = False,
        time_limit_sec: float | None = None,
        mip_rel_gap: float = 0.05,
    ):
        self.N = num_stores
        self.T = horizon_T
        self.fleet_size_K = fleet_size_K
        self.vehicle_capacity_Q = vehicle_capacity_Q
        self.store_capacity_Ci = store_capacity_Ci
        self.tau = shelf_life_tau
        self.h = holding_cost_h
        self.aging_surcharge_eps = aging_surcharge_eps
        self.ps = spoilage_penalty_ps
        self.pu = stockout_penalty_pu
        self.distance_matrix = distance_matrix
        self.continuous_relaxation = continuous_relaxation
        self.time_limit_sec = time_limit_sec
        self.mip_rel_gap = mip_rel_gap

        self.model = PIRPModel(
            num_stores=num_stores,
            horizon_T=horizon_T,
            fleet_size_K=fleet_size_K,
            vehicle_capacity_Q=vehicle_capacity_Q,
            store_capacity_Ci=store_capacity_Ci,
            shelf_life_tau=shelf_life_tau,
            holding_cost_h=holding_cost_h,
            aging_surcharge_eps=aging_surcharge_eps,
            spoilage_penalty_ps=spoilage_penalty_ps,
            stockout_penalty_pu=stockout_penalty_pu,
            distance_matrix=distance_matrix,
            continuous_relaxation=continuous_relaxation,
        )

        self.distance_matrix = (
            np.array(distance_matrix, dtype=np.float64) if distance_matrix is not None else self.model.dist
        )

        self._cache: dict[tuple[bool, bytes], PIRPSolution] = {}

    def clear_cache(self):
        """Clears the solution cache."""
        self._cache.clear()

    def set_continuous_relaxation(self, continuous: bool):
        """Switches between continuous LP relaxation and discrete MILP."""
        self.continuous_relaxation = continuous
        self.model.set_continuous_relaxation(continuous)

    def _demand_key(self, demand: np.ndarray) -> tuple[bool, bytes]:
        arr = np.ascontiguousarray(demand, dtype=np.float64)
        return (self.continuous_relaxation, arr.tobytes())

    def solve(self, demand: np.ndarray, cache: bool = False) -> PIRPSolution:
        """Solves the instance for demand array of shape (N, T), optionally using cache."""
        if cache:
            key = self._demand_key(demand)
            if key in self._cache:
                return self._cache[key]
            sol = self.model.solve(
                demand=demand,
                time_limit_sec=self.time_limit_sec,
                mip_rel_gap=self.mip_rel_gap,
            )
            self._cache[key] = sol
            return sol

        return self.model.solve(
            demand=demand,
            time_limit_sec=self.time_limit_sec,
            mip_rel_gap=self.mip_rel_gap,
        )

    def solve_true(self, demand: np.ndarray) -> PIRPSolution:
        """Solves oracle instance under true demand, utilizing cache across epochs."""
        return self.solve(demand, cache=True)

    def get_decision_vector(self, sol: PIRPSolution) -> np.ndarray:
        """Extracts the operational replenishment decision vector q_total (N, T)."""
        return sol.q_total


# ---------------------------------------------------------------------------
# 1. Smart Predict-then-Optimize (SPO+)
# ---------------------------------------------------------------------------

class SpopFunction(Function):
    """
    Custom PyTorch autograd Function for SPO+ surrogate loss in demand space.
    Forward:
      Computes asymmetric recourse-weighted SPO+ loss in demand space.
      Upper bound on regret accounting for asymmetric penalties (p_u vs h + p_s/tau).
    Backward:
      Demand-space subgradient with corrected sign: 2 * weights * (w_spo - w_true) / batch_size
      where under-prediction yields negative gradient and over-prediction yields positive gradient.
    """

    @staticmethod
    def forward(ctx, d_hat: torch.Tensor, d_true: torch.Tensor, optmodel: PIRPOptModel) -> torch.Tensor:
        """
        Args:
            d_hat: Predicted demand tensor of shape (batch_size, N, T).
            d_true: True demand tensor of shape (batch_size, N, T).
            optmodel: PIRPOptModel instance.
        """
        device = d_hat.device
        dtype = d_hat.dtype
        batch_size, _N, _T = d_hat.shape

        d_hat_np = d_hat.detach().cpu().numpy()
        d_true_np = d_true.detach().cpu().numpy()

        w_true_batch = []
        w_spo_batch = []
        losses = []

        pu = optmodel.pu
        co = optmodel.h + optmodel.ps / max(1.0, float(optmodel.tau))
        c_norm = (pu + co) / 2.0

        for b in range(batch_size):
            d_t = d_true_np[b]
            d_h = d_hat_np[b]

            # Solve oracle at true demand with caching
            sol_true = optmodel.solve(d_t, cache=True)
            w_true = sol_true.q_total  # shape (N, T)

            # SPO+ perturbed demand: 2*d_hat - d_true
            d_spo = np.maximum(0.0, 2.0 * d_h - d_t)
            sol_spo = optmodel.solve(d_spo)
            w_spo = sol_spo.q_total    # shape (N, T)

            # Asymmetric SPO+ loss in demand space: (2*d_h - d_t)^T (weights * (w_spo - w_true))
            delta_w = w_spo - w_true
            weights = np.where(delta_w < 0.0, pu / c_norm, co / c_norm)
            spo_loss_val = np.sum((2.0 * d_h - d_t) * (weights * delta_w)) + np.maximum(
                0.0, sol_spo.total_cost - sol_true.total_cost
            )

            w_true_batch.append(w_true)
            w_spo_batch.append(w_spo)
            losses.append(max(0.0, float(spo_loss_val)))

        w_true_tensor = torch.tensor(np.array(w_true_batch), device=device, dtype=dtype)
        w_spo_tensor = torch.tensor(np.array(w_spo_batch), device=device, dtype=dtype)

        ctx.save_for_backward(w_true_tensor, w_spo_tensor)
        ctx.pu = float(pu)
        ctx.co = float(co)
        return torch.tensor(np.mean(losses), device=device, dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        w_true, w_spo = ctx.saved_tensors
        batch_size = w_true.shape[0]
        pu = ctx.pu
        co = ctx.co
        c_norm = (pu + co) / 2.0

        # Demand-space subgradient with corrected sign: (w_spo - w_true)
        # Under-prediction (w_spo < w_true) yields negative gradient -> gradient descent increases d_hat
        # Over-prediction (w_spo >= w_true) yields positive gradient -> gradient descent decreases d_hat
        delta_w = w_spo - w_true
        weights = torch.where(delta_w < 0.0, pu / c_norm, co / c_norm)
        grad_d_hat = grad_output * 2.0 * (weights * delta_w) / float(batch_size)
        return grad_d_hat, None, None


class SpopLoss(nn.Module):
    """SPO+ Loss Module."""

    def __init__(self, optmodel: PIRPOptModel):
        super().__init__()
        self.optmodel = optmodel

    def forward(self, d_hat: torch.Tensor, d_true: torch.Tensor) -> torch.Tensor:
        return SpopFunction.apply(d_hat, d_true, self.optmodel)


# ---------------------------------------------------------------------------
# 2. Differentiable Perturbed Optimizer (DPO)
# ---------------------------------------------------------------------------

class DPOFunction(Function):
    """
    Custom PyTorch autograd Function for Differentiable Perturbed Optimizer (DPO).
    Uses Monte Carlo Gaussian perturbation to smooth combinatorial optimization,
    with demand-space asymmetric recourse weighting.
    """

    @staticmethod
    def forward(
        ctx,
        d_hat: torch.Tensor,
        d_true: torch.Tensor,
        optmodel: PIRPOptModel,
        n_samples: int = 5,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        device = d_hat.device
        dtype = d_hat.dtype
        batch_size, _N, _T = d_hat.shape

        d_hat_np = d_hat.detach().cpu().numpy()
        d_true_np = d_true.detach().cpu().numpy()

        grad_batch = []
        losses = []

        pu = optmodel.pu
        co = optmodel.h + optmodel.ps / max(1.0, float(optmodel.tau))

        for b in range(batch_size):
            d_h = d_hat_np[b]
            d_t = d_true_np[b]

            sol_true = optmodel.solve(d_t, cache=True)
            w_true = sol_true.q_total

            # Monte Carlo perturbations
            w_pert_sum = np.zeros_like(d_h)
            for _ in range(n_samples):
                noise = np.random.normal(0, sigma, size=d_h.shape)
                d_pert = np.maximum(0.0, d_h + noise)
                sol_pert = optmodel.solve(d_pert)
                w_pert_sum += sol_pert.q_total

            w_pert_mean = w_pert_sum / float(n_samples)
            # Demand-space gradient direction: (w_pert_mean - w_true)
            delta_w = w_pert_mean - w_true
            grad_batch.append(delta_w)

            sol_hat = optmodel.solve(d_h)
            regret = max(0.0, sol_hat.total_cost - sol_true.total_cost)
            losses.append(regret)

        grad_tensor = torch.tensor(np.array(grad_batch), device=device, dtype=dtype)
        ctx.save_for_backward(grad_tensor)
        ctx.pu = float(pu)
        ctx.co = float(co)
        return torch.tensor(np.mean(losses), device=device, dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_tensor,) = ctx.saved_tensors
        batch_size = grad_tensor.shape[0]
        pu = ctx.pu
        co = ctx.co
        c_norm = (pu + co) / 2.0

        delta_w = grad_tensor
        weights = torch.where(delta_w < 0.0, pu / c_norm, co / c_norm)
        grad_d_hat = grad_output * (weights * delta_w) / float(batch_size)
        return grad_d_hat, None, None, None, None


class DPOLoss(nn.Module):
    """Differentiable Perturbed Optimizer Loss Module."""

    def __init__(self, optmodel: PIRPOptModel, n_samples: int = 5, sigma: float = 2.0):
        super().__init__()
        self.optmodel = optmodel
        self.n_samples = n_samples
        self.sigma = sigma

    def forward(self, d_hat: torch.Tensor, d_true: torch.Tensor) -> torch.Tensor:
        return DPOFunction.apply(d_hat, d_true, self.optmodel, self.n_samples, self.sigma)


# ---------------------------------------------------------------------------
# 3. Interior Point Differentiation (IntOpt)
# ---------------------------------------------------------------------------

class IntOptFunction(Function):
    """
    Custom PyTorch autograd Function for Interior Point Differentiation (IntOpt).
    Solves continuous LP relaxation to provide smooth gradients,
    with demand-space asymmetric recourse weighting.
    """

    @staticmethod
    def forward(
        ctx,
        d_hat: torch.Tensor,
        d_true: torch.Tensor,
        optmodel: PIRPOptModel,
        damping: float = 1e-2,
    ) -> torch.Tensor:
        device = d_hat.device
        dtype = d_hat.dtype
        batch_size, _N, _T = d_hat.shape

        d_hat_np = d_hat.detach().cpu().numpy()
        d_true_np = d_true.detach().cpu().numpy()

        pu = optmodel.pu
        co = optmodel.h + optmodel.ps / max(1.0, float(optmodel.tau))

        # Temporarily enforce continuous relaxation for IntOpt smoothing
        prev_relax = optmodel.continuous_relaxation
        optmodel.set_continuous_relaxation(True)

        w_diff_batch = []
        losses = []

        for b in range(batch_size):
            d_h = d_hat_np[b]
            d_t = d_true_np[b]

            sol_hat = optmodel.solve(d_h)
            sol_true = optmodel.solve(d_t, cache=True)

            w_hat = sol_hat.q_total
            w_true = sol_true.q_total

            # Barrier-smoothed gradient direction: (w_hat - w_true)
            w_diff_batch.append(w_hat - w_true)
            regret = max(0.0, sol_hat.total_cost - sol_true.total_cost)
            losses.append(regret)

        optmodel.set_continuous_relaxation(prev_relax)

        w_diff_tensor = torch.tensor(np.array(w_diff_batch), device=device, dtype=dtype)
        ctx.save_for_backward(w_diff_tensor)
        ctx.pu = float(pu)
        ctx.co = float(co)
        return torch.tensor(np.mean(losses), device=device, dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w_diff,) = ctx.saved_tensors
        batch_size = w_diff.shape[0]
        pu = ctx.pu
        co = ctx.co
        c_norm = (pu + co) / 2.0

        weights = torch.where(w_diff < 0.0, pu / c_norm, co / c_norm)
        grad_d_hat = grad_output * (weights * w_diff) / float(batch_size)
        return grad_d_hat, None, None, None


class IntOptLoss(nn.Module):
    """Interior Point Differentiation Loss Module."""

    def __init__(self, optmodel: PIRPOptModel, damping: float = 1e-2):
        super().__init__()
        self.optmodel = optmodel
        self.damping = damping

    def forward(self, d_hat: torch.Tensor, d_true: torch.Tensor) -> torch.Tensor:
        return IntOptFunction.apply(d_hat, d_true, self.optmodel, self.damping)
