"""Custom operational heuristics for the synthetic PIRP benchmark.

These implementations are exploratory adaptations, not reproductions of the
published methods named in the previous version of this module. The validation
protocol has not been preregistered and Gate 2 has not passed. In particular,
the residual-scenario search is not a nonanticipative SAA formulation and the
randomized destroy/repair routine is not adaptive large-neighborhood search.
Do not label either routine SAA or ALNS in results. The primary comparison uses
only methods with decision rules and information timing specified in the
revision experiment script.
"""

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from src.optimization.pirp_model import PIRPModel, PIRPSolution
from src.utils.metrics import simulate_perishable_inventory_fifo


@dataclass
class OperationalPolicyResult:
    """Standardized decision container for all operational baselines."""
    q_plan: np.ndarray          # Deliveries (N, T)
    x_plan: np.ndarray          # Routes (N+1, N+1, K, T)
    solve_time_sec: float       # Execution / inference runtime in seconds
    method_name: str            # Baseline identifier
    metadata: dict[str, Any]    # Additional diagnostics (iterations, gap, etc.)


def build_capacitated_routes_2opt(
    q_target: np.ndarray,
    distance_matrix: np.ndarray,
    fleet_size_K: int,
    vehicle_capacity_Q: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Allocates target deliveries q_target (N,) to K vehicles of capacity Q
    and constructs 2-opt TSP tours from central depot 0.

    Args:
        q_target: Non-negative delivery target for stores 1..N, shape (N,).
        distance_matrix: (N+1, N+1) distance matrix (0 is depot).
        fleet_size_K: Number of homogeneous vehicles.
        vehicle_capacity_Q: Maximum payload per vehicle.

    Returns:
        q_delivered: Feasible delivered quantities (N,), clipped to total fleet capacity.
        x_routes: Binary routing matrix (N+1, N+1, K).
        total_routing_cost: Sum of travel distances across all vehicles.
    """
    N = len(q_target)
    K = fleet_size_K
    Q = float(vehicle_capacity_Q)

    q_deliv = np.zeros(N, dtype=np.float64)
    x_routes = np.zeros((N + 1, N + 1, K), dtype=np.float64)

    # Active stores requiring delivery
    active_stores = [i + 1 for i in range(N) if q_target[i] > 1e-6]
    if not active_stores:
        return q_deliv, x_routes, 0.0

    # Sort stores by angular coordinate or distance from depot for clustered packing
    active_stores.sort(key=lambda s: distance_matrix[0, s], reverse=True)

    bins: list[list[int]] = [[] for _ in range(K)]
    bin_loads = np.zeros(K, dtype=np.float64)

    for store_i in active_stores:
        dem = float(q_target[store_i - 1])
        # Find vehicle with minimum load that has space
        assigned = False
        sorted_k = np.argsort(bin_loads)
        for k in sorted_k:
            avail = max(0.0, Q - bin_loads[k])
            if avail > 1e-6:
                alloc = min(dem, avail)
                q_deliv[store_i - 1] += alloc
                bin_loads[k] += alloc
                bins[k].append(store_i)
                assigned = True
                break
        if not assigned:
            # Fleet capacity exhausted; store cannot be served in this period
            continue

    # Build 2-opt TSP tour for each vehicle with assigned stores
    total_routing_cost = 0.0
    for k in range(K):
        tour = bins[k]
        if not tour:
            continue

        # Nearest-neighbor heuristic from depot 0
        unvisited = list(tour)
        ordered: list[int] = []
        curr = 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: distance_matrix[curr, j])
            ordered.append(nxt)
            unvisited.remove(nxt)
            curr = nxt

        # 2-opt local search improvement
        m = len(ordered)
        if m >= 3:
            improved = True
            while improved:
                improved = False
                nodes = [0] + ordered + [0]
                for a_idx in range(1, m):
                    for b_idx in range(a_idx + 1, m + 1):
                        u1, u2 = nodes[a_idx - 1], nodes[a_idx]
                        v1, v2 = nodes[b_idx], nodes[b_idx + 1]
                        if distance_matrix[u1, v1] + distance_matrix[u2, v2] + 1e-9 < distance_matrix[u1, u2] + distance_matrix[v1, v2]:
                            ordered[a_idx - 1 : b_idx] = reversed(ordered[a_idx - 1 : b_idx])
                            improved = True
                            break
                    if improved:
                        break

        # Record tour edges
        x_routes[0, ordered[0], k] = 1.0
        x_routes[ordered[-1], 0, k] = 1.0
        total_routing_cost += float(distance_matrix[0, ordered[0]] + distance_matrix[ordered[-1], 0])

        for idx_pos in range(len(ordered) - 1):
            u, v = ordered[idx_pos], ordered[idx_pos + 1]
            x_routes[u, v, k] = 1.0
            total_routing_cost += float(distance_matrix[u, v])

    return q_deliv, x_routes, total_routing_cost


# =========================================================================
# EXPLORATORY CUSTOM AGE-AWARE BASE-STOCK POLICY
# =========================================================================
class AgeAwareBaseStockPolicy:
    """
    Custom age-aware order-up-to policy for this benchmark.

    This is not a reproduction of the cited perishable-inventory papers or a
    learned projected-inventory-level policy. Its safety factor must be selected
    using validation episodes only.

    At each period t and store i:
      1. S_i(t) = d_hat_{it} + k_alpha * sigma_hat_{it}
      2. IP_i(t) = sum_{a=1}^{tau-1} omega_a * I_{i, t-1, a}  (omega_a = 1 - (a-1)/tau)
      3. q_i^{target}(t) = max(0, min(C_i - sum I_{i, t-1}, S_i(t) - IP_i(t)))
      4. Target deliveries are projected into capacitated 2-opt vehicle tours.
    """

    def __init__(
        self,
        num_stores: int = 10,
        horizon_T: int = 5,
        fleet_size_K: int = 2,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        shelf_life_tau: int = 3,
        service_factor_k_alpha: float = 1.645,
        distance_matrix: np.ndarray | None = None,
    ):
        self.N = num_stores
        self.T = horizon_T
        self.K = fleet_size_K
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.tau = shelf_life_tau
        self.k_alpha = float(service_factor_k_alpha)
        self.dist = distance_matrix if distance_matrix is not None else np.zeros((self.N + 1, self.N + 1))

        # Shelf-life discount weights: older stock has smaller weight in effective position
        # omega_1 = 1.0 (fresh), omega_a decays toward 0 as a approaches tau
        self.omega = np.array([1.0 - (a / float(self.tau)) for a in range(self.tau - 1)], dtype=np.float64)

    def fit_from_validation(self, val_forecasts: np.ndarray, val_actuals: np.ndarray, p_u: float, h: float) -> None:
        """
        Calibrates safety factor k_alpha on validation residuals to match
        the critical fractile alpha = p_u / (p_u + h).
        """
        residuals = val_actuals - val_forecasts
        self.residual_std = float(np.std(residuals))
        target_fractile = min(0.99, max(0.50, p_u / (p_u + h)))
        self.k_alpha = float(norm.ppf(target_fractile))

    def plan_horizon(
        self,
        predicted_demand: np.ndarray,
        residual_std: float | None = None,
        initial_inventory: np.ndarray | None = None,
    ) -> OperationalPolicyResult:
        """
        Executes multi-period base-stock planning over horizon T.

        Args:
            predicted_demand: (N, T) predicted demand.
            residual_std: Estimated standard deviation of forecast errors.
            initial_inventory: (N, tau-1) starting on-hand inventory.

        Returns:
            OperationalPolicyResult containing q_plan and x_plan.
        """
        start_t = time.perf_counter()
        sigma = float(residual_std if residual_std is not None else getattr(self, "residual_std", 5.0))

        q_plan = np.zeros((self.N, self.T), dtype=np.float64)
        x_plan = np.zeros((self.N + 1, self.N + 1, self.K, self.T), dtype=np.float64)

        current_I = np.zeros((self.N, self.tau - 1), dtype=np.float64)
        if initial_inventory is not None:
            current_I[:] = initial_inventory

        for t in range(self.T):
            q_target = np.zeros(self.N, dtype=np.float64)
            for i in range(self.N):
                s_target = predicted_demand[i, t] + self.k_alpha * sigma
                # Age-weighted effective inventory position
                effective_ip = float(np.sum(current_I[i, :] * self.omega))
                on_hand_total = float(np.sum(current_I[i, :]))
                max_room = max(0.0, self.Ci - on_hand_total)
                desired = max(0.0, s_target - effective_ip)
                q_target[i] = min(max_room, desired)

            # Route vehicles under capacity constraint Q
            q_deliv_t, x_routes_t, _ = build_capacitated_routes_2opt(
                q_target=q_target,
                distance_matrix=self.dist,
                fleet_size_K=self.K,
                vehicle_capacity_Q=self.Q,
            )

            q_plan[:, t] = q_deliv_t
            x_plan[:, :, :, t] = x_routes_t

            # Propagate expected inventory state using predicted demand for next period
            I_next, _, _, _ = simulate_perishable_inventory_fifo(
                q_delivered=q_deliv_t[:, None],
                true_demand=predicted_demand[:, t : t + 1],
                tau=self.tau,
                initial_inventory=current_I,
                store_capacity_Ci=self.Ci,
            )
            current_I = I_next[:, 0, :]

        elapsed = time.perf_counter() - start_t
        return OperationalPolicyResult(
            q_plan=q_plan,
            x_plan=x_plan,
            solve_time_sec=elapsed,
            method_name="Tier1_AgeAwareBaseStock",
            metadata={"k_alpha": self.k_alpha, "residual_std": sigma},
        )


class CramaULambdaAdaptation:
    """Validation-calibrated adaptation of the published deliver-up-to rule.

    Crama et al. (2018) define UL_lambda around a target service level and a
    demand quantile over lambda periods, with state updates after observed demand.
    This benchmark implementation borrows that order-up-to rule but adapts it to
    a fixed five-period open-loop comparison: it bootstraps validation residual
    vectors, uses age-structured predicted inventory for plan propagation, and
    sends target deliveries through the shared capacitated routing heuristic.
    It is therefore labeled an adaptation, not a faithful replication.
    """

    def __init__(
        self,
        num_stores: int,
        horizon_T: int,
        fleet_size_K: int,
        vehicle_capacity_Q: float,
        store_capacity_Ci: float,
        shelf_life_tau: int,
        lambda_periods: int,
        target_service_level: float,
        distance_matrix: np.ndarray,
        residuals: np.ndarray,
    ):
        self.N = int(num_stores)
        self.T = int(horizon_T)
        self.K = int(fleet_size_K)
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.tau = int(shelf_life_tau)
        self.lam = int(lambda_periods)
        self.target_service_level = float(target_service_level)
        self.dist = np.asarray(distance_matrix, dtype=np.float64)
        self.residuals = np.asarray(residuals, dtype=np.float64)
        if self.residuals.ndim != 3 or self.residuals.shape[1:] != (self.N, self.T):
            raise ValueError("residuals must have shape (validation_episodes, num_stores, horizon)")
        if not 1 <= self.lam <= min(self.tau, self.T):
            raise ValueError("lambda_periods must be between one and min(shelf_life, horizon)")
        if not 0.5 < self.target_service_level < 1.0:
            raise ValueError("target_service_level must lie between 0.5 and 1.0")

    def plan_horizon(
        self,
        predicted_demand: np.ndarray,
        initial_inventory: np.ndarray | None = None,
    ) -> OperationalPolicyResult:
        start_t = time.perf_counter()
        predictions = np.maximum(0.0, np.asarray(predicted_demand, dtype=np.float64))
        if predictions.shape != (self.N, self.T):
            raise ValueError(f"predicted_demand must have shape ({self.N}, {self.T})")
        q_plan = np.zeros((self.N, self.T), dtype=np.float64)
        x_plan = np.zeros((self.N + 1, self.N + 1, self.K, self.T), dtype=np.float64)
        current_I = np.zeros((self.N, self.tau - 1), dtype=np.float64)
        if initial_inventory is not None:
            current_I[:] = np.asarray(initial_inventory, dtype=np.float64)

        for t in range(self.T):
            q_target = np.zeros(self.N, dtype=np.float64)
            lam_t = min(self.lam, self.T - t)
            for i in range(self.N):
                on_hand = float(np.sum(current_I[i]))
                one_period_demands = np.maximum(
                    0.0, predictions[i, t] + self.residuals[:, i, t]
                )
                probability_already_covered = float(np.mean(one_period_demands <= on_hand))
                if probability_already_covered >= self.target_service_level:
                    continue

                residual_window = self.residuals[:, i, t : t + lam_t]
                cumulative_samples = np.maximum(
                    0.0,
                    predictions[i, t : t + lam_t][None, :] + residual_window,
                ).sum(axis=1)
                target_level = float(
                    np.quantile(cumulative_samples, self.target_service_level, method="higher")
                )
                room = max(0.0, self.Ci - on_hand)
                q_target[i] = min(room, max(0.0, target_level - on_hand))

            q_deliv_t, x_routes_t, _ = build_capacitated_routes_2opt(
                q_target=q_target,
                distance_matrix=self.dist,
                fleet_size_K=self.K,
                vehicle_capacity_Q=self.Q,
            )
            q_plan[:, t] = q_deliv_t
            x_plan[:, :, :, t] = x_routes_t
            I_next, _, _, _ = simulate_perishable_inventory_fifo(
                q_delivered=q_deliv_t[:, None],
                true_demand=predictions[:, t : t + 1],
                tau=self.tau,
                initial_inventory=current_I,
                store_capacity_Ci=self.Ci,
            )
            current_I = I_next[:, 0, :]

        return OperationalPolicyResult(
            q_plan=q_plan,
            x_plan=x_plan,
            solve_time_sec=time.perf_counter() - start_t,
            method_name="Crama_UL_lambda_adaptation",
            metadata={
                "lambda_periods": self.lam,
                "target_service_level": self.target_service_level,
                "num_validation_residual_episodes": int(len(self.residuals)),
                "source_fidelity": "adapted open-loop deliver-up-to rule; not a replication",
            },
        )


# =========================================================================
# EXPLORATORY ROLLING-HORIZON EXTENSION
# =========================================================================
class RollingHorizonMPCPIRP:
    """
    Custom rolling-horizon PIRP extension for this benchmark.

    This code is not a reproduction of the cited MPC methods.

    At each period t in 1..T:
      1. Re-optimizes a lookahead PIRP over window [t, min(t + H - 1, T)] with lookahead H in {2, 3}.
      2. Fixes initial inventory to true realized state I_{i, t-1, a}.
      3. Dispatches immediate period t routes and deliveries.
      4. Observes true demand d_{it}, transitions FIFO cohorts, and rolls forward.
    """

    def __init__(
        self,
        lookahead_H: int = 3,
        num_stores: int = 10,
        horizon_T: int = 5,
        fleet_size_K: int = 2,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        shelf_life_tau: int = 3,
        holding_cost_h: float = 1.0,
        spoilage_penalty_ps: float = 5.0,
        stockout_penalty_pu: float = 20.0,
        distance_matrix: np.ndarray | None = None,
    ):
        self.H = lookahead_H
        self.N = num_stores
        self.T = horizon_T
        self.K = fleet_size_K
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.tau = shelf_life_tau
        self.h = float(holding_cost_h)
        self.ps = float(spoilage_penalty_ps)
        self.pu = float(stockout_penalty_pu)
        self.dist = distance_matrix if distance_matrix is not None else np.zeros((self.N + 1, self.N + 1))

    def plan_open_loop(
        self,
        predicted_demand: np.ndarray,
        initial_inventory: np.ndarray | None = None,
    ) -> OperationalPolicyResult:
        """
        Executes rolling-horizon lookahead optimization under predicted demand.
        """
        start_t = time.perf_counter()
        q_plan = np.zeros((self.N, self.T), dtype=np.float64)
        x_plan = np.zeros((self.N + 1, self.N + 1, self.K, self.T), dtype=np.float64)

        current_I = np.zeros((self.N, self.tau - 1), dtype=np.float64)
        if initial_inventory is not None:
            current_I[:] = initial_inventory

        for t in range(self.T):
            sub_T = min(self.H, self.T - t)
            sub_demand = predicted_demand[:, t : t + sub_T]

            # Build sub-problem PIRP model over window sub_T
            sub_model = PIRPModel(
                num_stores=self.N,
                horizon_T=sub_T,
                fleet_size_K=self.K,
                vehicle_capacity_Q=self.Q,
                store_capacity_Ci=self.Ci,
                shelf_life_tau=self.tau,
                holding_cost_h=self.h,
                spoilage_penalty_ps=self.ps,
                stockout_penalty_pu=self.pu,
                distance_matrix=self.dist,
                continuous_relaxation=False,
            )

            sol = sub_model.solve(sub_demand, time_limit_sec=5.0)

            # Extract immediate decisions for period 0 of sub-window (period t of global horizon)
            q_plan[:, t] = sol.q_total[:, 0]
            x_plan[:, :, :, t] = sol.x_routes[:, :, :, 0]

            # Advance expected inventory state
            I_next, _, _, _ = simulate_perishable_inventory_fifo(
                q_delivered=q_plan[:, t : t + 1],
                true_demand=predicted_demand[:, t : t + 1],
                tau=self.tau,
                initial_inventory=current_I,
                store_capacity_Ci=self.Ci,
            )
            current_I = I_next[:, 0, :]

        elapsed = time.perf_counter() - start_t
        return OperationalPolicyResult(
            q_plan=q_plan,
            x_plan=x_plan,
            solve_time_sec=elapsed,
            method_name="Tier2_RollingHorizonMPC",
            metadata={"lookahead_H": self.H},
        )


# =========================================================================
# EXPLORATORY RESIDUAL-SCENARIO SEARCH (NOT SAA)
# =========================================================================
class ScenarioBasedSAAPIRP:
    """
    Exploratory residual-scenario scaling search; not stochastic-programming SAA.

    It solves a deterministic mean-demand problem and evaluates scaled delivery
    candidates over residual scenarios. It does not enforce scenario-wise
    nonanticipativity in an extensive-form PIRP model and is excluded from primary
    method claims until replaced by a valid stochastic-programming formulation.
    """

    def __init__(
        self,
        num_scenarios_S: int = 10,
        num_stores: int = 10,
        horizon_T: int = 5,
        fleet_size_K: int = 2,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        shelf_life_tau: int = 3,
        holding_cost_h: float = 1.0,
        spoilage_penalty_ps: float = 5.0,
        stockout_penalty_pu: float = 20.0,
        distance_matrix: np.ndarray | None = None,
    ):
        self.S = num_scenarios_S
        self.N = num_stores
        self.T = horizon_T
        self.K = fleet_size_K
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.tau = shelf_life_tau
        self.h = float(holding_cost_h)
        self.ps = float(spoilage_penalty_ps)
        self.pu = float(stockout_penalty_pu)
        self.dist = distance_matrix if distance_matrix is not None else np.zeros((self.N + 1, self.N + 1))
        self.historical_residuals: np.ndarray | None = None

    def fit_residuals(self, val_forecasts: np.ndarray, val_actuals: np.ndarray) -> None:
        """Stores historical validation residuals for bootstrap scenario generation."""
        self.historical_residuals = np.array(val_actuals - val_forecasts, dtype=np.float64)

    def plan_horizon(
        self,
        predicted_demand: np.ndarray,
        initial_inventory: np.ndarray | None = None,
        random_seed: int = 42,
    ) -> OperationalPolicyResult:
        """
        Generates S demand scenarios around predicted demand, solves candidate
        scenario plans, and selects the consensus plan minimizing expected recourse cost.
        """
        start_t = time.perf_counter()
        rng = np.random.default_rng(random_seed)

        # Generate S joint scenarios
        scenarios: list[np.ndarray] = []
        if self.historical_residuals is not None and len(self.historical_residuals) > 0:
            flat_res = self.historical_residuals.flatten()
            for _ in range(self.S):
                noise = rng.choice(flat_res, size=(self.N, self.T), replace=True)
                scen_d = np.maximum(0.0, predicted_demand + noise)
                scenarios.append(scen_d)
        else:
            # Fallback Gaussian residuals
            std_d = np.std(predicted_demand) if np.std(predicted_demand) > 1e-3 else 5.0
            for _ in range(self.S):
                noise = rng.normal(0.0, std_d * 0.25, size=(self.N, self.T))
                scen_d = np.maximum(0.0, predicted_demand + noise)
                scenarios.append(scen_d)

        # Average scenario demand (expected scenario profile)
        mean_demand = np.mean(scenarios, axis=0)

        # Solve base candidate plan on expected scenario demand
        base_model = PIRPModel(
            num_stores=self.N,
            horizon_T=self.T,
            fleet_size_K=self.K,
            vehicle_capacity_Q=self.Q,
            store_capacity_Ci=self.Ci,
            shelf_life_tau=self.tau,
            holding_cost_h=self.h,
            spoilage_penalty_ps=self.ps,
            stockout_penalty_pu=self.pu,
            distance_matrix=self.dist,
            continuous_relaxation=False,
        )
        base_sol = base_model.solve(mean_demand, time_limit_sec=10.0)

        # Evaluate candidate delivery levels and adjust safety stock buffer to minimize expected scenario cost
        best_q = np.array(base_sol.q_total, copy=True)
        best_x = np.array(base_sol.x_routes, copy=True)
        best_expected_cost = float("inf")

        # Explore candidate service scaling factors alpha in {0.9, 1.0, 1.1, 1.2}
        scaling_candidates = [0.90, 1.00, 1.05, 1.10, 1.15, 1.20]
        for factor in scaling_candidates:
            cand_q = np.zeros_like(best_q)
            cand_x = np.zeros_like(best_x)

            for t in range(self.T):
                target_t = np.minimum(self.Ci, base_sol.q_total[:, t] * factor)
                q_deliv_t, x_routes_t, _ = build_capacitated_routes_2opt(
                    q_target=target_t,
                    distance_matrix=self.dist,
                    fleet_size_K=self.K,
                    vehicle_capacity_Q=self.Q,
                )
                cand_q[:, t] = q_deliv_t
                cand_x[:, :, :, t] = x_routes_t

            # Compute expected recourse cost across all S scenarios
            total_scen_cost = 0.0
            for scen in scenarios:
                I_on_hand, y_pos, y_neg, _ = simulate_perishable_inventory_fifo(
                    q_delivered=cand_q,
                    true_demand=scen,
                    tau=self.tau,
                    initial_inventory=initial_inventory,
                    store_capacity_Ci=self.Ci,
                )
                # Compute costs
                h_weights = np.array([self.h + a * 0.01 for a in range(self.tau - 1)], dtype=np.float64)
                c_hold = float(np.sum(I_on_hand * h_weights))
                c_spoil = float(self.ps * np.sum(y_neg))
                c_stock = float(self.pu * np.sum(y_pos))
                c_route = float(np.sum(cand_x * self.dist[:, :, None, None]))
                total_scen_cost += (c_route + c_hold + c_spoil + c_stock)

            exp_cost = total_scen_cost / float(self.S)
            if exp_cost < best_expected_cost:
                best_expected_cost = exp_cost
                best_q = cand_q
                best_x = cand_x

        elapsed = time.perf_counter() - start_t
        return OperationalPolicyResult(
            q_plan=best_q,
            x_plan=best_x,
            solve_time_sec=elapsed,
            method_name="ResidualScenarioScalingSearch_NOT_SAA",
            metadata={"num_scenarios_S": self.S, "expected_scenario_cost": best_expected_cost},
        )


# =========================================================================
# EXPLORATORY RANDOMIZED DESTROY/REPAIR SEARCH (NOT ALNS)
# =========================================================================
class ALNSMatheuristicPIRP:
    """
    Exploratory randomized destroy/repair search; not adaptive large-neighborhood search.

    This routine does not learn operator weights or reproduce the cited methods.
    Its outputs are excluded from primary comparisons until feasibility and source
    fidelity are established.

    Utilizes perishable-aware ruin and recreate operators:
      - Ruin: Random Store Removal, Worst-Cost Removal, High-Spoilage Removal
      - Recreate: Greedy FIFO Insertion, Regret-2 Insertion
      - Acceptance: Simulated Annealing with geometric cooling
    """

    def __init__(
        self,
        max_iterations: int = 100,
        num_stores: int = 10,
        horizon_T: int = 5,
        fleet_size_K: int = 2,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        shelf_life_tau: int = 3,
        holding_cost_h: float = 1.0,
        spoilage_penalty_ps: float = 5.0,
        stockout_penalty_pu: float = 20.0,
        distance_matrix: np.ndarray | None = None,
    ):
        self.max_iter = max_iterations
        self.N = num_stores
        self.T = horizon_T
        self.K = fleet_size_K
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.tau = shelf_life_tau
        self.h = float(holding_cost_h)
        self.ps = float(spoilage_penalty_ps)
        self.pu = float(stockout_penalty_pu)
        self.dist = distance_matrix if distance_matrix is not None else np.zeros((self.N + 1, self.N + 1))

    def _evaluate_candidate(
        self,
        q_plan: np.ndarray,
        predicted_demand: np.ndarray,
        initial_inventory: np.ndarray | None,
    ) -> tuple[float, np.ndarray, float, float, float, float]:
        """Evaluates operational cost of a candidate delivery plan."""
        x_plan = np.zeros((self.N + 1, self.N + 1, self.K, self.T), dtype=np.float64)
        c_route = 0.0

        for t in range(self.T):
            _, x_t, r_cost_t = build_capacitated_routes_2opt(
                q_target=q_plan[:, t],
                distance_matrix=self.dist,
                fleet_size_K=self.K,
                vehicle_capacity_Q=self.Q,
            )
            x_plan[:, :, :, t] = x_t
            c_route += r_cost_t

        I_on_hand, y_pos, y_neg, _ = simulate_perishable_inventory_fifo(
            q_delivered=q_plan,
            true_demand=predicted_demand,
            tau=self.tau,
            initial_inventory=initial_inventory,
            store_capacity_Ci=self.Ci,
        )

        h_weights = np.array([self.h + a * 0.01 for a in range(self.tau - 1)], dtype=np.float64)
        c_hold = float(np.sum(I_on_hand * h_weights))
        c_spoil = float(self.ps * np.sum(y_neg))
        c_stock = float(self.pu * np.sum(y_pos))
        total_cost = c_route + c_hold + c_spoil + c_stock

        return total_cost, x_plan, c_route, c_hold, c_spoil, c_stock

    def plan_horizon(
        self,
        predicted_demand: np.ndarray,
        initial_inventory: np.ndarray | None = None,
        random_seed: int = 42,
    ) -> OperationalPolicyResult:
        """
        Executes fixed-probability randomized destroy-and-repair iterations.
        """
        start_t = time.perf_counter()
        rng = np.random.default_rng(random_seed)

        # Initial solution: myopic unconstrained base-stock target
        curr_q = np.minimum(self.Ci, np.maximum(0.0, predicted_demand))
        curr_cost, curr_x, _, _, _, _ = self._evaluate_candidate(curr_q, predicted_demand, initial_inventory)

        best_q = np.array(curr_q, copy=True)
        best_x = np.array(curr_x, copy=True)
        best_cost = curr_cost

        # Simulated Annealing temperature schedule
        temp = curr_cost * 0.15
        cooling_rate = 0.95

        for it in range(self.max_iter):
            cand_q = np.array(curr_q, copy=True)

            # Choose destroy operator: 0 = Random, 1 = Worst-Cost, 2 = High-Spoilage
            destroy_op = rng.choice([0, 1, 2])
            num_destroy = rng.integers(1, max(2, self.N // 2))

            if destroy_op == 0:
                # Random removal: clear deliveries for selected stores
                stores_to_remove = rng.choice(self.N, size=num_destroy, replace=False)
                periods_to_remove = rng.choice(self.T, size=num_destroy, replace=True)
                for s_idx, t_idx in zip(stores_to_remove, periods_to_remove):
                    cand_q[s_idx, t_idx] = 0.0

            elif destroy_op == 1:
                # Worst-cost removal: target deliveries with high distance from depot
                dist_scores = [self.dist[0, i + 1] * np.sum(cand_q[i, :]) for i in range(self.N)]
                worst_stores = np.argsort(dist_scores)[-num_destroy:]
                for s_idx in worst_stores:
                    cand_q[s_idx, :] *= 0.50

            elif destroy_op == 2:
                # High-spoilage removal: simulate to find stores with spoilage and reduce deliveries
                _, _, y_neg_curr, _ = simulate_perishable_inventory_fifo(
                    q_delivered=cand_q,
                    true_demand=predicted_demand,
                    tau=self.tau,
                    initial_inventory=initial_inventory,
                    store_capacity_Ci=self.Ci,
                )
                spoil_stores = np.where(np.sum(y_neg_curr, axis=1) > 1e-4)[0]
                if len(spoil_stores) > 0:
                    for s_idx in spoil_stores[:num_destroy]:
                        cand_q[s_idx, :] = np.maximum(0.0, cand_q[s_idx, :] * 0.80)
                else:
                    # Minor perturbation if no spoilage
                    rand_i = rng.integers(0, self.N)
                    rand_t = rng.integers(0, self.T)
                    cand_q[rand_i, rand_t] *= 0.70

            # Recreate operator: Greedy FIFO replenishment
            for t in range(self.T):
                unmet_est = np.maximum(0.0, predicted_demand[:, t] - cand_q[:, t])
                priority = np.argsort(-unmet_est)
                for s_idx in priority:
                    room = max(0.0, self.Ci - cand_q[s_idx, t])
                    if room > 1e-6 and unmet_est[s_idx] > 1e-6:
                        cand_q[s_idx, t] += min(room, unmet_est[s_idx] * 1.10)

            # Evaluate candidate solution
            cand_cost, cand_x, _, _, _, _ = self._evaluate_candidate(cand_q, predicted_demand, initial_inventory)

            # Metropolis acceptance criterion
            delta = cand_cost - curr_cost
            if delta < 0 or rng.uniform(0.0, 1.0) < math.exp(-delta / max(1e-4, temp)):
                curr_q = cand_q
                curr_x = cand_x
                curr_cost = cand_cost

                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_q = np.array(curr_q, copy=True)
                    best_x = np.array(curr_x, copy=True)

            temp *= cooling_rate

        elapsed = time.perf_counter() - start_t
        return OperationalPolicyResult(
            q_plan=best_q,
            x_plan=best_x,
            solve_time_sec=elapsed,
            method_name="RandomizedDestroyRepair_NOT_ALNS",
            metadata={"iterations": self.max_iter, "best_internal_cost": best_cost},
        )
