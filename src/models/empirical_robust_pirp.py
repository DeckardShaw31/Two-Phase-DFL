"""Finite-scenario robust plan selection for the revised PIRP benchmark.

This is a transparent operational adaptation, not a reproduction of He et al.
(2024)'s robust-counterpart reformulation, Benders decomposition, or ILS-BD.
It constructs a finite uncertainty set from validation-only forecast residual
paths, generates feasible plans by solving deterministic PIRPs at five
residual quantiles, and selects the single episode-start plan with the lowest
worst-case FIFO cost over the full validation residual set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.optimization.pirp_model import PIRPSolution, solve_pirp
from src.utils.metrics import (
    evaluate_decisions_under_true_demand,
    simulate_perishable_inventory_fifo,
)


@dataclass
class RobustPlanSelection:
    q_plan: np.ndarray
    x_plan: np.ndarray
    solve_time_sec: float
    selected_candidate: int
    worst_case_cost: float
    mean_scenario_cost: float
    max_fifo_parity_error: float
    candidate_diagnostics: list[dict[str, Any]]
    selected_solution: PIRPSolution


class EmpiricalRobustPIRP:
    """Select a common fixed plan against validation-derived residual paths.

    Validation residual paths retain cross-store and cross-period dependence.
    Candidate demand paths use residual episodes spanning the empirical total-
    residual distribution. All candidates are evaluated on the same scenario
    bank. One plan is then selected and held fixed for the test episode, so the
    comparator uses the same episode-start decision timing as fixed-plan PTO.
    """

    candidate_quantiles = (0.0, 0.25, 0.50, 0.75, 1.0)

    def __init__(
        self,
        residual_paths: np.ndarray,
        distance_matrix: np.ndarray,
        *,
        shelf_life_tau: int,
        holding_cost_h: float,
        aging_surcharge_eps: float,
        spoilage_penalty_ps: float,
        stockout_penalty_pu: float,
        vehicle_capacity_Q: float,
        store_capacity_Ci: float,
        time_limit_sec: float,
        mip_rel_gap: float,
    ):
        residual_paths = np.asarray(residual_paths, dtype=np.float64)
        if residual_paths.ndim != 3 or residual_paths.shape[0] < 2:
            raise ValueError("residual_paths must have shape (scenarios, stores, periods), with at least two scenarios")
        if not np.all(np.isfinite(residual_paths)):
            raise ValueError("residual_paths must contain only finite values")
        self.residual_paths = residual_paths
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.N, self.T = residual_paths.shape[1:]
        if self.distance_matrix.shape != (self.N + 1, self.N + 1):
            raise ValueError("distance_matrix dimensions do not match residual_paths")
        self.tau = int(shelf_life_tau)
        self.h = float(holding_cost_h)
        self.eps = float(aging_surcharge_eps)
        self.ps = float(spoilage_penalty_ps)
        self.pu = float(stockout_penalty_pu)
        self.Q = float(vehicle_capacity_Q)
        self.Ci = float(store_capacity_Ci)
        self.time_limit_sec = float(time_limit_sec)
        self.mip_rel_gap = float(mip_rel_gap)
        self._candidate_residual_indices = self._select_candidate_indices()

    def _select_candidate_indices(self) -> list[int]:
        """Select whole validation paths at prespecified aggregate quantiles."""
        totals = self.residual_paths.sum(axis=(1, 2))
        order = np.argsort(totals, kind="stable")
        count = len(order)
        return [int(order[round(q * (count - 1))]) for q in self.candidate_quantiles]

    def _cost(self, q: np.ndarray, x: np.ndarray, demand: np.ndarray) -> float:
        metrics = evaluate_decisions_under_true_demand(
            q_plan=q,
            x_plan=x,
            true_demand=demand,
            distance_matrix=self.distance_matrix,
            h=self.h,
            aging_surcharge_eps=self.eps,
            p_s=self.ps,
            p_u=self.pu,
            tau=self.tau,
            store_capacity_Ci=self.Ci,
            vehicle_capacity_Q=self.Q,
        )
        return float(metrics["total_cost"])

    def plan(self, point_forecast: np.ndarray) -> RobustPlanSelection:
        """Return the lowest worst-case-cost plan among five scenario plans."""
        start = time.perf_counter()
        forecast = np.maximum(0.0, np.asarray(point_forecast, dtype=np.float64))
        if forecast.shape != (self.N, self.T):
            raise ValueError(f"point_forecast must have shape ({self.N}, {self.T})")

        scenario_demands = np.maximum(
            0.0,
            forecast[None, :, :] + self.residual_paths,
        )
        candidate_rows: list[dict[str, Any]] = []
        candidate_solutions: list[PIRPSolution] = []
        for candidate_id, residual_idx in enumerate(self._candidate_residual_indices):
            candidate_demand = np.maximum(
                0.0,
                forecast + self.residual_paths[residual_idx],
            )
            solution = solve_pirp(
                demand=candidate_demand,
                distance_matrix=self.distance_matrix,
                continuous_relaxation=False,
                shelf_life_tau=self.tau,
                holding_cost_h=self.h,
                aging_surcharge_eps=self.eps,
                stockout_penalty_pu=self.pu,
                spoilage_penalty_ps=self.ps,
                vehicle_capacity_Q=self.Q,
                store_capacity_Ci=self.Ci,
                time_limit_sec=self.time_limit_sec,
                mip_rel_gap=self.mip_rel_gap,
                use_milp=True,
            )
            if not solution.success:
                raise RuntimeError(
                    f"No feasible robust candidate {candidate_id} for deterministic scenario {residual_idx}: {solution.message}"
                )
            sim_I, sim_pos, sim_neg, sim_w = simulate_perishable_inventory_fifo(
                q_delivered=solution.q_total,
                true_demand=candidate_demand,
                tau=self.tau,
                store_capacity_Ci=self.Ci,
            )
            parity_error = max(
                float(np.max(np.abs(sim_I - solution.I_inventory))),
                float(np.max(np.abs(sim_pos - solution.y_pos))),
                float(np.max(np.abs(sim_neg - solution.y_neg))),
                float(np.max(np.abs(sim_w - solution.w_consumed))),
            )
            if parity_error > 1e-4:
                raise RuntimeError(
                    f"Candidate {candidate_id} failed FIFO parity: max absolute error {parity_error}"
                )
            costs = [
                self._cost(solution.q_total, solution.x_routes, scenario)
                for scenario in scenario_demands
            ]
            row = {
                "candidate_id": candidate_id,
                "residual_quantile": self.candidate_quantiles[candidate_id],
                "residual_scenario_index": residual_idx,
                "worst_case_cost": float(np.max(costs)),
                "mean_scenario_cost": float(np.mean(costs)),
                "scenario_costs": costs,
                "solver_name": solution.solver_name,
                "solver_status": solution.status,
                "solver_optimal": solution.is_optimal,
                "solver_mip_gap": solution.mip_gap,
                "solver_used_heuristic_fallback": solution.used_heuristic_fallback,
                "solver_time_sec": solution.solve_time_sec,
                "fifo_parity_max_abs_error": parity_error,
            }
            candidate_rows.append(row)
            candidate_solutions.append(solution)

        selected_idx = min(
            range(len(candidate_rows)),
            key=lambda idx: (candidate_rows[idx]["worst_case_cost"], candidate_rows[idx]["mean_scenario_cost"], idx),
        )
        chosen = candidate_solutions[selected_idx]
        selected = candidate_rows[selected_idx]
        return RobustPlanSelection(
            q_plan=chosen.q_total,
            x_plan=chosen.x_routes,
            solve_time_sec=time.perf_counter() - start,
            selected_candidate=selected_idx,
            worst_case_cost=float(selected["worst_case_cost"]),
            mean_scenario_cost=float(selected["mean_scenario_cost"]),
            max_fifo_parity_error=max(float(row["fifo_parity_max_abs_error"]) for row in candidate_rows),
            candidate_diagnostics=candidate_rows,
            selected_solution=chosen,
        )
