"""Exploratory finite-scenario robust PIRP adaptation on the frozen test set.

This add-on does not overwrite the revision2 primary run. It trains the same
point LightGBM forecaster, builds a validation-only residual scenario bank,
selects one fixed plan per test episode by empirical worst-case FIFO cost, and
reports descriptive paired contrasts against the already-frozen point-PTO and
DFL results. The new comparator is post hoc and all contrasts are exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_benchmark import load_dataset, structure_instances
from src.models.baselines import LightGBMPTO
from src.models.empirical_robust_pirp import EmpiricalRobustPIRP
from src.utils.metrics import (
    compute_forecast_accuracy,
    evaluate_decisions_under_true_demand,
    simulate_perishable_inventory_fifo,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    data_dir: str = "data/revision2_final/processed",
    reference_dir: str = "results/revision2_final",
    output_dir: str = "results/revision2_robust_adaptation_exploratory",
    solver_time_limit_sec: float = 0.5,
    mip_rel_gap: float = 0.01,
    num_eval_episodes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    reference_path = Path(reference_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test, config, distance, features = load_dataset(
        data_dir=str(data_path),
        network_path=str(data_path / "network_solomon_c1.json"),
    )
    manifest = json.loads((data_path / "episode_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("independent_episode_rng") or manifest.get("m5_data_used"):
        raise ValueError("Expected the frozen independent synthetic dataset with no M5 data")
    X_train, _ = structure_instances(df_train, features, config["num_stores"], config["horizon_T"])
    X_val, d_val = structure_instances(df_val, features, config["num_stores"], config["horizon_T"])
    X_test, d_test = structure_instances(df_test, features, config["num_stores"], config["horizon_T"])
    val_ids = sorted(df_val.instance_id.unique().astype(int).tolist())
    test_ids = sorted(df_test.instance_id.unique().astype(int).tolist())
    if num_eval_episodes is not None:
        X_test, d_test, test_ids = X_test[:num_eval_episodes], d_test[:num_eval_episodes], test_ids[:num_eval_episodes]

    model = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05)
    model.fit(df_train[features].to_numpy(), df_train["demand"].to_numpy())
    pred_val = np.maximum(0.0, model.predict(X_val))
    pred_test = np.maximum(0.0, model.predict(X_test))
    residual_paths = d_val - pred_val

    policy = EmpiricalRobustPIRP(
        residual_paths=residual_paths,
        distance_matrix=distance,
        shelf_life_tau=config["default_shelf_life_tau"],
        holding_cost_h=config["holding_cost_h"],
        aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
        spoilage_penalty_ps=config["spoilage_penalty_ps"],
        stockout_penalty_pu=config["default_stockout_penalty_pu"],
        vehicle_capacity_Q=config["vehicle_capacity_Q"],
        store_capacity_Ci=config["store_capacity_Ci"],
        time_limit_sec=solver_time_limit_sec,
        mip_rel_gap=mip_rel_gap,
    )

    records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for idx, episode_id in enumerate(test_ids):
        result = policy.plan(pred_test[idx])
        metrics = evaluate_decisions_under_true_demand(
            q_plan=result.q_plan,
            x_plan=result.x_plan,
            true_demand=d_test[idx],
            distance_matrix=distance,
            h=config["holding_cost_h"],
            aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
            p_s=config["spoilage_penalty_ps"],
            p_u=config["default_stockout_penalty_pu"],
            tau=config["default_shelf_life_tau"],
            store_capacity_Ci=config["store_capacity_Ci"],
            vehicle_capacity_Q=config["vehicle_capacity_Q"],
        )

        selected_diag = result.candidate_diagnostics[result.selected_candidate]
        plan_demand = np.maximum(
            0.0,
            pred_test[idx] + residual_paths[selected_diag["residual_scenario_index"]],
        )
        sim_I, sim_pos, sim_neg, sim_w = simulate_perishable_inventory_fifo(
            q_delivered=result.q_plan,
            true_demand=plan_demand,
            tau=config["default_shelf_life_tau"],
            store_capacity_Ci=config["store_capacity_Ci"],
        )
        parity_error = max(
            float(np.max(np.abs(sim_I - result.selected_solution.I_inventory))),
            float(np.max(np.abs(sim_pos - result.selected_solution.y_pos))),
            float(np.max(np.abs(sim_neg - result.selected_solution.y_neg))),
            float(np.max(np.abs(sim_w - result.selected_solution.w_consumed))),
        )
        if parity_error > 1e-4:
            raise RuntimeError(f"Robust-plan/FIFO parity failed for episode {episode_id}: {parity_error}")

        accuracy = compute_forecast_accuracy(pred_test[idx], d_test[idx])
        records.append({
            "method": "Empirical_Robust_PIRP_adaptation",
            "episode_id": int(episode_id),
            "training_seed": 0,
            "total_cost": metrics["total_cost"],
            **{key: value for key, value in metrics.items() if key != "total_cost"},
            "forecast_mae": accuracy["mae"],
            "forecast_rmse": accuracy["rmse"],
            "selected_candidate": result.selected_candidate,
            "selected_residual_scenario_index": selected_diag["residual_scenario_index"],
            "empirical_scenario_worst_case_cost": result.worst_case_cost,
            "empirical_scenario_mean_cost": result.mean_scenario_cost,
            "solver_time_sec": result.solve_time_sec,
            "solver_candidate_count": len(result.candidate_diagnostics),
            "solver_optimal_candidate_count": sum(bool(row["solver_optimal"]) for row in result.candidate_diagnostics),
            "solver_fallback_candidate_count": sum(bool(row["solver_used_heuristic_fallback"]) for row in result.candidate_diagnostics),
            "solver_fifo_parity_max_abs_error": max(parity_error, result.max_fifo_parity_error),
        })
        for candidate in result.candidate_diagnostics:
            candidate_records.append({"episode_id": int(episode_id), **candidate})

    episode_metrics = pd.DataFrame(records)
    episode_metrics.to_csv(out_path / "episode_metrics.csv", index=False)
    pd.DataFrame(candidate_records).to_json(
        out_path / "candidate_diagnostics.jsonl", orient="records", lines=True
    )

    summary: dict[str, Any] = {
        "method": "Empirical_Robust_PIRP_adaptation",
        "n_test_episodes": int(episode_metrics.episode_id.nunique()),
        "n_validation_residual_scenarios": int(len(residual_paths)),
        "candidate_quantiles": list(policy.candidate_quantiles),
        "mean_total_cost": float(episode_metrics.total_cost.mean()),
        "sd_episode_total_cost": float(episode_metrics.total_cost.std(ddof=1)),
        "mean_routing_cost": float(episode_metrics.routing_cost.mean()),
        "mean_holding_cost": float(episode_metrics.holding_cost.mean()),
        "mean_expiry_cost": float(episode_metrics.expiry_cost.mean()),
        "mean_capacity_disposal_cost": float(episode_metrics.capacity_disposal_cost.mean()),
        "mean_stockout_cost": float(episode_metrics.stockout_cost.mean()),
        "mean_fill_rate": float(episode_metrics.fill_rate.mean()),
        "mean_solver_time_sec": float(episode_metrics.solver_time_sec.mean()),
        "optimal_candidate_fraction": float(
            episode_metrics.solver_optimal_candidate_count.sum()
            / episode_metrics.solver_candidate_count.sum()
        ),
        "fallback_candidate_fraction": float(
            episode_metrics.solver_fallback_candidate_count.sum()
            / episode_metrics.solver_candidate_count.sum()
        ),
        "max_fifo_parity_error": float(episode_metrics.solver_fifo_parity_max_abs_error.max()),
        "max_candidate_fifo_parity_error": float(
            pd.DataFrame(candidate_records)["fifo_parity_max_abs_error"].max()
        ),
    }
    pd.DataFrame([summary]).to_csv(out_path / "summary.csv", index=False)

    comparisons = []
    ref_metrics_path = reference_path / "episode_metrics.csv"
    if ref_metrics_path.exists():
        reference = pd.read_csv(ref_metrics_path)
        reference = reference.groupby(["method", "episode_id"], as_index=False).mean(numeric_only=True)
        robust_cost = episode_metrics.set_index("episode_id").total_cost
        for comparator in ("LightGBM_PTO", "Two-Phase_DFL"):
            comp = reference[reference.method == comparator].set_index("episode_id").total_cost
            common = sorted(set(robust_cost.index) & set(comp.index))
            delta = robust_cost.loc[common].to_numpy() - comp.loc[common].to_numpy()
            rng = np.random.default_rng(20261027 + len(comparisons))
            draws = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            comparisons.append({
                "outcome": "total_cost",
                "comparison": f"Empirical robust adaptation minus {comparator}",
                "n_paired_test_episodes": len(common),
                "mean_paired_difference": float(np.mean(delta)),
                "bootstrap_95ci_low": float(lo),
                "bootstrap_95ci_high": float(hi),
                "status": "exploratory; comparator added after the revision2 pilot outcomes were inspected",
            })
    pd.DataFrame(comparisons).to_csv(out_path / "exploratory_paired_comparisons.csv", index=False)

    run_manifest = {
        "protocol_version": "posthoc-finite-scenario-robust-adaptation-2026-10-27",
        "dataset_directory": str(data_path.resolve()),
        "status": "exploratory comparator add-on; not preregistered; not a source-faithful He et al. (2024) reproduction",
        "method": "One episode-start plan is selected from five deterministic PIRP candidate plans. Candidate demand paths use whole validation residual episodes at aggregate residual quantiles 0, .25, .50, .75, and 1. The candidate with the lowest maximum FIFO cost over every validation residual path is selected.",
        "source_scope": "Problem-class and robust-decision motivation is related to He et al. (2024), but this code does not implement their uncertainty-set robust counterpart, Benders decomposition, or ILS-BD.",
        "information_timing": "Episode-start features only; validation residual paths form the uncertainty set; no test demand enters plan generation or selection; test actuals are used only by the common FIFO evaluator.",
        "nonanticipativity": "Each candidate has one fixed delivery-and-route schedule shared across every scenario used to score that candidate.",
        "training_episode_ids": sorted(df_train.instance_id.unique().astype(int).tolist()),
        "validation_episode_ids": val_ids,
        "test_episode_ids": test_ids,
        "solver": {
            "name": "SciPy HiGHS MILP with feasible construction fallback",
            "time_limit_sec": solver_time_limit_sec,
            "mip_rel_gap": mip_rel_gap,
            "optimality_assumed": False,
        },
        "input_hashes": {
            path.name: _sha256(path)
            for path in sorted(data_path.glob("*"))
            if path.is_file()
        },
        "reference_run_manifest_sha256": _sha256(reference_path / "run_manifest.json") if (reference_path / "run_manifest.json").exists() else None,
        "result_audit": {
            "all_test_episodes_have_one_plan": bool(len(episode_metrics) == len(test_ids)),
            "max_fifo_parity_error": summary["max_fifo_parity_error"],
            "max_candidate_fifo_parity_error": summary["max_candidate_fifo_parity_error"],
            "fallback_candidate_count": int(episode_metrics.solver_fallback_candidate_count.sum()),
            "total_candidate_count": int(episode_metrics.solver_candidate_count.sum()),
        },
    }
    (out_path / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return episode_metrics, pd.DataFrame([summary])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run post-hoc empirical robust PIRP comparator")
    parser.add_argument("--data-dir", default="data/revision2_final/processed")
    parser.add_argument("--reference-dir", default="results/revision2_final")
    parser.add_argument("--output-dir", default="results/revision2_robust_adaptation_exploratory")
    parser.add_argument("--solver-time-limit-sec", type=float, default=0.5)
    parser.add_argument("--mip-rel-gap", type=float, default=0.01)
    parser.add_argument("--num-eval-episodes", type=int)
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        solver_time_limit_sec=args.solver_time_limit_sec,
        mip_rel_gap=args.mip_rel_gap,
        num_eval_episodes=args.num_eval_episodes,
    )
