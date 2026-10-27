"""Reproducible primary experiment for the revised synthetic PIRP manuscript.

All decisions are made once at the start of a five-period episode from the
episode-start feature set. No test demand or test residuals enter training,
tuning, scenario construction, or decisions. Every schedule is evaluated by the
same FIFO simulator against the same held-out demand episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_benchmark import load_dataset, structure_instances
from src.models.baselines import LightGBMPTO, LightGBMQuantilePTO
from src.models.dfl_models import DFLPipeline
from src.models.operational_baselines import CramaULambdaAdaptation
from src.optimization.pirp_model import solve_pirp
from src.utils.metrics import (
    compute_forecast_accuracy,
    evaluate_decisions_under_true_demand,
    simulate_perishable_inventory_fifo,
)


PRIMARY_COMPARISONS = ["LightGBM_PTO", "Crama_UL_lambda_adaptation"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_record(
    method: str,
    episode_id: int,
    seed: int,
    predictions: np.ndarray,
    true_demand: np.ndarray,
    distance_matrix: np.ndarray,
    config: dict[str, Any],
    train_time_sec: float,
    q_plan: np.ndarray | None = None,
    x_plan: np.ndarray | None = None,
    solver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = evaluate_decisions_under_true_demand(
        q_plan=q_plan,
        x_plan=x_plan,
        true_demand=true_demand,
        distance_matrix=distance_matrix,
        h=config["holding_cost_h"],
        aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
        p_s=config["spoilage_penalty_ps"],
        p_u=config["default_stockout_penalty_pu"],
        tau=config["default_shelf_life_tau"],
        store_capacity_Ci=config["store_capacity_Ci"],
        vehicle_capacity_Q=config["vehicle_capacity_Q"],
    )
    accuracy = compute_forecast_accuracy(predictions, true_demand)
    record: dict[str, Any] = {
        "method": method,
        "episode_id": int(episode_id),
        "training_seed": int(seed),
        "train_time_sec": float(train_time_sec),
        **metrics,
        "forecast_mae": accuracy["mae"],
        "forecast_rmse": accuracy["rmse"],
    }
    if solver:
        record.update({f"solver_{key}": value for key, value in solver.items()})
    return record


def _evaluate_predictor(
    method: str,
    predictions: np.ndarray,
    actuals: np.ndarray,
    episode_ids: list[int],
    distance_matrix: np.ndarray,
    config: dict[str, Any],
    train_time_sec: float,
    mip_rel_gap: float,
    solver_time_limit_sec: float,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, episode_id in enumerate(episode_ids):
        sol = solve_pirp(
            demand=np.maximum(0.0, predictions[idx]),
            distance_matrix=distance_matrix,
            continuous_relaxation=False,
            shelf_life_tau=config["default_shelf_life_tau"],
            holding_cost_h=config["holding_cost_h"],
            aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
            stockout_penalty_pu=config["default_stockout_penalty_pu"],
            spoilage_penalty_ps=config["spoilage_penalty_ps"],
            vehicle_capacity_Q=config["vehicle_capacity_Q"],
            store_capacity_Ci=config["store_capacity_Ci"],
            time_limit_sec=solver_time_limit_sec,
            mip_rel_gap=mip_rel_gap,
            use_milp=True,
        )
        if not sol.success:
            raise RuntimeError(
                f"No feasible PIRP plan for {method}, episode {episode_id}: {sol.message}"
            )
        sim_I, sim_y_pos, sim_y_neg, sim_w = simulate_perishable_inventory_fifo(
            q_delivered=sol.q_total,
            true_demand=np.maximum(0.0, predictions[idx]),
            tau=config["default_shelf_life_tau"],
            store_capacity_Ci=config["store_capacity_Ci"],
        )
        parity_error = max(
            float(np.max(np.abs(sim_I - sol.I_inventory))),
            float(np.max(np.abs(sim_y_pos - sol.y_pos))),
            float(np.max(np.abs(sim_y_neg - sol.y_neg))),
            float(np.max(np.abs(sim_w - sol.w_consumed))),
        )
        if parity_error > 1e-4:
            raise RuntimeError(
                f"MILP/FIFO parity failed for {method}, episode {episode_id}: max error {parity_error}"
            )
        records.append(
            _policy_record(
                method=method,
                episode_id=episode_id,
                seed=seed,
                predictions=predictions[idx],
                true_demand=actuals[idx],
                distance_matrix=distance_matrix,
                config=config,
                train_time_sec=train_time_sec,
                q_plan=sol.q_total,
                x_plan=sol.x_routes,
                solver={
                    "name": sol.solver_name,
                    "status": sol.status,
                    "message": sol.message,
                    "optimal": sol.is_optimal,
                    "mip_gap": sol.mip_gap,
                    "mip_dual_bound": sol.mip_dual_bound,
                    "mip_node_count": sol.mip_node_count,
                    "time_sec": sol.solve_time_sec,
                    "used_heuristic_fallback": sol.used_heuristic_fallback,
                    "fifo_parity_max_abs_error": parity_error,
                },
            )
        )
    return records


def _evaluate_ul_lambda(
    policy: CramaULambdaAdaptation,
    method: str,
    predictions: np.ndarray,
    actuals: np.ndarray,
    episode_ids: list[int],
    distance_matrix: np.ndarray,
    config: dict[str, Any],
    seed: int,
    validation_tuning_time_sec: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, episode_id in enumerate(episode_ids):
        plan = policy.plan_horizon(predictions[idx])
        record = _policy_record(
            method=method,
            episode_id=episode_id,
            seed=seed,
            predictions=predictions[idx],
            true_demand=actuals[idx],
            distance_matrix=distance_matrix,
            config=config,
        train_time_sec=0.0,
            q_plan=plan.q_plan,
            x_plan=plan.x_plan,
            solver={
                "name": "nearest-neighbor packing + 2-opt routing heuristic",
                "optimal": False,
                "mip_gap": None,
                "mip_dual_bound": None,
                "mip_node_count": None,
                "time_sec": plan.solve_time_sec,
                "used_heuristic_fallback": False,
            },
        )
        record["validation_tuning_time_sec"] = float(validation_tuning_time_sec)
        records.append(record)
    return records


def _tune_ul_lambda(
    residuals: np.ndarray,
    val_predictions: np.ndarray,
    val_actuals: np.ndarray,
    val_ids: list[int],
    distance_matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[CramaULambdaAdaptation, pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    best_policy: CramaULambdaAdaptation | None = None
    best_mean_cost = float("inf")
    max_lambda = min(config["default_shelf_life_tau"], config["horizon_T"])
    for lam in range(1, max_lambda + 1):
        for target_level in (0.80, 0.90, 0.95, 0.98):
            policy = CramaULambdaAdaptation(
                num_stores=config["num_stores"],
                horizon_T=config["horizon_T"],
                fleet_size_K=config["fleet_size_K"],
                vehicle_capacity_Q=config["vehicle_capacity_Q"],
                store_capacity_Ci=config["store_capacity_Ci"],
                shelf_life_tau=config["default_shelf_life_tau"],
                lambda_periods=lam,
                target_service_level=target_level,
                distance_matrix=distance_matrix,
                residuals=residuals,
            )
            costs: list[float] = []
            for idx in range(len(val_actuals)):
                plan = policy.plan_horizon(val_predictions[idx])
                metrics = evaluate_decisions_under_true_demand(
                    q_plan=plan.q_plan,
                    x_plan=plan.x_plan,
                    true_demand=val_actuals[idx],
                    distance_matrix=distance_matrix,
                    h=config["holding_cost_h"],
                    aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
                    p_s=config["spoilage_penalty_ps"],
                    p_u=config["default_stockout_penalty_pu"],
                    tau=config["default_shelf_life_tau"],
                    store_capacity_Ci=config["store_capacity_Ci"],
                    vehicle_capacity_Q=config["vehicle_capacity_Q"],
                )
                costs.append(metrics["total_cost"])
            mean_cost = float(np.mean(costs))
            candidates.append({
                "lambda_periods": lam,
                "target_service_level": target_level,
                "validation_mean_cost": mean_cost,
                "validation_episode_count": len(val_actuals),
            })
            if mean_cost < best_mean_cost:
                best_mean_cost = mean_cost
                best_policy = policy

    assert best_policy is not None
    return best_policy, pd.DataFrame(candidates)


def _holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _paired_inference(
    episode_metrics: pd.DataFrame,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20261029,
) -> pd.DataFrame:
    per_episode = (
        episode_metrics.groupby(["method", "episode_id"], as_index=False)
        .mean(numeric_only=True)
    )
    ours = per_episode[per_episode.method == "Two-Phase_DFL"].set_index("episode_id")
    test_rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for comparator in PRIMARY_COMPARISONS:
        other = per_episode[per_episode.method == comparator].set_index("episode_id")
        common = sorted(set(ours.index) & set(other.index))
        differences = ours.loc[common, "total_cost"].to_numpy() - other.loc[common, "total_cost"].to_numpy()
        if len(differences) < 2:
            test_rows.append({
                "outcome": "total_cost",
                "comparison": f"Two-Phase_DFL minus {comparator}",
                "n_independent_episodes": len(common),
                "mean_paired_difference": float(np.mean(differences)) if len(differences) else np.nan,
                "paired_sd": np.nan,
                "bootstrap_95ci_low": np.nan,
                "bootstrap_95ci_high": np.nan,
                "paired_t_p_raw": np.nan,
                "holm_adjusted_p": np.nan,
                "primary_family": "two prespecified total-cost comparisons",
                "inference_status": "descriptive only; at least two paired episodes are required",
            })
            continue
        t_result = stats.ttest_1samp(differences, popmean=0.0)
        p_value = float(t_result.pvalue)
        raw_p.append(p_value)
        rng = np.random.default_rng(bootstrap_seed + len(test_rows))
        draws = rng.choice(differences, size=(bootstrap_replicates, len(differences)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        test_rows.append({
            "outcome": "total_cost",
            "comparison": f"Two-Phase_DFL minus {comparator}",
            "n_independent_episodes": len(common),
            "mean_paired_difference": float(np.mean(differences)),
            "paired_sd": float(np.std(differences, ddof=1)),
            "bootstrap_95ci_low": float(lo),
            "bootstrap_95ci_high": float(hi),
            "paired_t_p_raw": p_value,
            "inference_status": "paired inference computed",
        })
    valid_indices = [idx for idx, value in enumerate(raw_p) if np.isfinite(value)]
    adjusted = _holm_adjust([raw_p[idx] for idx in valid_indices])
    for idx, p_adj in zip(valid_indices, adjusted, strict=True):
        test_rows[idx]["holm_adjusted_p"] = p_adj
        test_rows[idx]["primary_family"] = "two prespecified total-cost comparisons"
    return pd.DataFrame(test_rows)


def run_revision_benchmark(
    data_dir: str = "data/revision2_final/processed",
    output_dir: str = "results/revision2_final",
    dfl_seeds: tuple[int, ...] = (101, 202, 303),
    dfl_epochs: int = 50,
    mip_rel_gap: float = 0.01,
    solver_time_limit_sec: float = 0.5,
    num_eval_episodes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_train, df_val, df_test, config, dist_matrix, feature_cols = load_dataset(
        data_dir=str(data_path),
        network_path=str(data_path / "network_solomon_c1.json"),
    )
    episode_manifest = json.loads((data_path / "episode_manifest.json").read_text(encoding="utf-8"))
    if not episode_manifest.get("independent_episode_rng") or episode_manifest.get("m5_data_used"):
        raise ValueError("Expected independently generated synthetic episodes with no M5 data")
    split_ids = [
        set(episode_manifest[key])
        for key in ("train_instance_ids", "validation_instance_ids", "test_instance_ids")
    ]
    if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train, validation, and test episode IDs must be disjoint")
    forbidden_features = {
        "demand", "temperature", "rain_prob", "episode_seed", "episode_calendar_start",
        "instance_id", "t", "global_day",
    }
    if forbidden_features.intersection(feature_cols):
        raise ValueError(f"Forbidden or future-only predictor columns: {forbidden_features.intersection(feature_cols)}")
    for split_name, frame in (("train", df_train), ("validation", df_val), ("test", df_test)):
        periods_per_episode = frame.groupby("instance_id")["t"].nunique()
        if len(periods_per_episode) != frame.instance_id.nunique() or not periods_per_episode.eq(config["horizon_T"]).all():
            raise ValueError(f"{split_name} episodes do not all have the configured horizon")
    N, T = config["num_stores"], config["horizon_T"]
    X_train, d_train = structure_instances(df_train, feature_cols, N=N, T=T)
    X_val, d_val = structure_instances(df_val, feature_cols, N=N, T=T)
    X_test, d_test = structure_instances(df_test, feature_cols, N=N, T=T)
    val_ids = sorted(df_val.instance_id.unique().astype(int).tolist())
    test_ids = sorted(df_test.instance_id.unique().astype(int).tolist())
    if num_eval_episodes is not None:
        X_test, d_test, test_ids = X_test[:num_eval_episodes], d_test[:num_eval_episodes], test_ids[:num_eval_episodes]

    X_train_flat = df_train[feature_cols].to_numpy()
    y_train_flat = df_train["demand"].to_numpy()
    X_val_flat = df_val[feature_cols].to_numpy()
    y_val_flat = df_val["demand"].to_numpy()

    # Fit the common point forecast and derive the validation-only residual set.
    lightgbm = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05)
    tick = time.perf_counter()
    lightgbm.fit(X_train_flat, y_train_flat)
    lgb_train_time = time.perf_counter() - tick
    pred_val_lgb = np.maximum(0.0, lightgbm.predict(X_val))
    pred_test_lgb = np.maximum(0.0, lightgbm.predict(X_test))
    residuals = d_val - pred_val_lgb

    # The policy's two parameters are chosen only on the validation split.
    tuning_start = time.perf_counter()
    ul_policy, tuning_df = _tune_ul_lambda(
        residuals=residuals,
        val_predictions=pred_val_lgb,
        val_actuals=d_val,
        val_ids=val_ids,
        distance_matrix=dist_matrix,
        config=config,
    )
    ul_tuning_time_sec = time.perf_counter() - tuning_start
    tuning_df.to_csv(out_path / "validation_ul_lambda_tuning.csv", index=False)

    quantile = LightGBMQuantilePTO(
        stockout_penalty_pu=config["default_stockout_penalty_pu"],
        spoilage_penalty_ps=config["spoilage_penalty_ps"],
        holding_cost_h=config["holding_cost_h"],
        shelf_life_tau=config["default_shelf_life_tau"],
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
    )
    tick = time.perf_counter()
    quantile.fit(X_train_flat, y_train_flat)
    quantile_train_time = time.perf_counter() - tick
    pred_test_quantile = np.maximum(0.0, quantile.predict(X_test))

    result_records: list[dict[str, Any]] = []
    dfl_training_solver_rows: list[dict[str, Any]] = []
    result_records.extend(_evaluate_predictor(
        method="LightGBM_PTO", predictions=pred_test_lgb, actuals=d_test,
        episode_ids=test_ids, distance_matrix=dist_matrix, config=config,
        train_time_sec=lgb_train_time, mip_rel_gap=mip_rel_gap,
        solver_time_limit_sec=solver_time_limit_sec, seed=0,
    ))
    result_records.extend(_evaluate_predictor(
        method="LightGBM_Quantile_PTO", predictions=pred_test_quantile, actuals=d_test,
        episode_ids=test_ids, distance_matrix=dist_matrix, config=config,
        train_time_sec=quantile_train_time, mip_rel_gap=mip_rel_gap,
        solver_time_limit_sec=solver_time_limit_sec, seed=0,
    ))
    result_records.extend(_evaluate_ul_lambda(
        policy=ul_policy, method="Crama_UL_lambda_adaptation", predictions=pred_test_lgb,
        actuals=d_test, episode_ids=test_ids, distance_matrix=dist_matrix,
        config=config, seed=0, validation_tuning_time_sec=ul_tuning_time_sec,
    ))

    # Repeated DFL fits estimate training-seed variability; they do not increase
    # the number of independent test episodes.
    for seed in dfl_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dfl = DFLPipeline(
            loss_type="SPO+",
            in_features=len(feature_cols),
            num_stores=N,
            horizon_T=T,
            fleet_size_K=config["fleet_size_K"],
            vehicle_capacity_Q=config["vehicle_capacity_Q"],
            store_capacity_Ci=config["store_capacity_Ci"],
            shelf_life_tau=config["default_shelf_life_tau"],
            holding_cost_h=config["holding_cost_h"],
            aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
            spoilage_penalty_ps=config["spoilage_penalty_ps"],
            stockout_penalty_pu=config["default_stockout_penalty_pu"],
            distance_matrix=dist_matrix,
            decision_oracle_use_milp=True,
            solver_time_limit_sec=solver_time_limit_sec,
            mip_rel_gap=mip_rel_gap,
        )
        tick = time.perf_counter()
        dfl.fit_two_phase(
            X_train=X_train,
            d_train=d_train,
            X_val=X_val,
            d_val=d_val,
            total_epochs=dfl_epochs,
            transition_ratio=0.8,
            batch_size=16,
            val_subset_size=min(10, len(X_val)),
            verbose=False,
        )
        train_time = time.perf_counter() - tick
        for call_idx, diag in enumerate(dfl.optmodel.model.solver_diagnostics):
            dfl_training_solver_rows.append({"training_seed": seed, "oracle_call": call_idx, **diag})
        pred_dfl = np.maximum(0.0, dfl.predict(X_test))
        result_records.extend(_evaluate_predictor(
            method="Two-Phase_DFL", predictions=pred_dfl, actuals=d_test,
            episode_ids=test_ids, distance_matrix=dist_matrix, config=config,
            train_time_sec=train_time, mip_rel_gap=mip_rel_gap,
            solver_time_limit_sec=solver_time_limit_sec, seed=seed,
        ))

    episode_metrics = pd.DataFrame(result_records)
    episode_metrics.to_csv(out_path / "episode_metrics.csv", index=False)
    per_episode = episode_metrics.groupby(["method", "episode_id"], as_index=False).mean(numeric_only=True)
    primary_metrics = [
        "total_cost", "routing_cost", "holding_cost", "spoilage_cost", "expiry_cost",
        "capacity_disposal_cost", "expired_units", "capacity_discard_units", "stockout_cost",
        "unmet_demand_units", "fill_rate", "delivery_units", "on_hand_unit_periods",
        "average_on_hand_per_store_period", "ending_inventory_units", "route_distance", "routes_used",
        "fleet_capacity_utilization", "forecast_mae", "forecast_rmse",
    ]
    primary_metrics.extend(
        column for column in episode_metrics.columns
        if column.startswith(("fill_rate_period_", "fill_rate_store_", "average_on_hand_age_"))
        and column not in primary_metrics
    )
    summary_rows = []
    for method, group in per_episode.groupby("method"):
        row: dict[str, Any] = {"method": method, "n_test_episodes": group.episode_id.nunique()}
        seed_counts = episode_metrics[episode_metrics.method == method].groupby("episode_id").training_seed.nunique()
        row["n_training_seeds"] = int(seed_counts.max())
        for metric in primary_metrics:
            if metric in group:
                row[f"mean_{metric}"] = float(group[metric].mean())
                row[f"sd_episode_{metric}"] = (
                    float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
                )
        row["mean_training_time_sec"] = float(episode_metrics[episode_metrics.method == method].train_time_sec.mean())
        row["mean_validation_tuning_time_sec"] = float(
            episode_metrics[episode_metrics.method == method].get(
                "validation_tuning_time_sec", pd.Series([0.0])
            ).mean()
        )
        row["mean_planning_runtime_sec"] = float(
            episode_metrics[episode_metrics.method == method].solver_time_sec.mean()
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("method")
    summary.to_csv(out_path / "summary.csv", index=False)

    paired_tests = _paired_inference(episode_metrics)
    paired_tests.to_csv(out_path / "paired_primary_tests.csv", index=False)

    solver_details = episode_metrics[[
        "method", "episode_id", "training_seed", "solver_name", "solver_optimal",
        "solver_status", "solver_message", "solver_mip_gap", "solver_mip_dual_bound", "solver_mip_node_count",
        "solver_time_sec", "solver_used_heuristic_fallback", "solver_fifo_parity_max_abs_error",
    ]].drop_duplicates()
    solver_details.to_csv(out_path / "solver_diagnostics.csv", index=False)
    training_solver_details = pd.DataFrame(dfl_training_solver_rows)
    training_solver_details.to_csv(out_path / "dfl_training_solver_diagnostics.csv", index=False)

    manifest = {
        "protocol_version": "revision2-primary-2026-10-27",
        "master_data_seed": episode_manifest["master_seed"],
        "dataset_directory": str(data_path),
        "dataset_manifest_sha256": sha256(data_path / "episode_manifest.json"),
        "input_files_sha256": {
            p.name: sha256(p) for p in sorted(data_path.glob("*")) if p.is_file()
        },
        "independent_test_episodes": len(test_ids),
        "training_episodes": int(df_train.instance_id.nunique()),
        "validation_episodes": int(df_val.instance_id.nunique()),
        "test_episode_ids": test_ids,
        "feature_names": feature_cols,
        "initial_inventory": "zero units for every method at episode start",
        "information_timing": "All five-period plans use episode-start features and no realized within-episode demand; all methods are evaluated on identical held-out paths.",
        "primary_comparisons": PRIMARY_COMPARISONS,
        "sample_size_design": {
            "planned_test_episodes": 60,
            "episodes_in_this_run": len(test_ids),
            "design_target": "Normal-approximation target of 80% power for a paired standardized effect d=0.45 at two-sided alpha=0.025 for two primary comparisons.",
            "training_seed_unit": "Training seeds are averaged within each episode and are not counted as independent cases.",
        },
        "methods": {
            "LightGBM_PTO": "LightGBM point prediction followed by the common age-structured PIRP optimizer.",
            "LightGBM_Quantile_PTO": "LightGBM quantile prediction at the inventory-cost critical fractile followed by the same optimizer.",
            "Crama_UL_lambda_adaptation": "Validation-residual UL_lambda-inspired fixed-horizon order-up-to policy with shared route packing heuristic; not a faithful replication of Crama et al. (2018).",
            "Two-Phase_DFL": "Approximate SPO+-inspired hybrid: LP warmup, vectorized polyhedral loss on most batches in runs of at least 50 epochs, and HiGHS integer-oracle requests on the first two batches of the final 10 epochs; feasible construction fallbacks are used when HiGHS returns no acceptable incumbent. Checkpoints are selected by realized validation cost. This is not a full-MILP-per-batch SPO+ implementation.",
        },
        "optimizer": {
            "implementation": "Explicit SciPy HiGHS MILP request with LP-guided feasible construction fallback",
            "mip_rel_gap_requested": mip_rel_gap,
            "time_limit_sec_per_episode": solver_time_limit_sec,
            "optimality_is_not_assumed": True,
            "milp_fifo_state_parity": "Each forecast plan is checked against the common FIFO simulator under its own planning demand; a max absolute discrepancy above 1e-4 stops the run.",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "dfl_seeds": list(dfl_seeds),
        "dfl_epochs": dfl_epochs,
        "dfl_training_oracle": {
            "use_milp": True,
            "requested_mip_rel_gap": mip_rel_gap,
            "time_limit_sec": solver_time_limit_sec,
            "calls_by_solver": [
                {"training_seed": int(seed), "solver_name": str(name), "calls": int(count)}
                for (seed, name), count in training_solver_details.groupby(["training_seed", "solver_name"]).size().items()
            ] if not training_solver_details.empty else [],
            "all_calls_proven_optimal": bool(training_solver_details["is_optimal"].all()) if not training_solver_details.empty else False,
            "fallback_calls": int(training_solver_details["used_heuristic_fallback"].sum()) if not training_solver_details.empty else 0,
        },
        "primary_outcome": "total operating cost per five-period synthetic episode",
        "physical_outcomes": ["expiry waste", "capacity disposal", "unmet demand", "fill rate", "carryover inventory", "deliveries", "route distance"],
        "legacy_results": "Existing results/ files are preserved and not used as evidence for this run.",
    }
    (out_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return episode_metrics, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run revised primary synthetic PIRP benchmark")
    parser.add_argument("--data-dir", default="data/revision2_final/processed")
    parser.add_argument("--output-dir", default="results/revision2_final")
    parser.add_argument("--dfl-seeds", default="101,202,303")
    parser.add_argument("--dfl-epochs", type=int, default=50)
    parser.add_argument("--solver-time-limit-sec", type=float, default=0.5)
    parser.add_argument("--mip-rel-gap", type=float, default=0.01)
    parser.add_argument("--num-eval-episodes", type=int)
    args = parser.parse_args()
    run_revision_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dfl_seeds=tuple(int(value) for value in args.dfl_seeds.split(",") if value.strip()),
        dfl_epochs=args.dfl_epochs,
        solver_time_limit_sec=args.solver_time_limit_sec,
        mip_rel_gap=args.mip_rel_gap,
        num_eval_episodes=args.num_eval_episodes,
    )
