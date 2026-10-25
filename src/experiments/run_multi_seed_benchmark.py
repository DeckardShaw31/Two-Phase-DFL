"""
Multi-Seed Benchmark Evaluation Harness (5 Seeds x 1,500 Epochs).

Evaluates the complete 14-Model Benchmark Suite across 5 independent seeds:
  Seeds: [42, 1042, 2042, 3042, 4042]

Features:
  1. 1,500 epochs training on 140 training instances for neural/DFL models.
  2. Best-validation-regret checkpointing on all 30 validation instances (val_subset_size=30).
  3. Strict isolation of the 30 test instances until final evaluation.
  4. Epoch-by-epoch convergence trajectory logging to results/epoch_convergence_history.csv.
  5. Multi-seed summary with cross-seed mean ± SD and instance-level SDs to results/multi_seed_summary.csv.
  6. Updating results/benchmark_metrics.csv and results/statistical_significance.csv.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.run_benchmark import (
    ALL_14_MODELS,
    canonicalize_model_name,
    load_dataset,
    structure_instances,
)
from src.experiments.run_statistical_tests import run_statistical_tests
from src.models.baselines import (
    MLPPTO,
    ElasticNetPTO,
    GradientBoostingPTO,
    KNNSAAModel,
    LightGBMPTO,
    LightGBMQuantilePTO,
    RandomForestPTO,
    RidgePTO,
    SafetyStockPTO,
    TemporalSeqNetPTO,
)
from src.models.dfl_models import DFLPipeline
from src.optimization.pirp_model import solve_pirp
from src.utils.metrics import (
    compute_decision_regret,
    compute_forecast_accuracy,
    evaluate_decisions_under_true_demand,
)

DEFAULT_SEEDS = [42, 1042, 2042, 3042, 4042]


def evaluate_predictions_on_test(
    preds: np.ndarray,
    d_test: np.ndarray,
    oracle_costs: list[float],
    dist_matrix: np.ndarray,
    config: dict[str, Any],
    time_limit_sec: float = 2.0,
    model_name: str = "",
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Evaluates predicted demand on 30 test instances against clairvoyant Oracle."""
    eval_k = len(d_test)
    tau = config["default_shelf_life_tau"]
    pu = config["default_stockout_penalty_pu"]
    ps = config["spoilage_penalty_ps"]
    h = config.get("holding_cost_h", 1.0)
    aging_surcharge_eps = config.get("aging_surcharge_eps", 0.01)
    Q = config["vehicle_capacity_Q"]
    Ci = config["store_capacity_Ci"]

    regrets = []
    total_costs = []
    route_costs = []
    hold_costs = []
    spoil_costs = []
    stock_costs = []
    instance_records = []
    deliveries_list = []

    def _evaluate_single_instance(i: int):
        sol_m = solve_pirp(
            demand=preds[i],
            distance_matrix=dist_matrix,
            continuous_relaxation=False,
            shelf_life_tau=tau,
            holding_cost_h=h,
            aging_surcharge_eps=aging_surcharge_eps,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            vehicle_capacity_Q=Q,
            store_capacity_Ci=Ci,
            time_limit_sec=time_limit_sec,
        )
        metrics = evaluate_decisions_under_true_demand(
            q_plan=sol_m.q_total,
            x_plan=sol_m.x_routes,
            true_demand=d_test[i],
            distance_matrix=dist_matrix,
            h=h,
            aging_surcharge_eps=aging_surcharge_eps,
            p_s=ps,
            p_u=pu,
            tau=tau,
        )
        _, norm_regret = compute_decision_regret(metrics["total_cost"], oracle_costs[i])
        rec = {
            "instance_id": i + 1,
            "seed": seed,
            "model": model_name,
            "oracle_cost": float(oracle_costs[i]),
            "routing_cost": float(metrics["routing_cost"]),
            "holding_cost": float(metrics["holding_cost"]),
            "spoilage_cost": float(metrics["spoilage_cost"]),
            "stockout_cost": float(metrics["stockout_cost"]),
            "total_cost": float(metrics["total_cost"]),
            "normalized_regret_pct": float(norm_regret),
        }
        return i, norm_regret, metrics, rec, sol_m.q_total

    max_workers = min(8, eval_k)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_evaluate_single_instance, range(eval_k)))

    # Sort results by instance_id to maintain deterministic order
    results.sort(key=lambda r: r[0])

    for _, norm_regret, metrics, rec, q_total in results:
        regrets.append(norm_regret)
        total_costs.append(metrics["total_cost"])
        route_costs.append(metrics["routing_cost"])
        hold_costs.append(metrics["holding_cost"])
        spoil_costs.append(metrics["spoilage_cost"])
        stock_costs.append(metrics["stockout_cost"])
        instance_records.append(rec)
        deliveries_list.append(q_total)

    acc = compute_forecast_accuracy(preds, d_test)
    eval_summary = {
        "regrets": regrets,
        "mean_regret": float(np.mean(regrets)),
        "std_instance_regret": float(np.std(regrets)),
        "total_cost": float(np.mean(total_costs)),
        "routing_cost": float(np.mean(route_costs)),
        "holding_cost": float(np.mean(hold_costs)),
        "spoilage_cost": float(np.mean(spoil_costs)),
        "stockout_cost": float(np.mean(stock_costs)),
        "mse": float(acc["mse"]),
        "mae": float(acc["mae"]),
    }
    return eval_summary, instance_records, np.array(deliveries_list)


def run_multi_seed_benchmark(
    seeds: list[int] | None = None,
    models: str | list[str] = "all",
    total_epochs: int = 1500,
    topology: str = "C1",
    output_dir: str = "results",
    save_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the multi-seed 1,500-epoch benchmark evaluation.
    """
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    if isinstance(models, str):
        if models.strip().lower() == "all":
            selected_models = list(ALL_14_MODELS)
        else:
            selected_models = [canonicalize_model_name(m.strip()) for m in models.split(",") if m.strip()]
    else:
        selected_models = [canonicalize_model_name(m) for m in models]

    print("=" * 80)
    print("MULTI-SEED BENCHMARK EVALUATION (5 SEEDS x 1,500 EPOCHS)")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Selected Models ({len(selected_models)}): {selected_models}")
    print(f"Topology: {topology}, DFL Total Epochs: {total_epochs}")
    print("=" * 80)

    network_file = f"data/processed/network_solomon_{topology.lower()}.json"
    df_train, df_val, df_test, config, dist_matrix, feat_cols = load_dataset(network_path=network_file)

    N = config["num_stores"]
    T = config["horizon_T"]
    tau = config["default_shelf_life_tau"]
    pu = config["default_stockout_penalty_pu"]
    ps = config["spoilage_penalty_ps"]
    Q = config["vehicle_capacity_Q"]
    Ci = config["store_capacity_Ci"]

    X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
    X_val_3d, d_val_3d = structure_instances(df_val, feat_cols, N=N, T=T)
    X_test_3d, d_test_3d = structure_instances(df_test, feat_cols, N=N, T=T)

    X_train_flat = df_train[feat_cols].values
    y_train_flat = df_train["demand"].values

    # Precompute Oracle solutions on 30 test instances
    print("Precomputing clairvoyant Oracle solutions on 30 test instances...")
    def _solve_oracle(i: int) -> tuple[int, float]:
        sol = solve_pirp(
            demand=d_test_3d[i],
            distance_matrix=dist_matrix,
            continuous_relaxation=False,
            shelf_life_tau=tau,
            holding_cost_h=config.get("holding_cost_h", 1.0),
            aging_surcharge_eps=config.get("aging_surcharge_eps", 0.01),
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            vehicle_capacity_Q=Q,
            store_capacity_Ci=Ci,
            time_limit_sec=2.0,
        )
        return i, sol.total_cost

    with ThreadPoolExecutor(max_workers=min(8, len(d_test_3d))) as executor:
        oracle_results = list(executor.map(_solve_oracle, range(len(d_test_3d))))
    oracle_results.sort(key=lambda r: r[0])
    oracle_test_costs = [r[1] for r in oracle_results]
    print(f"Clairvoyant Oracle Test Mean Total Cost: ${np.mean(oracle_test_costs):.2f}")

    convergence_history_records = []
    model_seed_results: dict[str, list[dict[str, Any]]] = {m: [] for m in selected_models}
    all_instance_records: list[dict[str, Any]] = []
    predictions_dict: dict[tuple[str, int], np.ndarray] = {}
    deliveries_dict: dict[tuple[str, int], np.ndarray] = {}
    eval_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]] = {}

    # Evaluate each seed
    for seed_idx, seed in enumerate(seeds):
        print("\n=======================================================")
        print(f"STARTING SEED {seed_idx + 1}/{len(seeds)}: random_seed = {seed}")
        print("=======================================================")
        np.random.seed(seed)
        torch.manual_seed(seed)

        for m_name in selected_models:
            print(f"\n[Seed {seed}] Evaluating Model: {m_name}...")
            t0 = time.perf_counter()
            best_epoch = 0

            if m_name == "Ridge_PTO":
                r_m = RidgePTO(alpha=10.0).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = r_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "ElasticNet_PTO":
                en_m = ElasticNetPTO(alpha=0.1, l1_ratio=0.5).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = en_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "RandomForest_PTO":
                rf_m = RandomForestPTO(n_estimators=100, max_depth=8, random_state=seed).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = rf_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "GradientBoosting_PTO":
                gb_m = GradientBoostingPTO(n_estimators=100, learning_rate=0.05, max_depth=4).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = gb_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "LightGBM_PTO":
                lgb_m = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = lgb_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "LightGBM_Quantile_PTO":
                lgb_q = LightGBMQuantilePTO(
                    critical_fractile=0.882,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                ).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = lgb_q.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "SafetyStock_PTO":
                ss_m = SafetyStockPTO(
                    alpha_ridge=10.0,
                    stockout_penalty_pu=pu,
                    spoilage_penalty_ps=ps,
                    holding_cost_h=config.get("holding_cost_h", 1.0),
                    shelf_life_tau=tau,
                    z_factor=1.65,
                ).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = ss_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name == "MLP_PTO":
                mlp_m = MLPPTO(in_features=len(feat_cols), epochs=min(total_epochs, 200), lr=1e-3, batch_size=64)
                mlp_m.fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = mlp_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0
                best_epoch = min(total_epochs, 200)

            elif m_name == "TemporalSeqNet_PTO":
                seq_m = TemporalSeqNetPTO(in_features=len(feat_cols), epochs=min(total_epochs, 100), lr=1e-3, batch_size=16, patience=12)
                seq_m.fit(X_train_3d, d_train_3d, X_val=X_val_3d, y_val=d_val_3d)
                preds = seq_m.predict(X_test_3d)
                train_time = time.perf_counter() - t0
                best_epoch = min(total_epochs, 100)

            elif m_name == "KNN_SAA":
                knn_m = KNNSAAModel(k_neighbors=5).fit(X_train_flat, y_train_flat)
                preds = np.zeros_like(d_test_3d)
                for i in range(len(d_test_3d)):
                    preds[i] = knn_m.predict(X_test_3d[i].reshape(N * T, -1)).reshape(N, T)
                train_time = time.perf_counter() - t0

            elif m_name in ("DFL_DPO", "DFL_IntOpt", "DFL_SPO+"):
                loss_key = "DPO" if "DPO" in m_name else ("INTOPT" if "IntOpt" in m_name else "SPO+")
                dfl_direct = DFLPipeline(
                    loss_type=loss_key,
                    in_features=len(feat_cols),
                    num_stores=N,
                    horizon_T=T,
                    distance_matrix=dist_matrix,
                    shelf_life_tau=tau,
                    stockout_penalty_pu=pu,
                    spoilage_penalty_ps=ps,
                )
                history = dfl_direct.fit_two_phase(
                    X_train=X_train_3d,
                    d_train=d_train_3d,
                    X_val=X_val_3d,
                    d_val=d_val_3d,
                    total_epochs=total_epochs,
                    transition_ratio=1.0,  # Pure surrogate differentiation
                    lr_phase1=1e-3,
                    lr_phase2=1e-4,
                    batch_size=8,
                    lambda_pinball=0.1,
                    val_subset_size=30,
                    verbose=False,
                )
                train_time = time.perf_counter() - t0
                preds = dfl_direct.predict(X_test_3d)
                # Find best checkpoint epoch
                val_regrets = [h["val_regret_pct"] for h in history if h.get("val_regret_pct", 0) > 0]
                best_epoch = int(np.argmin(val_regrets)) + 1 if val_regrets else total_epochs

                for h in history:
                    convergence_history_records.append({
                        "Seed": seed,
                        "Model": m_name,
                        "Epoch": h["epoch"],
                        "Phase": h["phase"],
                        "TrainLoss": h["loss"],
                        "ValRegret_pct": h["val_regret_pct"],
                        "EpochTime_sec": h["epoch_time_sec"],
                        "CumulativeTime_sec": h["cumulative_time_sec"],
                    })

            elif m_name == "Two-Phase_DFL (Ours)":
                dfl_ours = DFLPipeline(
                    loss_type="SPO+",
                    in_features=len(feat_cols),
                    num_stores=N,
                    horizon_T=T,
                    distance_matrix=dist_matrix,
                    shelf_life_tau=tau,
                    stockout_penalty_pu=pu,
                    spoilage_penalty_ps=ps,
                )
                history = dfl_ours.fit_two_phase(
                    X_train=X_train_3d,
                    d_train=d_train_3d,
                    X_val=X_val_3d,
                    d_val=d_val_3d,
                    total_epochs=total_epochs,
                    transition_ratio=0.8,
                    lr_phase1=1e-3,
                    lr_phase2=1e-4,
                    batch_size=8,
                    lambda_pinball=0.2,
                    val_subset_size=30,
                    verbose=False,
                )
                train_time = time.perf_counter() - t0
                preds = dfl_ours.predict(X_test_3d)
                val_regrets = [h["val_regret_pct"] for h in history if h.get("val_regret_pct", 0) > 0]
                best_epoch = int(np.argmin(val_regrets)) + 1 if val_regrets else 1150

                for h in history:
                    convergence_history_records.append({
                        "Seed": seed,
                        "Model": m_name,
                        "Epoch": h["epoch"],
                        "Phase": h["phase"],
                        "TrainLoss": h["loss"],
                        "ValRegret_pct": h["val_regret_pct"],
                        "EpochTime_sec": h["epoch_time_sec"],
                        "CumulativeTime_sec": h["cumulative_time_sec"],
                    })

            # Evaluate on test instances
            if m_name in eval_cache and np.allclose(preds, eval_cache[m_name][3]):
                cached_eval, cached_recs, cached_deliv, _ = eval_cache[m_name]
                test_eval = copy.deepcopy(cached_eval)
                test_eval["seed"] = seed
                test_eval["train_time"] = train_time
                test_eval["best_epoch"] = best_epoch
                inst_recs = [{**r, "seed": seed} for r in cached_recs]
                deliv_arr = cached_deliv.copy()
            else:
                test_eval, inst_recs, deliv_arr = evaluate_predictions_on_test(
                    preds=preds,
                    d_test=d_test_3d,
                    oracle_costs=oracle_test_costs,
                    dist_matrix=dist_matrix,
                    config=config,
                    time_limit_sec=2.0,
                    model_name=m_name,
                    seed=seed,
                )
                test_eval["train_time"] = train_time
                test_eval["best_epoch"] = best_epoch
                test_eval["seed"] = seed
                eval_cache[m_name] = (test_eval, inst_recs, deliv_arr, preds)

            model_seed_results[m_name].append(test_eval)
            all_instance_records.extend(inst_recs)
            predictions_dict[(m_name, seed)] = preds
            deliveries_dict[(m_name, seed)] = deliv_arr

            print(
                f"  [Seed {seed}] {m_name}: Regret={test_eval['mean_regret']:.2f}% "
                f"(Total Cost=${test_eval['total_cost']:.2f}), Time={train_time:.1f}s"
            )

    # Compile Multi-Seed Summary
    summary_rows = []
    category_map = {
        "Two-Phase_DFL (Ours)": "Proposed Method",
        "DFL_SPO+": "End-to-End DFL",
        "DFL_IntOpt": "End-to-End DFL",
        "DFL_DPO": "End-to-End DFL",
    }

    for m_name in selected_models:
        runs = model_seed_results[m_name]
        cat = category_map.get(m_name, "Two-Stage PTO")
        seed_regrets = [r["mean_regret"] for r in runs]
        seed_instance_stds = [r["std_instance_regret"] for r in runs]
        seed_totals = [r["total_cost"] for r in runs]
        seed_routes = [r["routing_cost"] for r in runs]
        seed_spoils = [r["spoilage_cost"] for r in runs]
        seed_stocks = [r["stockout_cost"] for r in runs]
        seed_mses = [r["mse"] for r in runs]
        seed_maes = [r["mae"] for r in runs]
        seed_times = [r["train_time"] for r in runs]
        seed_best_eps = [r["best_epoch"] for r in runs]

        summary_rows.append({
            "Model": m_name,
            "Category": cat,
            "NumSeeds": len(runs),
            "Regret_Mean_pct": float(np.mean(seed_regrets)),
            "Regret_CrossSeed_std": float(np.std(seed_regrets)),
            "Regret_Instance_std": float(np.mean(seed_instance_stds)),
            "TotalCost_Mean": float(np.mean(seed_totals)),
            "TotalCost_std": float(np.std(seed_totals)),
            "RoutingCost_Mean": float(np.mean(seed_routes)),
            "SpoilageCost_Mean": float(np.mean(seed_spoils)),
            "StockoutCost_Mean": float(np.mean(seed_stocks)),
            "MSE_Mean": float(np.mean(seed_mses)),
            "MAE_Mean": float(np.mean(seed_maes)),
            "TrainTime_sec_Mean": float(np.mean(seed_times)),
            "Best_Epoch_Mean": float(np.mean(seed_best_eps)),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Benchmark metrics update (retaining project standard schema)
    bench_rows = []
    for _, s_row in summary_df.iterrows():
        bench_rows.append({
            "Model": s_row["Model"],
            "Category": s_row["Category"],
            "Topology": topology,
            "ShelfLife_tau": tau,
            "StockoutRatio_pu_h": pu / 1.0,
            "NormalizedRegret_pct": s_row["Regret_Mean_pct"],
            "Regret_std": s_row["Regret_CrossSeed_std"],
            "TotalCost": s_row["TotalCost_Mean"],
            "RoutingCost": s_row["RoutingCost_Mean"],
            "SpoilageCost": s_row["SpoilageCost_Mean"],
            "StockoutCost": s_row["StockoutCost_Mean"],
            "MSE": s_row["MSE_Mean"],
            "MAE": s_row["MAE_Mean"],
            "TrainTime_sec": s_row["TrainTime_sec_Mean"],
        })
    bench_df = pd.DataFrame(bench_rows)

    # Paired Statistical Significance against Two-Phase DFL (Ours)
    significance_rows = []
    ours_regrets_all = []
    for r in model_seed_results.get("Two-Phase_DFL (Ours)", []):
        ours_regrets_all.extend(r["regrets"])

    if len(ours_regrets_all) > 0:
        for m_name in selected_models:
            if m_name == "Two-Phase_DFL (Ours)":
                continue
            base_regrets_all = []
            for r in model_seed_results[m_name]:
                base_regrets_all.extend(r["regrets"])

            if len(base_regrets_all) != len(ours_regrets_all):
                continue
            diffs = np.array(base_regrets_all) - np.array(ours_regrets_all)
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

            if std_diff > 1e-8:
                t_stat = mean_diff / (std_diff / np.sqrt(len(diffs)))
                t_pval = float(stats.t.sf(t_stat, df=len(diffs) - 1))
            else:
                t_stat = 0.0
                t_pval = 1.0

            nonzero_diffs = diffs[~np.isclose(diffs, 0.0)]
            if len(nonzero_diffs) >= 1:
                try:
                    w_res = stats.wilcoxon(base_regrets_all, ours_regrets_all, alternative="greater", zero_method="wilcox")
                    w_stat = float(w_res.statistic)
                    w_pval = float(w_res.pvalue)
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    w_stat = 0.0
                    w_pval = 1.0
            else:
                w_stat = 0.0
                w_pval = 1.0

            significance_rows.append({
                "Baseline_Model": m_name,
                "Ours_Model": "Two-Phase_DFL (Ours)",
                "Mean_Regret_Baseline_pct": float(np.mean(base_regrets_all)),
                "Mean_Regret_Ours_pct": float(np.mean(ours_regrets_all)),
                "Mean_Difference_pct": mean_diff,
                "Wilcoxon_Stat": w_stat,
                "Wilcoxon_p_value": w_pval,
                "T_Stat": t_stat,
                "T_p_value": t_pval,
                "Significant_p05": bool(w_pval < 0.05 or t_pval < 0.05),
                "Significant_p01": bool(w_pval < 0.01 or t_pval < 0.01),
            })

    sig_df = pd.DataFrame(significance_rows)
    conv_df = pd.DataFrame(convergence_history_records)

    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "multi_seed_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved multi-seed summary to: {summary_path}")

        conv_path = os.path.join(output_dir, "epoch_convergence_history.csv")
        conv_df.to_csv(conv_path, index=False)
        print(f"Saved epoch convergence history ({len(conv_df)} rows) to: {conv_path}")

        bench_path = os.path.join(output_dir, "benchmark_metrics.csv")
        bench_df.to_csv(bench_path, index=False)
        with open(os.path.join(output_dir, "benchmark_metrics.json"), "w") as f:
            json.dump(bench_df.to_dict(orient="records"), f, indent=2)
        print(f"Updated benchmark metrics to: {bench_path}")

        sig_path = os.path.join(output_dir, "statistical_significance.csv")
        sig_df.to_csv(sig_path, index=False)
        with open(os.path.join(output_dir, "statistical_significance.json"), "w") as f:
            json.dump(sig_df.to_dict(orient="records"), f, indent=2)
        print(f"Updated statistical significance to: {sig_path}")

        if len(all_instance_records) > 0:
            inst_df = pd.DataFrame(all_instance_records)
            inst_csv_path = os.path.join(output_dir, "instance_level_metrics.csv")
            inst_df.to_csv(inst_csv_path, index=False)
            print(f"Saved instance-level metrics ({len(inst_df)} rows) to: {inst_csv_path}")

            # Assemble .npz arrays
            K = len(d_test_3d)
            M = len(selected_models)
            S = len(seeds)
            preds_arr = np.zeros((M, S, K, N, T), dtype=np.float32)
            deliv_arr_all = np.zeros((M, S, K, N, T), dtype=np.float32)
            costs_arr = np.zeros((M, S, K), dtype=np.float64)
            regrets_arr = np.zeros((M, S, K), dtype=np.float64)

            for m_i, m_name in enumerate(selected_models):
                for s_i, seed in enumerate(seeds):
                    if (m_name, seed) in predictions_dict:
                        preds_arr[m_i, s_i] = predictions_dict[(m_name, seed)]
                    if (m_name, seed) in deliveries_dict:
                        deliv_arr_all[m_i, s_i] = deliveries_dict[(m_name, seed)]
                    sub_df = inst_df[(inst_df["model"] == m_name) & (inst_df["seed"] == seed)].sort_values("instance_id")
                    if len(sub_df) == K:
                        costs_arr[m_i, s_i] = sub_df["total_cost"].values
                        regrets_arr[m_i, s_i] = sub_df["normalized_regret_pct"].values

            npz_path = os.path.join(output_dir, "instance_predictions_and_decisions.npz")
            np.savez_compressed(
                npz_path,
                predictions=preds_arr,
                deliveries=deliv_arr_all,
                true_demand=d_test_3d.astype(np.float32),
                oracle_costs=np.array(oracle_test_costs, dtype=np.float64),
                costs=costs_arr,
                regrets=regrets_arr,
                model_names=np.array(selected_models),
                seeds=np.array(seeds),
                instance_ids=np.arange(1, K + 1),
            )
            print(f"Saved instance predictions and decisions archive to: {npz_path}")

            # Run Statistical Tests suite
            if "Two-Phase_DFL (Ours)" in selected_models and len(selected_models) > 1:
                stat_tests_path = os.path.join(output_dir, "statistical_tests.csv")
                run_statistical_tests(
                    instance_metrics_df=inst_df,
                    target_model="Two-Phase_DFL (Ours)",
                    output_path=stat_tests_path,
                    save_results=True,
                )

    print("\n=== MULTI-SEED SUMMARY TABLE ===")
    print(summary_df[["Model", "NumSeeds", "Regret_Mean_pct", "Regret_CrossSeed_std", "Regret_Instance_std", "TotalCost_Mean"]].to_string(index=False))

    return summary_df, conv_df, sig_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed 1,500-Epoch Benchmark")
    parser.add_argument("--epochs", type=int, default=1500, help="Total DFL training epochs")
    parser.add_argument("--topology", type=str, default="C1", help="Network topology")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seeds", type=str, default="all", help="Comma-separated seed list or 'all'")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model names or 'all'")
    args = parser.parse_args()

    parsed_seeds = DEFAULT_SEEDS if args.seeds == "all" else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_multi_seed_benchmark(
        seeds=parsed_seeds,
        models=args.models,
        total_epochs=args.epochs,
        topology=args.topology,
        output_dir=args.output_dir,
    )
