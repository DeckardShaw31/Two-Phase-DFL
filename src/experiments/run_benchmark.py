# HISTORICAL: superseded legacy experiment script. Do not use its outputs as evidence for the revised manuscript; see src/experiments/LEGACY_README.md and use run_revision_benchmark.py.
"""
Comprehensive Benchmark Execution Harness for Perishable Inventory-Routing (PIRP).

Evaluates the complete 14-Model Benchmark Suite:
  Two-Stage Predict-then-Optimize (PTO) Baselines:
    1. Ridge_PTO (L2 Ridge Regression + MILP)
    2. ElasticNet_PTO (L1+L2 ElasticNet Regression + MILP)
    3. RandomForest_PTO (Bagging Decision Trees + MILP)
    4. GradientBoosting_PTO (Sequential Gradient Boosted Trees + MILP)
    5. LightGBM_PTO (Symmetric MSE Histogram GBDT + MILP)
    6. LightGBM_Quantile_PTO (Asymmetric Critical-Fractile Pinball GBDT + MILP)
    7. SafetyStock_PTO (Ridge Forecast + Empirical Safety Buffer + MILP)
    8. MLP_PTO (Deep Feedforward Neural Network + MILP)
    9. TemporalSeqNet_PTO (Recurrent LSTM/GRU Sequence Net with Store Embeddings + MILP)
    10. KNN_SAA (Conditional Sample Average Approximation + MILP)

  End-to-End Decision-Focused Learning (DFL) Competitors:
    11. DFL_DPO (Differentiable Perturbed Optimizer + MILP)
    12. DFL_IntOpt (Interior Point Differentiation + MILP)
    13. DFL_SPO+ (Smart Predict-then-Optimize + MILP)

  Proposed Method:
    14. Two-Phase_DFL (Ours) (LP Warmup + MILP Fine-Tuning + Best-Validation Checkpointing)

Outputs:
  - {output_dir}/benchmark_metrics.csv
  - {output_dir}/benchmark_metrics.json
  - {output_dir}/instance_regrets.csv
  - {output_dir}/statistical_significance.csv
  - {output_dir}/statistical_significance.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

ALL_14_MODELS = [
    "Ridge_PTO",
    "ElasticNet_PTO",
    "RandomForest_PTO",
    "GradientBoosting_PTO",
    "LightGBM_PTO",
    "LightGBM_Quantile_PTO",
    "SafetyStock_PTO",
    "MLP_PTO",
    "TemporalSeqNet_PTO",
    "KNN_SAA",
    "DFL_DPO",
    "DFL_IntOpt",
    "DFL_SPO+",
    "Two-Phase_DFL (Ours)",
]

MODEL_NAME_CANONICAL_MAP = {
    "ridge": "Ridge_PTO",
    "ridge_pto": "Ridge_PTO",
    "elasticnet": "ElasticNet_PTO",
    "elasticnet_pto": "ElasticNet_PTO",
    "randomforest": "RandomForest_PTO",
    "randomforest_pto": "RandomForest_PTO",
    "rf": "RandomForest_PTO",
    "rf_pto": "RandomForest_PTO",
    "gradientboosting": "GradientBoosting_PTO",
    "gradientboosting_pto": "GradientBoosting_PTO",
    "gbdt": "GradientBoosting_PTO",
    "gbdt_pto": "GradientBoosting_PTO",
    "lightgbm": "LightGBM_PTO",
    "lightgbm_pto": "LightGBM_PTO",
    "lgb": "LightGBM_PTO",
    "lgb_pto": "LightGBM_PTO",
    "lightgbm_quantile": "LightGBM_Quantile_PTO",
    "lightgbm_quantile_pto": "LightGBM_Quantile_PTO",
    "lgb_quantile": "LightGBM_Quantile_PTO",
    "lgb_quantile_pto": "LightGBM_Quantile_PTO",
    "safetystock": "SafetyStock_PTO",
    "safetystock_pto": "SafetyStock_PTO",
    "safety_stock": "SafetyStock_PTO",
    "safety_stock_pto": "SafetyStock_PTO",
    "mlp": "MLP_PTO",
    "mlp_pto": "MLP_PTO",
    "temporalseqnet": "TemporalSeqNet_PTO",
    "temporalseqnet_pto": "TemporalSeqNet_PTO",
    "temporal_seq_net": "TemporalSeqNet_PTO",
    "temporal_seq_net_pto": "TemporalSeqNet_PTO",
    "temporal": "TemporalSeqNet_PTO",
    "temporal_pto": "TemporalSeqNet_PTO",
    "knn": "KNN_SAA",
    "knn_saa": "KNN_SAA",
    "knn_saa_pto": "KNN_SAA",
    "dfl_dpo": "DFL_DPO",
    "dpo": "DFL_DPO",
    "dfl_intopt": "DFL_IntOpt",
    "intopt": "DFL_IntOpt",
    "dfl_spo+": "DFL_SPO+",
    "dfl_spo": "DFL_SPO+",
    "spo+": "DFL_SPO+",
    "two_phase_dfl": "Two-Phase_DFL (Ours)",
    "two-phase_dfl": "Two-Phase_DFL (Ours)",
    "twophasedfl": "Two-Phase_DFL (Ours)",
    "two-phase_dfl (ours)": "Two-Phase_DFL (Ours)",
    "ours": "Two-Phase_DFL (Ours)",
}


def canonicalize_model_name(raw_name: str) -> str:
    """Normalizes model name to canonical identifier."""
    cleaned = raw_name.strip().lower()
    if cleaned in MODEL_NAME_CANONICAL_MAP:
        return MODEL_NAME_CANONICAL_MAP[cleaned]
    # Check exact case match
    for m in ALL_14_MODELS:
        if raw_name.strip().lower() == m.lower():
            return m
    return raw_name.strip()


def load_dataset(
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    params_path: str | None = None,
    network_path: str | None = None,
    data_dir: str = "data/processed",
):
    """Loads all split datasets and configuration."""
    train_path = train_path or os.path.join(data_dir, "demand_features_train.csv")
    val_path = val_path or os.path.join(data_dir, "demand_features_val.csv")
    test_path = test_path or os.path.join(data_dir, "demand_features_test.csv")
    params_path = params_path or os.path.join(data_dir, "params_config.json")
    network_path = network_path or os.path.join(data_dir, "network_solomon_c1.json")
    with open(params_path, "r") as f:
        config = json.load(f)

    with open(network_path, "r") as f:
        network = json.load(f)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    distance_matrix = np.array(network["distance_matrix"], dtype=np.float64)
    feature_names = config["feature_names"]

    return df_train, df_val, df_test, config, distance_matrix, feature_names


def structure_instances(
    df: pd.DataFrame, feature_cols: list[str], N: int = 10, T: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Structures tabular dataframe into 3D/4D arrays per instance."""
    instance_ids = sorted(df["instance_id"].unique())
    num_inst = len(instance_ids)
    num_feat = len(feature_cols)

    X_instances = np.zeros((num_inst, N, T, num_feat), dtype=np.float32)
    d_instances = np.zeros((num_inst, N, T), dtype=np.float32)

    for idx, inst_id in enumerate(instance_ids):
        sub = df[df["instance_id"] == inst_id]
        for _, row in sub.iterrows():
            s = int(row["store_id"]) - 1
            t = int(row["t"]) - 1
            X_instances[idx, s, t, :] = row[feature_cols].values
            d_instances[idx, s, t] = float(row["demand"])

    return X_instances, d_instances


def evaluate_model_decisions(
    model_name: str,
    category: str,
    preds: np.ndarray,
    d_test_eval: np.ndarray,
    oracle_costs: list[float],
    dist_matrix: np.ndarray,
    config: dict[str, Any],
    train_time_sec: float,
    topology: str = "C1",
    time_limit_sec: float = 2.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Evaluates model predictions by solving downstream PIRP on each test instance,
    computing realized operational costs, decision regret, and forecast accuracy.
    """
    eval_instances = len(d_test_eval)
    tau = config["default_shelf_life_tau"]
    pu = config["default_stockout_penalty_pu"]
    ps = config["spoilage_penalty_ps"]
    h = config["holding_cost_h"]
    Q = config["vehicle_capacity_Q"]
    Ci = config["store_capacity_Ci"]

    regrets = []
    total_costs = []
    route_costs = []
    spoil_costs = []
    stock_costs = []
    instance_records = []

    for i in range(eval_instances):
        sol_m = solve_pirp(
            demand=preds[i],
            distance_matrix=dist_matrix,
            continuous_relaxation=False,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            vehicle_capacity_Q=Q,
            store_capacity_Ci=Ci,
            time_limit_sec=time_limit_sec,
        )
        eval_metrics = evaluate_decisions_under_true_demand(
            q_plan=sol_m.q_total,
            x_plan=sol_m.x_routes,
            true_demand=d_test_eval[i],
            distance_matrix=dist_matrix,
            h=h,
            p_s=ps,
            p_u=pu,
            tau=tau,
            store_capacity_Ci=Ci,
        )
        abs_regret, norm_regret = compute_decision_regret(eval_metrics["total_cost"], oracle_costs[i])
        regrets.append(norm_regret)
        total_costs.append(eval_metrics["total_cost"])
        route_costs.append(eval_metrics["routing_cost"])
        spoil_costs.append(eval_metrics["spoilage_cost"])
        stock_costs.append(eval_metrics["stockout_cost"])

        instance_records.append({
            "instance_id": i + 1,
            "Model": model_name,
            "RealizedTotalCost": float(eval_metrics["total_cost"]),
            "OracleCost": float(oracle_costs[i]),
            "AbsoluteRegret": float(abs_regret),
            "NormalizedRegret_pct": float(norm_regret),
        })

    acc = compute_forecast_accuracy(preds, d_test_eval)

    summary_row = {
        "Model": model_name,
        "Category": category,
        "Topology": topology,
        "ShelfLife_tau": tau,
        "StockoutRatio_pu_h": pu / h,
        "NormalizedRegret_pct": float(np.mean(regrets)),
        "Regret_std": float(np.std(regrets)),
        "TotalCost": float(np.mean(total_costs)),
        "RoutingCost": float(np.mean(route_costs)),
        "SpoilageCost": float(np.mean(spoil_costs)),
        "StockoutCost": float(np.mean(stock_costs)),
        "MSE": acc["mse"],
        "MAE": acc["mae"],
        "TrainTime_sec": float(train_time_sec),
    }

    return summary_row, instance_records


def compute_statistical_significance(
    instance_regrets_df: pd.DataFrame,
    target_model: str = "Two-Phase_DFL (Ours)",
) -> pd.DataFrame:
    """
    Computes paired Wilcoxon signed-rank test and paired Student's t-test comparing
    the target model (Proposed Method) against each competing baseline model.
    """
    significance_rows = []
    all_models = sorted(instance_regrets_df["Model"].unique())

    if target_model not in all_models:
        return pd.DataFrame(columns=[
            "Baseline_Model",
            "Ours_Model",
            "Mean_Regret_Baseline_pct",
            "Mean_Regret_Ours_pct",
            "Mean_Difference_pct",
            "Wilcoxon_Stat",
            "Wilcoxon_p_value",
            "T_Stat",
            "T_p_value",
            "Significant_p05",
            "Significant_p01",
        ])

    df_ours = instance_regrets_df[instance_regrets_df["Model"] == target_model].sort_values("instance_id")
    regrets_ours = df_ours["NormalizedRegret_pct"].values

    for model_name in all_models:
        if model_name == target_model:
            continue

        df_base = instance_regrets_df[instance_regrets_df["Model"] == model_name].sort_values("instance_id")
        regrets_base = df_base["NormalizedRegret_pct"].values

        # Ensure matched instances
        n_common = min(len(regrets_base), len(regrets_ours))
        r_base = regrets_base[:n_common]
        r_ours = regrets_ours[:n_common]

        diffs = r_base - r_ours
        mean_diff = float(np.mean(diffs))

        # Paired t-test (alternative="greater": base > ours)
        if len(diffs) >= 2:
            if np.all(np.isclose(diffs, diffs[0])):
                if diffs[0] > 1e-9:
                    t_stat = float("inf")
                    t_pval = 0.0
                elif diffs[0] < -1e-9:
                    t_stat = float("-inf")
                    t_pval = 1.0
                else:
                    t_stat = 0.0
                    t_pval = 1.0
            else:
                try:
                    t_res = stats.ttest_rel(r_base, r_ours, alternative="greater")
                    t_stat = float(t_res.statistic)
                    t_pval = float(t_res.pvalue)
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    t_stat = 0.0
                    t_pval = 1.0
        else:
            t_stat = 0.0
            t_pval = 1.0

        # Wilcoxon signed-rank test
        nonzero_diffs = diffs[~np.isclose(diffs, 0.0)]
        if len(nonzero_diffs) >= 1:
            try:
                w_res = stats.wilcoxon(r_base, r_ours, alternative="greater", zero_method="wilcox")
                w_stat = float(w_res.statistic)
                w_pval = float(w_res.pvalue)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                w_stat = 0.0
                w_pval = 1.0
        else:
            w_stat = 0.0
            w_pval = 1.0

        sig_p05 = bool(w_pval < 0.05 or t_pval < 0.05)
        sig_p01 = bool(w_pval < 0.01 or t_pval < 0.01)

        significance_rows.append({
            "Baseline_Model": model_name,
            "Ours_Model": target_model,
            "Mean_Regret_Baseline_pct": float(np.mean(r_base)),
            "Mean_Regret_Ours_pct": float(np.mean(r_ours)),
            "Mean_Difference_pct": mean_diff,
            "Wilcoxon_Stat": w_stat,
            "Wilcoxon_p_value": w_pval,
            "T_Stat": t_stat,
            "T_p_value": t_pval,
            "Significant_p05": sig_p05,
            "Significant_p01": sig_p01,
        })

    return pd.DataFrame(significance_rows)


def run_benchmark(
    models: str | list[str] = "all",
    num_eval_instances: int = 30,
    num_train_instances: int = 140,
    epochs_dfl: int = 1000,
    batch_size: int = 8,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    transition_ratio: float = 0.8,
    topology: str = "C1",
    shelf_life_tau: int = 3,
    stockout_penalty_pu: float = 20.0,
    spoilage_penalty_ps: float = 5.0,
    holding_cost_h: float = 1.0,
    time_limit_sec: float = 2.5,
    output_dir: str = "results",
    save_results: bool = True,
    dry_run: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Executes the benchmark comparison suite.
    """
    np.random.seed(seed)

    if dry_run:
        num_eval_instances = min(num_eval_instances, 2)
        epochs_dfl = min(epochs_dfl, 2)
        num_train_instances = min(num_train_instances, 4)

    # Determine which models to evaluate
    if isinstance(models, str):
        if models.strip().lower() == "all":
            selected_models = list(ALL_14_MODELS)
        else:
            raw_list = [m.strip() for m in models.split(",") if m.strip()]
            selected_models = [canonicalize_model_name(m) for m in raw_list]
    else:
        selected_models = [canonicalize_model_name(m) for m in models]

    print("=" * 80)
    print(f"BENCHMARK HARNESS: Topology={topology}, Eval Instances={num_eval_instances}, Models={len(selected_models)}")
    print("=" * 80)

    network_file = f"data/processed/network_solomon_{topology.lower()}.json"
    df_train, df_val, df_test, config, dist_matrix, feat_cols = load_dataset(network_path=network_file)

    # Override config costs if specified
    config["default_shelf_life_tau"] = shelf_life_tau
    config["default_stockout_penalty_pu"] = stockout_penalty_pu
    config["spoilage_penalty_ps"] = spoilage_penalty_ps
    config["holding_cost_h"] = holding_cost_h

    N = config["num_stores"]
    T = config["horizon_T"]
    tau = shelf_life_tau
    pu = stockout_penalty_pu
    ps = spoilage_penalty_ps
    h = holding_cost_h
    Q = config["vehicle_capacity_Q"]
    Ci = config["store_capacity_Ci"]

    X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
    X_val_3d, d_val_3d = structure_instances(df_val, feat_cols, N=N, T=T)
    X_test_3d, d_test_3d = structure_instances(df_test, feat_cols, N=N, T=T)

    # Flat data for Two-Stage PTO baselines
    num_train_sub = min(num_train_instances, len(X_train_3d))
    X_train_sub = X_train_3d[:num_train_sub]
    d_train_sub = d_train_3d[:num_train_sub]
    X_train_flat = X_train_sub.reshape(-1, len(feat_cols))
    y_train_flat = d_train_sub.reshape(-1)

    eval_instances = min(num_eval_instances, len(X_test_3d))
    X_test_eval = X_test_3d[:eval_instances]
    d_test_eval = d_test_3d[:eval_instances]

    # Precompute Oracle solutions
    print(f"Precomputing Oracle hindsight solutions for {eval_instances} test instances...")
    oracle_costs = []
    for i in range(eval_instances):
        sol_oracle = solve_pirp(
            demand=d_test_eval[i],
            distance_matrix=dist_matrix,
            continuous_relaxation=False,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            vehicle_capacity_Q=Q,
            store_capacity_Ci=Ci,
            time_limit_sec=time_limit_sec,
        )
        oracle_costs.append(sol_oracle.total_cost)
    print(f"Oracle mean optimal cost: {np.mean(oracle_costs):.2f}\n")

    benchmark_rows = []
    all_instance_records = []

    # =============================================================
    # 1. Ridge_PTO
    # =============================================================
    if "Ridge_PTO" in selected_models:
        print("[Model] Training & Evaluating Ridge_PTO...")
        t0 = time.perf_counter()
        ridge = RidgePTO(alpha=10.0).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = ridge.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "Ridge_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 2. ElasticNet_PTO
    # =============================================================
    if "ElasticNet_PTO" in selected_models:
        print("[Model] Training & Evaluating ElasticNet_PTO...")
        t0 = time.perf_counter()
        enet = ElasticNetPTO(alpha=0.1, l1_ratio=0.5).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = enet.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "ElasticNet_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 3. RandomForest_PTO
    # =============================================================
    if "RandomForest_PTO" in selected_models:
        print("[Model] Training & Evaluating RandomForest_PTO...")
        t0 = time.perf_counter()
        rf = RandomForestPTO(n_estimators=100, max_depth=10, min_samples_split=5).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = rf.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "RandomForest_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 4. GradientBoosting_PTO
    # =============================================================
    if "GradientBoosting_PTO" in selected_models:
        print("[Model] Training & Evaluating GradientBoosting_PTO...")
        t0 = time.perf_counter()
        gb = GradientBoostingPTO(n_estimators=100, learning_rate=0.05, max_depth=4).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = gb.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "GradientBoosting_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 5. LightGBM_PTO
    # =============================================================
    if "LightGBM_PTO" in selected_models:
        print("[Model] Training & Evaluating LightGBM_PTO...")
        t0 = time.perf_counter()
        lgb_model = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = lgb_model.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "LightGBM_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 6. LightGBM_Quantile_PTO
    # =============================================================
    if "LightGBM_Quantile_PTO" in selected_models:
        print("[Model] Training & Evaluating LightGBM_Quantile_PTO...")
        t0 = time.perf_counter()
        lgb_q = LightGBMQuantilePTO(
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            holding_cost_h=h,
            shelf_life_tau=tau,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
        ).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = lgb_q.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "LightGBM_Quantile_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 7. SafetyStock_PTO
    # =============================================================
    if "SafetyStock_PTO" in selected_models:
        print("[Model] Training & Evaluating SafetyStock_PTO...")
        t0 = time.perf_counter()
        ss = SafetyStockPTO(
            alpha_ridge=1.0,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            holding_cost_h=h,
            shelf_life_tau=tau,
        ).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = ss.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "SafetyStock_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 8. MLP_PTO
    # =============================================================
    if "MLP_PTO" in selected_models:
        print("[Model] Training & Evaluating MLP_PTO...")
        t0 = time.perf_counter()
        mlp_epochs = 3 if dry_run else 100
        mlp = MLPPTO(in_features=len(feat_cols), hidden_dim=64, epochs=mlp_epochs, lr=1e-3).fit(
            X_train_flat, y_train_flat
        )
        train_time = time.perf_counter() - t0
        preds = mlp.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "MLP_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 9. TemporalSeqNet_PTO
    # =============================================================
    if "TemporalSeqNet_PTO" in selected_models:
        print("[Model] Training & Evaluating TemporalSeqNet_PTO...")
        t0 = time.perf_counter()
        seq_epochs = 3 if dry_run else 100
        seq_net = TemporalSeqNetPTO(
            in_features=len(feat_cols),
            num_stores=N,
            horizon_T=T,
            hidden_dim=32 if dry_run else 64,
            num_layers=2,
            rnn_type="LSTM",
            epochs=seq_epochs,
            batch_size=batch_size,
            patience=12,
        ).fit(X_train_sub, d_train_sub, X_val=X_val_3d, y_val=d_val_3d)
        train_time = time.perf_counter() - t0
        preds = seq_net.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "TemporalSeqNet_PTO", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 10. KNN_SAA
    # =============================================================
    if "KNN_SAA" in selected_models:
        print("[Model] Training & Evaluating KNN_SAA...")
        t0 = time.perf_counter()
        knn = KNNSAAModel(k_neighbors=5).fit(X_train_flat, y_train_flat)
        train_time = time.perf_counter() - t0
        preds = knn.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "KNN_SAA", "Two-Stage PTO", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 11. DFL_DPO
    # =============================================================
    if "DFL_DPO" in selected_models:
        print("[Model] Training & Evaluating DFL_DPO...")
        t0 = time.perf_counter()
        dpo_pipe = DFLPipeline(
            loss_type="DPO",
            in_features=len(feat_cols),
            num_stores=N,
            horizon_T=T,
            distance_matrix=dist_matrix,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
        )
        dpo_pipe.fit_two_phase(
            X_train=X_train_sub,
            d_train=d_train_sub,
            total_epochs=epochs_dfl,
            transition_ratio=transition_ratio,
            lr_phase1=lr_phase1,
            lr_phase2=lr_phase2,
            batch_size=batch_size,
            verbose=False,
        )
        train_time = time.perf_counter() - t0
        preds = dpo_pipe.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "DFL_DPO", "End-to-End DFL", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 12. DFL_IntOpt
    # =============================================================
    if "DFL_IntOpt" in selected_models:
        print("[Model] Training & Evaluating DFL_IntOpt...")
        t0 = time.perf_counter()
        intopt_pipe = DFLPipeline(
            loss_type="IntOpt",
            in_features=len(feat_cols),
            num_stores=N,
            horizon_T=T,
            distance_matrix=dist_matrix,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
        )
        intopt_pipe.fit_two_phase(
            X_train=X_train_sub,
            d_train=d_train_sub,
            total_epochs=epochs_dfl,
            transition_ratio=transition_ratio,
            lr_phase1=lr_phase1,
            lr_phase2=lr_phase2,
            batch_size=batch_size,
            verbose=False,
        )
        train_time = time.perf_counter() - t0
        preds = intopt_pipe.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "DFL_IntOpt", "End-to-End DFL", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 13. DFL_SPO+
    # =============================================================
    if "DFL_SPO+" in selected_models:
        print("[Model] Training & Evaluating DFL_SPO+...")
        t0 = time.perf_counter()
        spo_pipe = DFLPipeline(
            loss_type="SPO+",
            in_features=len(feat_cols),
            num_stores=N,
            horizon_T=T,
            distance_matrix=dist_matrix,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
        )
        spo_pipe.fit_two_phase(
            X_train=X_train_sub,
            d_train=d_train_sub,
            total_epochs=epochs_dfl,
            transition_ratio=transition_ratio,
            lr_phase1=lr_phase1,
            lr_phase2=lr_phase2,
            batch_size=batch_size,
            verbose=False,
        )
        train_time = time.perf_counter() - t0
        preds = spo_pipe.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "DFL_SPO+", "End-to-End DFL", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    # =============================================================
    # 14. Two-Phase_DFL (Ours)
    # =============================================================
    if "Two-Phase_DFL (Ours)" in selected_models:
        print("[Model] Training & Evaluating Two-Phase_DFL (Ours)...")
        t0 = time.perf_counter()
        two_phase_pipe = DFLPipeline(
            loss_type="SPO+",
            in_features=len(feat_cols),
            num_stores=N,
            horizon_T=T,
            distance_matrix=dist_matrix,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
        )
        two_phase_pipe.fit_two_phase(
            X_train=X_train_sub,
            d_train=d_train_sub,
            X_val=X_val_3d,
            d_val=d_val_3d,
            total_epochs=epochs_dfl,
            transition_ratio=transition_ratio,
            lr_phase1=lr_phase1,
            lr_phase2=lr_phase2,
            batch_size=batch_size,
            lambda_pinball=0.2,
            verbose=False,
        )
        train_time = time.perf_counter() - t0
        preds = two_phase_pipe.predict(X_test_eval)
        row, inst_recs = evaluate_model_decisions(
            "Two-Phase_DFL (Ours)", "Proposed Method", preds, d_test_eval, oracle_costs, dist_matrix,
            config, train_time, topology=topology, time_limit_sec=time_limit_sec
        )
        benchmark_rows.append(row)
        all_instance_records.extend(inst_recs)

    results_df = pd.DataFrame(benchmark_rows)
    instance_regrets_df = pd.DataFrame(all_instance_records)
    significance_df = compute_statistical_significance(instance_regrets_df, target_model="Two-Phase_DFL (Ours)")

    results_df.attrs["significance"] = significance_df
    results_df.attrs["instance_regrets"] = instance_regrets_df

    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, "benchmark_metrics.csv"), index=False)
        with open(os.path.join(output_dir, "benchmark_metrics.json"), "w") as f:
            json.dump(results_df.to_dict(orient="records"), f, indent=2)

        instance_regrets_df.to_csv(os.path.join(output_dir, "instance_regrets.csv"), index=False)

        significance_df.to_csv(os.path.join(output_dir, "statistical_significance.csv"), index=False)
        with open(os.path.join(output_dir, "statistical_significance.json"), "w") as f:
            json.dump(significance_df.to_dict(orient="records"), f, indent=2)

        print(f"\nResults successfully saved to `{output_dir}/` directory:")
        print(f"  - {output_dir}/benchmark_metrics.csv")
        print(f"  - {output_dir}/benchmark_metrics.json")
        print(f"  - {output_dir}/instance_regrets.csv")
        print(f"  - {output_dir}/statistical_significance.csv")
        print(f"  - {output_dir}/statistical_significance.json")

    print("\n=== FINAL BENCHMARK SUMMARY TABLE ===")
    display_cols = [c for c in ["Model", "NormalizedRegret_pct", "TotalCost", "StockoutCost", "SpoilageCost", "MSE", "TrainTime_sec"] if c in results_df.columns]
    if display_cols:
        print(results_df[display_cols].to_string(index=False))

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="PIRP Benchmark Execution Harness")
    parser.add_argument("--models", type=str, default="all", help="Models to evaluate: 'all' or comma-separated")
    parser.add_argument("--num_eval_instances", type=int, default=30, help="Number of test instances to evaluate")
    parser.add_argument("--num_train_instances", type=int, default=140, help="Number of training instances")
    parser.add_argument("--epochs_dfl", type=int, default=1000, help="Training epochs for DFL models")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr_phase1", type=float, default=1e-3, help="Learning rate for Phase 1")
    parser.add_argument("--lr_phase2", type=float, default=1e-4, help="Learning rate for Phase 2")
    parser.add_argument("--transition_ratio", type=float, default=0.8, help="LP warmup transition ratio")
    parser.add_argument("--topology", type=str, default="C1", choices=["C1", "R1", "RC1"], help="Solomon network topology")
    parser.add_argument("--shelf_life_tau", type=int, default=3, help="Perishable shelf life tau")
    parser.add_argument("--stockout_penalty_pu", type=float, default=20.0, help="Unmet demand stockout penalty pu")
    parser.add_argument("--spoilage_penalty_ps", type=float, default=5.0, help="Spoiled product waste penalty ps")
    parser.add_argument("--holding_cost_h", type=float, default=1.0, help="Inventory holding cost h")
    parser.add_argument("--time_limit_sec", type=float, default=2.5, help="Solver time limit per instance")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output metrics")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry run with 2 instances and 2 epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        models=args.models,
        num_eval_instances=args.num_eval_instances,
        num_train_instances=args.num_train_instances,
        epochs_dfl=args.epochs_dfl,
        batch_size=args.batch_size,
        lr_phase1=args.lr_phase1,
        lr_phase2=args.lr_phase2,
        transition_ratio=args.transition_ratio,
        topology=args.topology,
        shelf_life_tau=args.shelf_life_tau,
        stockout_penalty_pu=args.stockout_penalty_pu,
        spoilage_penalty_ps=args.spoilage_penalty_ps,
        holding_cost_h=args.holding_cost_h,
        time_limit_sec=args.time_limit_sec,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
    )
