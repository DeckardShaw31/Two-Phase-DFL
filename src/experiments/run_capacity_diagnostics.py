"""
Period-Level Capacity Diagnostics for Perishable Inventory-Routing.

Evaluates:
  1. Aggregate predicted demand vs. net replenishment need after carryover.
  2. Fleet capacity binding frequency Pr(sum(q_it) >= 0.98 * K * Q).
  3. Store-level stockout concentration (Gini index and max store stockout share).
  4. Operational recourse events (carryover inventory, realized stockout, realized spoilage).

Models evaluated:
  - Ridge_PTO
  - LightGBM_PTO
  - LightGBM_Quantile_PTO
  - SafetyStock_PTO
  - Two-Phase_DFL (Ours)

Outputs:
  - results/capacity_diagnostics.csv
"""

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.run_benchmark import load_dataset, structure_instances
from src.models.baselines import (
    LightGBMPTO,
    LightGBMQuantilePTO,
    RidgePTO,
    SafetyStockPTO,
)
from src.models.dfl_models import DFLPipeline
from src.optimization.pirp_model import solve_pirp
from src.utils.metrics import simulate_perishable_inventory_fifo


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Computes Gini concentration coefficient across stores for a single period."""
    val = np.asarray(values, dtype=np.float64).flatten()
    s = float(np.sum(val))
    if s <= 1e-6:
        return 0.0
    n = len(val)
    diff_sum = float(np.sum(np.abs(val[:, None] - val[None, :])))
    return diff_sum / (2.0 * n * s)


def run_capacity_diagnostics(
    models_to_evaluate: list[str] | None = None,
    num_eval_instances: int = 30,
    topology: str = "C1",
    output_path: str = "results/capacity_diagnostics.csv",
    store_output_path: str = "results/store_level_diagnostics.csv",
    save_results: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes period-level and store-level capacity diagnostics across test instances.
    """
    np.random.seed(seed)
    if models_to_evaluate is None:
        models_to_evaluate = [
            "Ridge_PTO",
            "LightGBM_PTO",
            "LightGBM_Quantile_PTO",
            "SafetyStock_PTO",
            "Two-Phase_DFL (Ours)",
        ]

    print("=" * 80)
    print("RUNNING PERIOD-LEVEL CAPACITY DIAGNOSTICS")
    print(f"Topology: {topology}, Test Instances: {num_eval_instances}, Models: {models_to_evaluate}")
    print("=" * 80)

    network_file = f"data/processed/network_solomon_{topology.lower()}.json"
    df_train, df_val, df_test, config, dist_matrix, feat_cols = load_dataset(network_path=network_file)

    N = config["num_stores"]
    T = config["horizon_T"]
    tau = config["default_shelf_life_tau"]
    pu = config["default_stockout_penalty_pu"]
    ps = config["spoilage_penalty_ps"]
    K = config["fleet_size_K"]
    Q = config["vehicle_capacity_Q"]
    Ci = config["store_capacity_Ci"]
    fleet_cap_KQ = float(K * Q)  # 200.0

    X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
    X_val_3d, d_val_3d = structure_instances(df_val, feat_cols, N=N, T=T)
    X_test_3d, d_test_3d = structure_instances(df_test, feat_cols, N=N, T=T)

    X_train_flat = df_train[feat_cols].values
    y_train_flat = df_train["demand"].values

    eval_k = min(num_eval_instances, len(X_test_3d))
    X_eval = X_test_3d[:eval_k]
    d_eval = d_test_3d[:eval_k]

    # Pre-train / fit each requested model
    fitted_models: dict[str, Any] = {}
    model_predictions: dict[str, np.ndarray] = {}

    for m_name in models_to_evaluate:
        print(f"Preparing model: {m_name}...")
        if m_name == "Ridge_PTO":
            r_model = RidgePTO(alpha=10.0).fit(X_train_flat, y_train_flat)
            preds = np.zeros_like(d_eval)
            for i in range(eval_k):
                p_flat = r_model.predict(X_eval[i].reshape(N * T, -1))
                preds[i] = p_flat.reshape(N, T)
            fitted_models[m_name] = r_model
            model_predictions[m_name] = preds

        elif m_name == "LightGBM_PTO":
            lgb = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_train_flat, y_train_flat)
            preds = np.zeros_like(d_eval)
            for i in range(eval_k):
                p_flat = lgb.predict(X_eval[i].reshape(N * T, -1))
                preds[i] = p_flat.reshape(N, T)
            fitted_models[m_name] = lgb
            model_predictions[m_name] = preds

        elif m_name == "LightGBM_Quantile_PTO":
            # Multi-period critical fractile beta_tau^* = pu / (pu + h + ps/tau) = 20 / (20 + 1 + 5/3) ≈ 0.882
            lgb_q = LightGBMQuantilePTO(
                critical_fractile=0.882,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
            ).fit(X_train_flat, y_train_flat)
            preds = np.zeros_like(d_eval)
            for i in range(eval_k):
                p_flat = lgb_q.predict(X_eval[i].reshape(N * T, -1))
                preds[i] = p_flat.reshape(N, T)
            fitted_models[m_name] = lgb_q
            model_predictions[m_name] = preds

        elif m_name == "SafetyStock_PTO":
            ss = SafetyStockPTO(
                alpha_ridge=10.0,
                stockout_penalty_pu=pu,
                spoilage_penalty_ps=ps,
                holding_cost_h=config.get("holding_cost_h", 1.0),
                shelf_life_tau=tau,
                z_factor=1.65,
            ).fit(X_train_flat, y_train_flat)
            preds = np.zeros_like(d_eval)
            for i in range(eval_k):
                p_flat = ss.predict(X_eval[i].reshape(N * T, -1))
                preds[i] = p_flat.reshape(N, T)
            fitted_models[m_name] = ss
            model_predictions[m_name] = preds

        elif m_name in ("Two-Phase_DFL (Ours)", "TwoPhase_DFL"):
            dfl_pipe = DFLPipeline(
                loss_type="SPO+",
                in_features=len(feat_cols),
                num_stores=N,
                horizon_T=T,
                distance_matrix=dist_matrix,
                shelf_life_tau=tau,
                stockout_penalty_pu=pu,
                spoilage_penalty_ps=ps,
            )
            dfl_pipe.fit_two_phase(
                X_train=X_train_3d[:140],
                d_train=d_train_3d[:140],
                X_val=X_val_3d[:30],
                d_val=d_val_3d[:30],
                total_epochs=200,
                transition_ratio=0.8,
                batch_size=8,
                lambda_pinball=0.2,
                val_subset_size=30,
                verbose=False,
            )
            preds = dfl_pipe.predict(X_eval)
            fitted_models[m_name] = dfl_pipe
            model_predictions[m_name] = preds

    # Period-by-period diagnostics and store-level diagnostics
    diagnostics_rows = []
    store_diagnostics_rows = []

    for m_name in models_to_evaluate:
        preds = model_predictions[m_name]
        print(f"Computing capacity diagnostics for: {m_name}...")

        # Per-instance records across time
        # shape: (eval_k, T)
        inst_agg_pred = np.zeros((eval_k, T), dtype=np.float64)
        inst_net_need = np.zeros((eval_k, T), dtype=np.float64)
        inst_delivered = np.zeros((eval_k, T), dtype=np.float64)
        inst_cap_binding = np.zeros((eval_k, T), dtype=np.float64)
        inst_carryover = np.zeros((eval_k, T), dtype=np.float64)
        inst_stockout = np.zeros((eval_k, T), dtype=np.float64)
        inst_spoilage = np.zeros((eval_k, T), dtype=np.float64)
        inst_gini = np.zeros((eval_k, T), dtype=np.float64)
        inst_max_share = np.zeros((eval_k, T), dtype=np.float64)

        # Store-level accumulators across all instances and periods
        store_deliv_sum = np.zeros(N, dtype=np.float64)
        store_deliv_count = np.zeros(N, dtype=np.float64)
        store_carry_sum = np.zeros(N, dtype=np.float64)
        store_stockout_sum = np.zeros(N, dtype=np.float64)
        store_demand_sum = np.zeros(N, dtype=np.float64)

        for i in range(eval_k):
            # Solve PIRP for instance i
            sol = solve_pirp(
                demand=preds[i],
                distance_matrix=dist_matrix,
                continuous_relaxation=False,
                shelf_life_tau=tau,
                stockout_penalty_pu=pu,
                spoilage_penalty_ps=ps,
                vehicle_capacity_Q=Q,
                store_capacity_Ci=Ci,
                time_limit_sec=2.5,
            )
            q_plan = sol.q_total  # shape (N, T)

            # Simulate exact FIFO inventory lifecycle
            I_on_hand, y_pos, y_neg, _ = simulate_perishable_inventory_fifo(
                q_delivered=q_plan,
                true_demand=d_eval[i],
                tau=tau,
                initial_inventory=None,
            )

            # Store-level accumulation
            store_deliv_sum += np.sum(q_plan, axis=1)
            store_deliv_count += np.sum(q_plan > 1e-4, axis=1)
            store_carry_sum += np.sum(I_on_hand, axis=(1, 2))
            store_stockout_sum += np.sum(y_pos, axis=1)
            store_demand_sum += np.sum(d_eval[i], axis=1)

            # Analyze period by period
            for t in range(T):
                # Aggregate predicted demand in period t
                agg_p = float(np.sum(preds[i, :, t]))
                inst_agg_pred[i, t] = agg_p

                # Incoming carryover stock available at start of period t before delivery
                if t == 0:
                    incoming_stock = np.zeros(N, dtype=np.float64)
                else:
                    # Sum across all valid shelf life ages at end of period t-1
                    incoming_stock = np.sum(I_on_hand[:, t - 1, :], axis=1)

                # Net replenishment need at store level: (d_hat_{i,t} - I_{i, t-1})_+
                net_need = np.maximum(0.0, preds[i, :, t] - incoming_stock)
                inst_net_need[i, t] = float(np.sum(net_need))

                # Delivered quantity in period t
                deliv = float(np.sum(q_plan[:, t]))
                inst_delivered[i, t] = deliv

                # Capacity binding flag: sum(q_it) >= 0.98 * K * Q
                is_binding = 1.0 if deliv >= 0.98 * fleet_cap_KQ else 0.0
                inst_cap_binding[i, t] = is_binding

                # Carryover inventory on hand at end of period t
                carry = float(np.sum(I_on_hand[:, t, :]))
                inst_carryover[i, t] = carry

                # Realized operational recourse
                stk = float(np.sum(y_pos[:, t]))
                spl = float(np.sum(y_neg[:, t]))
                inst_stockout[i, t] = stk
                inst_spoilage[i, t] = spl

                # Store-level stockout concentration
                inst_gini[i, t] = compute_gini_coefficient(y_pos[:, t])
                if stk > 1e-6:
                    inst_max_share[i, t] = float(np.max(y_pos[:, t])) / stk
                else:
                    inst_max_share[i, t] = 0.0

        # Aggregate across instances for each period t
        for t in range(T):
            diagnostics_rows.append({
                "Model": m_name,
                "Period_t": t + 1,
                "Agg_Predicted_Demand": float(np.mean(inst_agg_pred[:, t])),
                "Net_Replenishment_Need": float(np.mean(inst_net_need[:, t])),
                "Delivered_Quantity": float(np.mean(inst_delivered[:, t])),
                "Fleet_Capacity_KQ": fleet_cap_KQ,
                "Capacity_Binding_Freq": float(np.mean(inst_cap_binding[:, t])),
                "Carryover_Inventory": float(np.mean(inst_carryover[:, t])),
                "Realized_Stockout": float(np.mean(inst_stockout[:, t])),
                "Realized_Spoilage": float(np.mean(inst_spoilage[:, t])),
                "Store_Stockout_Concentration_Gini": float(np.mean(inst_gini[:, t])),
                "Max_Store_Stockout_Share": float(np.mean(inst_max_share[:, t])),
            })

        # Store-level aggregation across all evaluation instances
        total_store_eval_periods = float(eval_k * T)
        tot_model_stockout = float(np.sum(store_stockout_sum))
        tot_model_demand = float(np.sum(store_demand_sum))

        for s in range(N):
            stk_share = float(store_stockout_sum[s] / tot_model_stockout) if tot_model_stockout > 1e-6 else 0.0
            dem_share = float(store_demand_sum[s] / tot_model_demand) if tot_model_demand > 1e-6 else 0.0
            rationing_idx = float(stk_share / dem_share) if dem_share > 1e-6 else 0.0
            store_diagnostics_rows.append({
                "Model": m_name,
                "Store_ID": s + 1,
                "Depot_Distance": float(dist_matrix[0, s + 1]),
                "Delivery_Frequency": float(store_deliv_count[s] / total_store_eval_periods),
                "Avg_Delivered_Qty": float(store_deliv_sum[s] / total_store_eval_periods),
                "Avg_Carryover": float(store_carry_sum[s] / total_store_eval_periods),
                "Avg_Stockout": float(store_stockout_sum[s] / total_store_eval_periods),
                "Avg_Demand": float(store_demand_sum[s] / total_store_eval_periods),
                "Stockout_Share": stk_share,
                "Demand_Share": dem_share,
                "Peripheral_Rationing_Index": rationing_idx,
            })

    df_diag = pd.DataFrame(diagnostics_rows)
    df_store = pd.DataFrame(store_diagnostics_rows)

    if save_results:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_diag.to_csv(output_path, index=False)
        print(f"\nSaved capacity diagnostics to: {output_path}")

        os.makedirs(os.path.dirname(store_output_path), exist_ok=True)
        df_store.to_csv(store_output_path, index=False)
        print(f"Saved store-level diagnostics ({len(df_store)} rows) to: {store_output_path}")

    df_diag.attrs["store_diagnostics"] = df_store

    # Summary report
    print("\n=== CAPACITY DIAGNOSTICS SUMMARY (AVERAGED OVER PERIODS) ===")
    summary = df_diag.groupby("Model")[
        [
            "Agg_Predicted_Demand",
            "Net_Replenishment_Need",
            "Delivered_Quantity",
            "Capacity_Binding_Freq",
            "Carryover_Inventory",
            "Realized_Stockout",
            "Store_Stockout_Concentration_Gini",
            "Max_Store_Stockout_Share",
        ]
    ].mean()
    print(summary.to_string())

    print("\n=== STORE-LEVEL DIAGNOSTICS SUMMARY (AVERAGED OVER STORES) ===")
    store_summary = df_store.groupby("Model")[
        [
            "Delivery_Frequency",
            "Avg_Delivered_Qty",
            "Avg_Carryover",
            "Avg_Stockout",
            "Peripheral_Rationing_Index",
        ]
    ].mean()
    print(store_summary.to_string())

    return df_diag


if __name__ == "__main__":
    run_capacity_diagnostics()
