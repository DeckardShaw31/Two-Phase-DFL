"""
Sensitivity Analysis across Asymmetric Penalties, Shelf Lives, Solomon Topologies,
and Vehicle Capacity Tightness (Full 81-Configuration Factorial Grid).

Factorial Grid Dimensions:
  1. Topologies (3): C1 (Clustered), R1 (Random), RC1 (Random-Clustered)
  2. Shelf lives (3): tau in {2, 3, 5}
  3. Cost asymmetry ratios (3): pu/h in {10.0, 20.0, 50.0} (ps/h = 5.0, h = 1.0)
  4. Vehicle capacity tightness (3): gamma in {0.8, 1.0, 1.5}, i.e. Q in {80.0, 100.0, 150.0}

Total Factorial Configurations: 3 x 3 x 3 x 3 = 81 unique parameter tuples.
Models evaluated:
  - Ridge_PTO
  - LightGBM_PTO
  - Two-Phase_DFL (Ours)

Outputs:
  - results/sensitivity_analysis.csv (243 rows: 81 configurations x 3 models)
  - results/figures/fig_sensitivity_heatmap.png
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.run_benchmark import load_dataset, structure_instances
from src.models.baselines import LightGBMPTO, RidgePTO
from src.models.dfl_models import DFLPipeline
from src.optimization.pirp_model import solve_pirp
from src.utils.metrics import (
    compute_decision_regret,
    evaluate_decisions_under_true_demand,
)
from src.utils.plotting import plot_sensitivity_heatmap


def evaluate_single_instance(
    inst_idx: int,
    d_true: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    dist_matrix: np.ndarray,
    tau: int,
    pu: float,
    ps: float,
    h: float,
    Q: float,
    Ci: float,
    time_limit_sec: float = 2.0,
) -> dict[str, Any]:
    """Evaluates Oracle and model predictions for a single instance under specific (tau, pu, Q)."""
    # Oracle solution under true demand
    sol_oracle = solve_pirp(
        demand=d_true,
        distance_matrix=dist_matrix,
        continuous_relaxation=False,
        shelf_life_tau=tau,
        stockout_penalty_pu=pu,
        spoilage_penalty_ps=ps,
        vehicle_capacity_Q=Q,
        store_capacity_Ci=Ci,
        time_limit_sec=time_limit_sec,
    )
    c_oracle = sol_oracle.total_cost

    instance_res = {}
    for m_name, pred_arr in preds_dict.items():
        sol_m = solve_pirp(
            demand=pred_arr,
            distance_matrix=dist_matrix,
            continuous_relaxation=False,
            shelf_life_tau=tau,
            stockout_penalty_pu=pu,
            spoilage_penalty_ps=ps,
            vehicle_capacity_Q=Q,
            store_capacity_Ci=Ci,
            time_limit_sec=time_limit_sec,
        )
        metrics_m = evaluate_decisions_under_true_demand(
            sol_m.q_total,
            sol_m.x_routes,
            d_true,
            dist_matrix,
            h=h,
            p_s=ps,
            p_u=pu,
            tau=tau,
            store_capacity_Ci=Ci,
        )
        _, norm_regret = compute_decision_regret(metrics_m["total_cost"], c_oracle)
        instance_res[m_name] = {
            "norm_regret": norm_regret,
            "total_cost": metrics_m["total_cost"],
            "routing_cost": metrics_m["routing_cost"],
            "spoilage_cost": metrics_m["spoilage_cost"],
            "stockout_cost": metrics_m["stockout_cost"],
        }

    return instance_res


def run_sensitivity_experiments(
    pu_ratios: list[float] | None = None,
    tau_list: list[int] | None = None,
    topologies: list[str] | None = None,
    vehicle_capacities: list[float] | None = None,
    num_eval_instances: int = 10,
    save_results: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Executes complete 81-configuration factorial grid sensitivity experiments.
    """
    np.random.seed(seed)
    if topologies is None:
        topologies = ["C1", "R1", "RC1"]
    if tau_list is None:
        tau_list = [2, 3, 5]
    if pu_ratios is None:
        pu_ratios = [10.0, 20.0, 50.0]
    if vehicle_capacities is None:
        vehicle_capacities = [80.0, 100.0, 150.0]

    num_configs = len(topologies) * len(tau_list) * len(pu_ratios) * len(vehicle_capacities)
    print("=" * 80)
    print("STARTING FULL 81-CONFIGURATION FACTORIAL SENSITIVITY GRID EXPERIMENTS")
    print(f"Topologies ({len(topologies)}): {topologies}")
    print(f"Shelf Lives tau ({len(tau_list)}): {tau_list}")
    print(f"Cost Asymmetry Ratios pu/h ({len(pu_ratios)}): {pu_ratios}")
    print(f"Vehicle Capacities Q ({len(vehicle_capacities)}): {vehicle_capacities}")
    print(f"Total Unique Configurations: {num_configs} (Target = 81)")
    print(f"Evaluation instances per configuration: {num_eval_instances}")
    print("=" * 80)

    sensitivity_records = []
    config_counter = 0

    for topo in topologies:
        net_file = f"data/processed/network_solomon_{topo.lower()}.json"
        df_train, df_val, df_test, config, dist_matrix, feat_cols = load_dataset(network_path=net_file)

        N = config["num_stores"]
        T = config["horizon_T"]
        ps = config["spoilage_penalty_ps"]
        h = config["holding_cost_h"]
        Ci = config["store_capacity_Ci"]

        X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
        X_val_3d, d_val_3d = structure_instances(df_val, feat_cols, N=N, T=T)
        X_test_3d, d_test_3d = structure_instances(df_test, feat_cols, N=N, T=T)

        X_train_flat = df_train[feat_cols].values
        y_train_flat = df_train["demand"].values

        eval_k = min(num_eval_instances, len(X_test_3d))
        X_eval = X_test_3d[:eval_k]
        d_eval = d_test_3d[:eval_k]

        # Fit PTO point baselines once per topology
        ridge = RidgePTO(alpha=10.0).fit(X_train_flat, y_train_flat)
        lgb = LightGBMPTO(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_train_flat, y_train_flat)

        preds_ridge = np.zeros_like(d_eval)
        preds_lgb = np.zeros_like(d_eval)
        for i in range(eval_k):
            rf = ridge.predict(X_eval[i].reshape(N * T, -1))
            lf = lgb.predict(X_eval[i].reshape(N * T, -1))
            preds_ridge[i] = rf.reshape(N, T)
            preds_lgb[i] = lf.reshape(N, T)

        for tau in tau_list:
            for pu in pu_ratios:
                print(f"\n--- Training Two-Phase DFL for Topology={topo}, tau={tau}, pu/h={pu:.0f} ---")
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
                    X_val=X_val_3d[:10],
                    d_val=d_val_3d[:10],
                    total_epochs=120,
                    transition_ratio=0.8,
                    batch_size=16,
                    verbose=False,
                )
                preds_dfl = dfl_pipe.predict(X_eval)

                # Now evaluate across the 3 vehicle capacity tightness levels Q in {80, 100, 150}
                for Q in vehicle_capacities:
                    config_counter += 1
                    gamma = round(float(Q) / 100.0, 2)
                    print(
                        f"  [Config {config_counter}/81] Topo={topo}, tau={tau}, pu/h={pu:.0f}, "
                        f"Q={Q:.0f} (gamma={gamma:.1f})...",
                        end="",
                        flush=True,
                    )

                    regrets = {"Ridge_PTO": [], "LightGBM_PTO": [], "Two-Phase_DFL (Ours)": []}
                    total_costs = {"Ridge_PTO": [], "LightGBM_PTO": [], "Two-Phase_DFL (Ours)": []}
                    route_costs = {"Ridge_PTO": [], "LightGBM_PTO": [], "Two-Phase_DFL (Ours)": []}
                    stock_costs = {"Ridge_PTO": [], "LightGBM_PTO": [], "Two-Phase_DFL (Ours)": []}
                    spoil_costs = {"Ridge_PTO": [], "LightGBM_PTO": [], "Two-Phase_DFL (Ours)": []}

                    # Sequential evaluation across instances (fast and thread-safe)
                    for i in range(eval_k):
                        preds_dict = {
                            "Ridge_PTO": preds_ridge[i],
                            "LightGBM_PTO": preds_lgb[i],
                            "Two-Phase_DFL (Ours)": preds_dfl[i],
                        }
                        res = evaluate_single_instance(
                            i,
                            d_eval[i],
                            preds_dict,
                            dist_matrix,
                            tau,
                            pu,
                            ps,
                            h,
                            Q,
                            Ci,
                            2.0,
                        )
                        for m_name in ["Ridge_PTO", "LightGBM_PTO", "Two-Phase_DFL (Ours)"]:
                            regrets[m_name].append(res[m_name]["norm_regret"])
                            total_costs[m_name].append(res[m_name]["total_cost"])
                            route_costs[m_name].append(res[m_name]["routing_cost"])
                            stock_costs[m_name].append(res[m_name]["stockout_cost"])
                            spoil_costs[m_name].append(res[m_name]["spoilage_cost"])

                    for m_name in ["Ridge_PTO", "LightGBM_PTO", "Two-Phase_DFL (Ours)"]:
                        sensitivity_records.append({
                            "Topology": topo,
                            "ShelfLife_tau": tau,
                            "StockoutRatio_pu_h": pu / h,
                            "VehicleCapacity_Q": Q,
                            "CapacityTightness_gamma": gamma,
                            "Model": m_name,
                            "NormalizedRegret_pct": float(np.mean(regrets[m_name])),
                            "Regret_std": float(np.std(regrets[m_name])),
                            "TotalCost": float(np.mean(total_costs[m_name])),
                            "RoutingCost": float(np.mean(route_costs[m_name])),
                            "SpoilageCost": float(np.mean(spoil_costs[m_name])),
                            "StockoutCost": float(np.mean(stock_costs[m_name])),
                        })

                    print(
                        f" DFL={np.mean(regrets['Two-Phase_DFL (Ours)']):.1f}% vs "
                        f"LGB={np.mean(regrets['LightGBM_PTO']):.1f}%"
                    )

    df_sens = pd.DataFrame(sensitivity_records)

    # Verification of 81 unique parameter combinations
    param_tuples = df_sens[["Topology", "ShelfLife_tau", "StockoutRatio_pu_h", "VehicleCapacity_Q"]].drop_duplicates()
    print(f"\nCompleted {len(df_sens)} evaluation rows across {len(param_tuples)} unique parameter configurations.")
    assert len(param_tuples) == 81, f"Expected 81 unique configurations, got {len(param_tuples)}"

    if save_results:
        os.makedirs("results/figures", exist_ok=True)
        csv_path = "results/sensitivity_analysis.csv"
        df_sens.to_csv(csv_path, index=False)
        print(f"Saved full 81-configuration sensitivity analysis to: {csv_path}")

        # Regenerate sensitivity heatmap
        plot_sensitivity_heatmap(df_sens, "results/figures/fig_sensitivity_heatmap.png")

    return df_sens


if __name__ == "__main__":
    run_sensitivity_experiments(
        pu_ratios=[10.0, 20.0, 50.0],
        tau_list=[2, 3, 5],
        topologies=["C1", "R1", "RC1"],
        vehicle_capacities=[80.0, 100.0, 150.0],
        num_eval_instances=10,
    )
