# HISTORICAL: superseded legacy experiment script. Do not use its outputs as evidence for the revised manuscript; see src/experiments/LEGACY_README.md and use run_revision_benchmark.py.
"""
Empirical Profiling of Two-Phase LP Warmup Acceleration vs Direct MILP.

Compares:
  (a) TwoPhase_LP_Warmup: 75% epochs on continuous LP relaxation, then MILP fine-tuning.
  (b) Direct_MILP: 100% epochs directly on discrete MILP.
  (c) Pure_LP: Continuous LP relaxation throughout all epochs.

Outputs:
  - results/lp_warmup_speedup.csv
  - results/figures/fig_speedup_pareto.png
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd

from src.experiments.run_benchmark import load_dataset, structure_instances
from src.models.dfl_models import DFLPipeline
from src.utils.plotting import plot_speedup_pareto


def run_lp_warmup_experiment(
    total_epochs: int = 12,
    num_train_instances: int = 20,
    num_val_instances: int = 5,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Executes the LP-Warmup acceleration experiment, profiling solve times and convergence.
    """
    print("=" * 70)
    print("STARTING LP-WARMUP ACCELERATION PROFILING EXPERIMENT")
    print(f"Total Epochs: {total_epochs}, Train Instances: {num_train_instances}")
    print("=" * 70)

    df_train, df_val, _df_test, config, dist_matrix, feat_cols = load_dataset()
    N = config["num_stores"]
    T = config["horizon_T"]
    tau = config["default_shelf_life_tau"]
    pu = config["default_stockout_penalty_pu"]
    ps = config["spoilage_penalty_ps"]

    X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
    X_val_3d, d_val_3d = structure_instances(df_val, feat_cols, N=N, T=T)

    X_tr = X_train_3d[:num_train_instances]
    d_tr = d_train_3d[:num_train_instances]
    X_vl = X_val_3d[:num_val_instances]
    d_vl = d_val_3d[:num_val_instances]

    speedup_records = []

    # -------------------------------------------------------------
    # 1. Proposed: Two-Phase Training (LP Warmup + MILP Fine-Tuning)
    # -------------------------------------------------------------
    print("\n[Strategy A] Profiling Two-Phase Schedule (Proposed)...")
    pipe_two_phase = DFLPipeline(
        loss_type="SPO+",
        in_features=len(feat_cols),
        num_stores=N,
        horizon_T=T,
        distance_matrix=dist_matrix,
        shelf_life_tau=tau,
        stockout_penalty_pu=pu,
        spoilage_penalty_ps=ps,
    )
    hist_tp = pipe_two_phase.fit_two_phase(
        X_train=X_tr,
        d_train=d_tr,
        X_val=X_vl,
        d_val=d_vl,
        total_epochs=total_epochs,
        transition_ratio=0.75,
        lr_phase1=1e-3,
        lr_phase2=1e-4,
        batch_size=4,
        verbose=True,
    )
    for row in hist_tp:
        speedup_records.append({
            "Epoch": row["epoch"],
            "Method": "TwoPhase_LP_Warmup",
            "Phase": row["phase"],
            "Loss": row["loss"],
            "ValTotalCost": row["val_total_cost"],
            "EpochTime_sec": row["epoch_time_sec"],
            "CumulativeTime_sec": row["cumulative_time_sec"],
        })

    # -------------------------------------------------------------
    # 2. Baseline: Direct Full MILP Training
    # -------------------------------------------------------------
    print("\n[Strategy B] Profiling Direct Full MILP Training (Ablation)...")
    pipe_direct = DFLPipeline(
        loss_type="SPO+",
        in_features=len(feat_cols),
        num_stores=N,
        horizon_T=T,
        distance_matrix=dist_matrix,
        shelf_life_tau=tau,
        stockout_penalty_pu=pu,
        spoilage_penalty_ps=ps,
    )
    # Direct MILP: 0% LP warmup, 100% MILP
    hist_direct = pipe_direct.fit_direct_milp(
        X_train=X_tr,
        d_train=d_tr,
        X_val=X_vl,
        d_val=d_vl,
        total_epochs=total_epochs,
        lr=1e-4,
        batch_size=4,
        verbose=True,
    )
    for row in hist_direct:
        speedup_records.append({
            "Epoch": row["epoch"],
            "Method": "Direct_MILP",
            "Phase": "Direct_MILP",
            "Loss": row["loss"],
            "ValTotalCost": row["val_total_cost"],
            "EpochTime_sec": row["epoch_time_sec"],
            "CumulativeTime_sec": row["cumulative_time_sec"],
        })

    df_speedup = pd.DataFrame(speedup_records)

    # Compute overall speedup factor
    total_time_tp = df_speedup[df_speedup["Method"] == "TwoPhase_LP_Warmup"]["EpochTime_sec"].sum()
    total_time_milp = df_speedup[df_speedup["Method"] == "Direct_MILP"]["EpochTime_sec"].sum()
    speedup_factor = total_time_milp / max(1e-5, total_time_tp)

    print("\n=== SPEEDUP PROFILING SUMMARY ===")
    print(f"Total Two-Phase Training Time: {total_time_tp:.2f}s")
    print(f"Total Direct MILP Training Time: {total_time_milp:.2f}s")
    print(f"Empirical Wall-Clock Speedup Factor: {speedup_factor:.2f}x")

    if save_results:
        os.makedirs("results/figures", exist_ok=True)
        csv_path = "results/lp_warmup_speedup.csv"
        df_speedup.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Plot Pareto convergence curve
        plot_speedup_pareto(df_speedup, "results/figures/fig_speedup_pareto.png")

    return df_speedup


if __name__ == "__main__":
    run_lp_warmup_experiment(total_epochs=20, num_train_instances=24, num_val_instances=10)
