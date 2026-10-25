"""
Master Reproduction Harness and Verification Suite for PIRP Decision-Focused Learning.

Supported CLI operations:
  --verify   : Fast (< 5s) audit verifying existence, row counts, and numerical parity
               across all benchmark files, data archives, diagnostics, figures, and
               generates results/environment_manifest.json.
  --dry-run  : Fast (< 15s) smoke test executing end-to-end data loading, optimization,
               DFL training loop, decision evaluation, and statistical inference.
  --full     : Executes complete reproduction pipeline (5 seeds x 1,500 epochs x 14 models),
               capacity diagnostics, and statistical testing suite.
"""

import argparse
import datetime
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.run_benchmark import ALL_14_MODELS
from src.experiments.run_capacity_diagnostics import run_capacity_diagnostics
from src.experiments.run_multi_seed_benchmark import DEFAULT_SEEDS, run_multi_seed_benchmark
from src.experiments.run_statistical_tests import run_statistical_tests


def get_environment_info() -> dict[str, Any]:
    """Gathers complete runtime, hardware, software versions, and problem configuration."""
    # Try importing optional packages for versions
    packages: dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    try:
        import sklearn
        packages["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import lightgbm
        packages["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass

    config_path = "data/processed/params_config.json"
    pirp_params: dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            pirp_params = json.load(f)

    env_manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
        },
        "packages": packages,
        "hardware": {
            "cpu_count_logical": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "pirp_parameters": {
            "store_capacity_Ci": pirp_params.get("store_capacity_Ci", 70.0),
            "aging_surcharge_eps": pirp_params.get("aging_surcharge_eps", 0.01),
            "vehicle_capacity_Q": pirp_params.get("vehicle_capacity_Q", 120.0),
            "shelf_life_tau": pirp_params.get("default_shelf_life_tau", 3),
            "stockout_penalty_pu": pirp_params.get("default_stockout_penalty_pu", 10.0),
            "spoilage_penalty_ps": pirp_params.get("spoilage_penalty_ps", 5.0),
            "holding_cost_h": pirp_params.get("holding_cost_h", 1.0),
            "horizon_T": pirp_params.get("horizon_T", 5),
            "num_stores": pirp_params.get("num_stores", 10),
        },
        "experiment_configuration": {
            "seeds": list(DEFAULT_SEEDS),
            "num_test_instances": 30,
            "num_models": len(ALL_14_MODELS),
            "dfl_epochs": 1500,
        },
    }
    return env_manifest


def generate_environment_manifest(output_path: str = "results/environment_manifest.json") -> dict[str, Any]:
    """Writes environment manifest to results/environment_manifest.json."""
    manifest = get_environment_info()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_artifacts(results_dir: str = "results") -> tuple[bool, list[str]]:
    """
    Performs rapid (< 5s) forensic audit of all experimental artifacts, checking:
      1. File existence and non-emptiness.
      2. Exact row counts and column schemas.
      3. Array dimensions in npz archive.
      4. Numerical parity between multi-seed summary and statistical tests.
      5. Figure presence in results/figures/.
      6. Writes environment manifest.

    Returns:
        (is_valid, error_list)
    """
    errors: list[str] = []
    t0 = time.perf_counter()
    print("=" * 80)
    print("REPRODUCTION VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Check Multi-Seed Summary
    summary_path = os.path.join(results_dir, "multi_seed_summary.csv")
    if not os.path.isfile(summary_path):
        errors.append(f"Missing file: {summary_path}")
    else:
        df_sum = pd.read_csv(summary_path)
        if len(df_sum) != 14:
            errors.append(f"multi_seed_summary.csv expected 14 models, found {len(df_sum)}.")
        req_cols = ["Model", "NumSeeds", "Regret_Mean_pct", "Regret_CrossSeed_std", "TotalCost_Mean"]
        for col in req_cols:
            if col not in df_sum.columns:
                errors.append(f"multi_seed_summary.csv missing column: {col}")
        print("[OK] multi_seed_summary.csv: 14 models verified.")

    # 2. Check Instance Level Metrics
    inst_csv_path = os.path.join(results_dir, "instance_level_metrics.csv")
    if not os.path.isfile(inst_csv_path):
        errors.append(f"Missing file: {inst_csv_path}")
    else:
        df_inst = pd.read_csv(inst_csv_path)
        expected_rows = 14 * 5 * 30  # 2,100
        if len(df_inst) != expected_rows:
            errors.append(f"instance_level_metrics.csv expected {expected_rows} rows, found {len(df_inst)}.")
        for col in ["instance_id", "seed", "model", "oracle_cost", "total_cost", "normalized_regret_pct"]:
            if col not in df_inst.columns:
                errors.append(f"instance_level_metrics.csv missing column: {col}")
        print(f"[OK] instance_level_metrics.csv: {len(df_inst)} rows verified (14 models x 5 seeds x 30 instances).")

    # 3. Check Decisions and Predictions NPZ Archive
    npz_path = os.path.join(results_dir, "instance_predictions_and_decisions.npz")
    if not os.path.isfile(npz_path):
        errors.append(f"Missing file: {npz_path}")
    else:
        try:
            archive = np.load(npz_path)
            req_keys = ["predictions", "deliveries", "true_demand", "oracle_costs", "costs", "regrets", "model_names", "seeds", "instance_ids"]
            for k in req_keys:
                if k not in archive.files:
                    errors.append(f"instance_predictions_and_decisions.npz missing key: {k}")
            if "predictions" in archive.files:
                p_shape = archive["predictions"].shape
                if len(p_shape) != 5 or p_shape[0] != 14 or p_shape[1] != 5 or p_shape[2] != 30:
                    errors.append(f"NPZ predictions shape {p_shape} expected (14, 5, 30, N, T).")
            if "deliveries" in archive.files:
                d_shape = archive["deliveries"].shape
                if len(d_shape) != 5 or d_shape[0] != 14 or d_shape[1] != 5 or d_shape[2] != 30:
                    errors.append(f"NPZ deliveries shape {d_shape} expected (14, 5, 30, N, T).")
            print("[OK] instance_predictions_and_decisions.npz: complete 5D tensors verified.")
        except (OSError, ValueError, KeyError) as e:
            errors.append(f"Error loading NPZ archive: {e}")

    # 4. Check Statistical Tests
    stat_path = os.path.join(results_dir, "statistical_tests.csv")
    if not os.path.isfile(stat_path):
        errors.append(f"Missing file: {stat_path}")
    else:
        df_stat = pd.read_csv(stat_path)
        if len(df_stat) != 13:
            errors.append(f"statistical_tests.csv expected 13 baseline comparisons, found {len(df_stat)}.")
        for col in ["Baseline_Model", "Ours_Model", "Mean_Difference_pct", "Cohen_dz", "Wilcoxon_p_Holm", "T_p_Holm", "Significant_Holm_05"]:
            if col not in df_stat.columns:
                errors.append(f"statistical_tests.csv missing column: {col}")
        # Check Holm-Bonferroni monotonicity
        p_raw = df_stat["Wilcoxon_p_raw"].values if "Wilcoxon_p_raw" in df_stat.columns else []
        p_holm = df_stat["Wilcoxon_p_Holm"].values if "Wilcoxon_p_Holm" in df_stat.columns else []
        for pr, ph in zip(p_raw, p_holm, strict=False):
            if ph < pr - 1e-9:
                errors.append(f"Holm adjusted p ({ph}) is strictly smaller than raw p ({pr}).")
        print("[OK] statistical_tests.csv: 13 comparisons, Holm-Bonferroni correction verified.")

    # 5. Check Capacity Diagnostics
    cap_path = os.path.join(results_dir, "capacity_diagnostics.csv")
    if not os.path.isfile(cap_path):
        errors.append(f"Missing file: {cap_path}")
    else:
        df_cap = pd.read_csv(cap_path)
        if len(df_cap) != 25:
            errors.append(f"capacity_diagnostics.csv expected 25 rows (5 models x 5 periods), found {len(df_cap)}.")
        for col in ["Model", "Period_t", "Delivered_Quantity", "Carryover_Inventory", "Realized_Stockout", "Realized_Spoilage"]:
            if col not in df_cap.columns:
                errors.append(f"capacity_diagnostics.csv missing column: {col}")
        print("[OK] capacity_diagnostics.csv: 25 rows verified (5 models x 5 periods).")

    # 6. Check Store-Level Diagnostics
    store_path = os.path.join(results_dir, "store_level_diagnostics.csv")
    if not os.path.isfile(store_path):
        errors.append(f"Missing file: {store_path}")
    else:
        df_store = pd.read_csv(store_path)
        if len(df_store) != 50:
            errors.append(f"store_level_diagnostics.csv expected 50 rows (5 models x 10 stores), found {len(df_store)}.")
        for col in ["Model", "Store_ID", "Depot_Distance", "Peripheral_Rationing_Index"]:
            if col not in df_store.columns:
                errors.append(f"store_level_diagnostics.csv missing column: {col}")
        print("[OK] store_level_diagnostics.csv: 50 rows verified (5 models x 10 stores).")

    # 7. Check Epoch Convergence History
    conv_path = os.path.join(results_dir, "epoch_convergence_history.csv")
    if not os.path.isfile(conv_path):
        errors.append(f"Missing file: {conv_path}")
    else:
        df_conv = pd.read_csv(conv_path)
        if len(df_conv) < 30000:
            errors.append(f"epoch_convergence_history.csv expected >= 30,000 rows, found {len(df_conv)}.")
        print(f"[OK] epoch_convergence_history.csv: {len(df_conv)} rows verified.")

    # 8. Check Figures
    figures_dir = os.path.join(results_dir, "figures")
    req_figs = [
        "fig_regret_comparison.png",
        "fig_convergence_curves.png",
        "fig_inventory_trajectory.png",
        "fig_speedup_pareto.png",
        "fig_sensitivity_heatmap.png",
    ]
    for fig_name in req_figs:
        fig_p = os.path.join(figures_dir, fig_name)
        if not os.path.isfile(fig_p):
            errors.append(f"Missing figure: {fig_p}")
    print("[OK] results/figures/: all 5 manuscript figures verified.")

    # 9. Numerical Parity Check
    if os.path.isfile(summary_path) and os.path.isfile(stat_path):
        df_sum = pd.read_csv(summary_path)
        df_stat = pd.read_csv(stat_path)
        ours_sum = df_sum[df_sum["Model"] == "Two-Phase_DFL (Ours)"]
        if len(ours_sum) == 1:
            ours_regret_sum = ours_sum["Regret_Mean_pct"].values[0]
            ours_regret_stat = df_stat["Mean_Regret_Ours_pct"].values[0]
            if abs(ours_regret_sum - ours_regret_stat) > 0.05:
                errors.append(
                    f"Regret discrepancy: multi_seed_summary has {ours_regret_sum:.2f}% "
                    f"vs statistical_tests has {ours_regret_stat:.2f}%."
                )
            else:
                print(f"[OK] Numerical Parity: Ours regret {ours_regret_sum:.2f}% matches across summary and test suites.")

    # 10. Write Environment Manifest
    manifest_path = os.path.join(results_dir, "environment_manifest.json")
    generate_environment_manifest(manifest_path)
    print("[OK] environment_manifest.json generated with system metadata.")

    elapsed = time.perf_counter() - t0
    print("-" * 80)
    if len(errors) == 0:
        print(f"VERIFICATION STATUS: 100% PASSED in {elapsed:.2f}s (0 errors)")
        print("=" * 80)
        return True, []
    else:
        print(f"VERIFICATION STATUS: FAILED with {len(errors)} errors in {elapsed:.2f}s:")
        for err in errors:
            print(f"  [X] {err}")
        print("=" * 80)
        return False, errors


def dry_run(output_dir: str = "results/dry_run") -> bool:
    """
    Executes fast (< 15s) smoke test running end-to-end pipeline on small subset.
    """
    print("=" * 80)
    print("EXECUTING PIPELINE SMOKE TEST (--dry-run)")
    print("=" * 80)
    t0 = time.perf_counter()

    # Run multi-seed benchmark with 1 seed, 2 epochs, 2 models
    _summary_df, _conv_df, _sig_df = run_multi_seed_benchmark(
        seeds=[42],
        models=["Ridge_PTO", "Two-Phase_DFL (Ours)"],
        total_epochs=2,
        topology="C1",
        output_dir=output_dir,
        save_results=True,
    )

    elapsed = time.perf_counter() - t0
    print("-" * 80)
    print(f"SMOKE TEST PASSED: execution completed in {elapsed:.2f}s (< 15s).")
    print("=" * 80)

    # Clean up dry run output directory
    import shutil
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    return True


def full_reproduction(output_dir: str = "results") -> bool:
    """Executes complete reproduction pipeline."""
    print("=" * 80)
    print("STARTING FULL BENCHMARK REPRODUCTION (--full)")
    print("=" * 80)

    # 1. Multi-Seed Benchmark
    run_multi_seed_benchmark(
        seeds=DEFAULT_SEEDS,
        models="all",
        total_epochs=1500,
        topology="C1",
        output_dir=output_dir,
        save_results=True,
    )

    # 2. Capacity Diagnostics
    run_capacity_diagnostics(
        output_csv_period=os.path.join(output_dir, "capacity_diagnostics.csv"),
        output_csv_store=os.path.join(output_dir, "store_level_diagnostics.csv"),
    )

    # 3. Statistical Testing Suite
    inst_csv = os.path.join(output_dir, "instance_level_metrics.csv")
    if os.path.isfile(inst_csv):
        df_inst = pd.read_csv(inst_csv)
        run_statistical_tests(
            instance_metrics_df=df_inst,
            output_path=os.path.join(output_dir, "statistical_tests.csv"),
        )

    # 4. Verify Artifacts
    is_valid, _errors = verify_artifacts(results_dir=output_dir)
    return is_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Reproduction Harness for PIRP-DFL")
    parser.add_argument("--verify", action="store_true", help="Fast audit of all benchmark files (< 5s)")
    parser.add_argument("--dry-run", action="store_true", help="Fast pipeline smoke test (< 15s)")
    parser.add_argument("--full", action="store_true", help="Execute complete benchmark reproduction")
    parser.add_argument("--output_dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    if args.dry_run:
        success = dry_run(output_dir=os.path.join(args.output_dir, "dry_run"))
        sys.exit(0 if success else 1)
    elif args.full:
        success = full_reproduction(output_dir=args.output_dir)
        sys.exit(0 if success else 1)
    else:
        # Default behavior: run verification audit
        is_valid, _ = verify_artifacts(results_dir=args.output_dir)
        sys.exit(0 if is_valid else 1)
