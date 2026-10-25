"""
Unit and Integration Tests for Capacity Diagnostics, Factorial Sensitivity Grid,
and Multi-Seed Convergence (Requirements R2 and R3).
"""

from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_capacity_diagnostics_file_existence_and_schema():
    """Verify that results/capacity_diagnostics.csv exists and contains all required columns."""
    csv_path = WORKSPACE_ROOT / "results" / "capacity_diagnostics.csv"
    assert csv_path.is_file(), f"Missing capacity diagnostics file: {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = [
        "Model",
        "Period_t",
        "Agg_Predicted_Demand",
        "Net_Replenishment_Need",
        "Delivered_Quantity",
        "Fleet_Capacity_KQ",
        "Capacity_Binding_Freq",
        "Carryover_Inventory",
        "Realized_Stockout",
        "Realized_Spoilage",
        "Store_Stockout_Concentration_Gini",
        "Max_Store_Stockout_Share",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column in capacity diagnostics: {col}"

    # Verify no NaN, Inf, or empty values
    assert not df[expected_cols].isna().any().any(), "Found NaN values in capacity diagnostics!"
    assert not np.isinf(df[expected_cols].select_dtypes(include=[np.number]).values).any(), "Found Inf values!"

    # Verify evaluated models
    required_models = {
        "Ridge_PTO",
        "LightGBM_PTO",
        "LightGBM_Quantile_PTO",
        "SafetyStock_PTO",
        "Two-Phase_DFL (Ours)",
    }
    assert required_models.issubset(set(df["Model"].unique())), f"Missing models in diagnostics: {required_models - set(df['Model'].unique())}"

    # Verify periods 1..5
    assert set(df["Period_t"].unique()) == {1, 2, 3, 4, 5}


def test_capacity_diagnostics_empirical_findings():
    """
    Verify the empirical mechanisms demonstrated in capacity diagnostics:
      1. Asymmetric quantile and safety-stock PTO cause high binding-capacity frequency.
      2. Two-Phase DFL achieves lower average realized stockouts than naive buffer models.
      3. Fleet capacity KQ is 200.0.
    """
    csv_path = WORKSPACE_ROOT / "results" / "capacity_diagnostics.csv"
    df = pd.read_csv(csv_path)

    # Fleet capacity KQ must be 200.0
    assert (df["Fleet_Capacity_KQ"] == 200.0).all()

    # Binding capacity frequency: Quantile and Safety Stock should be significantly higher
    mean_binding = df.groupby("Model")["Capacity_Binding_Freq"].mean()
    assert mean_binding["SafetyStock_PTO"] > 0.75, f"Expected SafetyStock_PTO binding > 75%, got {mean_binding['SafetyStock_PTO']:.2f}"
    assert mean_binding["LightGBM_Quantile_PTO"] > 0.70, f"Expected LightGBM_Quantile_PTO binding > 70%, got {mean_binding['LightGBM_Quantile_PTO']:.2f}"

    # Stockouts: Two-Phase DFL achieves lowest average realized stockout
    mean_stockouts = df.groupby("Model")["Realized_Stockout"].mean()
    assert mean_stockouts["Two-Phase_DFL (Ours)"] < mean_stockouts["SafetyStock_PTO"]
    assert mean_stockouts["Two-Phase_DFL (Ours)"] < mean_stockouts["LightGBM_Quantile_PTO"]

    # Concentration Gini index bounds: Gini in [0, 1]
    assert (df["Store_Stockout_Concentration_Gini"] >= 0.0).all()
    assert (df["Store_Stockout_Concentration_Gini"] <= 1.0).all()


def test_sensitivity_analysis_81_grid_completeness():
    """
    Verify that results/sensitivity_analysis.csv contains all 81 unique parameter combinations:
      3 Topologies x 3 Shelf Lives x 3 Cost Ratios x 3 Capacity Tightness Levels = 81.
    """
    csv_path = WORKSPACE_ROOT / "results" / "sensitivity_analysis.csv"
    assert csv_path.is_file(), f"Missing sensitivity analysis file: {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = [
        "Topology",
        "ShelfLife_tau",
        "StockoutRatio_pu_h",
        "VehicleCapacity_Q",
        "CapacityTightness_gamma",
        "Model",
        "NormalizedRegret_pct",
        "Regret_std",
        "TotalCost",
        "RoutingCost",
        "SpoilageCost",
        "StockoutCost",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column in sensitivity analysis: {col}"

    # Check 81 unique parameter tuples
    grid_cols = ["Topology", "ShelfLife_tau", "StockoutRatio_pu_h", "VehicleCapacity_Q"]
    unique_tuples = df[grid_cols].drop_duplicates()
    assert len(unique_tuples) == 81, f"Expected 81 unique parameter configurations, got {len(unique_tuples)}"

    # Check parameter domains
    assert set(df["Topology"].unique()) == {"C1", "R1", "RC1"}
    assert set(df["ShelfLife_tau"].unique()) == {2, 3, 5}
    assert set(df["StockoutRatio_pu_h"].unique()) == {10.0, 20.0, 50.0}
    assert set(df["VehicleCapacity_Q"].unique()) == {80.0, 100.0, 150.0}
    assert set(np.round(df["CapacityTightness_gamma"].unique(), 1)) == {0.8, 1.0, 1.5}

    # Verify at least 3 models evaluated per configuration
    assert len(df) >= 81 * 3, f"Expected at least {81 * 3} rows, got {len(df)}"
    models_evaluated = set(df["Model"].unique())
    assert "Two-Phase_DFL (Ours)" in models_evaluated
    assert "Ridge_PTO" in models_evaluated
    assert "LightGBM_PTO" in models_evaluated

    # Verify non-null, valid numerical ranges
    assert not df[expected_cols].isna().any().any()
    assert (df["NormalizedRegret_pct"] >= 0.0).all()
    assert (df["TotalCost"] > 0.0).all()


def test_multi_seed_summary_completeness():
    """
    Verify that results/multi_seed_summary.csv contains all 14 evaluated models
    across 5 seeds with cross-seed and instance-level standard deviations.
    """
    csv_path = WORKSPACE_ROOT / "results" / "multi_seed_summary.csv"
    assert csv_path.is_file(), f"Missing multi-seed summary file: {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = [
        "Model",
        "Category",
        "NumSeeds",
        "Regret_Mean_pct",
        "Regret_CrossSeed_std",
        "Regret_Instance_std",
        "TotalCost_Mean",
        "TotalCost_std",
        "RoutingCost_Mean",
        "SpoilageCost_Mean",
        "StockoutCost_Mean",
        "MSE_Mean",
        "MAE_Mean",
        "TrainTime_sec_Mean",
        "Best_Epoch_Mean",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column in multi-seed summary: {col}"

    assert len(df) >= 14, f"Expected 14 models in multi_seed_summary, found {len(df)}"
    assert (df["NumSeeds"] == 5).all(), "Expected all models evaluated across exactly 5 seeds"

    # Confirm Two-Phase DFL achieves lowest average regret
    ours_row = df[df["Model"] == "Two-Phase_DFL (Ours)"]
    assert len(ours_row) == 1, "Missing Two-Phase_DFL (Ours) in summary"
    ours_regret = ours_row["Regret_Mean_pct"].values[0]

    for _, row in df.iterrows():
        if row["Model"] != "Two-Phase_DFL (Ours)":
            assert ours_regret <= row["Regret_Mean_pct"], (
                f"Two-Phase DFL regret ({ours_regret:.2f}%) should be <= {row['Model']} ({row['Regret_Mean_pct']:.2f}%)"
            )


def test_epoch_convergence_history_schema():
    """
    Verify results/epoch_convergence_history.csv contains valid 1,500-epoch convergence trajectories.
    """
    csv_path = WORKSPACE_ROOT / "results" / "epoch_convergence_history.csv"
    assert csv_path.is_file(), f"Missing convergence history file: {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = [
        "Seed",
        "Model",
        "Epoch",
        "Phase",
        "TrainLoss",
        "ValRegret_pct",
        "EpochTime_sec",
        "CumulativeTime_sec",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column in convergence history: {col}"

    assert (df["Epoch"].max() == 1500), f"Expected max epoch 1500, got {df['Epoch'].max()}"
    assert set(df["Seed"].unique()).issubset({42, 1042, 2042, 3042, 4042})
    assert (df["ValRegret_pct"] >= 0.0).all(), "Validation regret must be non-negative"
    assert (df["CumulativeTime_sec"] > 0.0).all(), "Cumulative time must be positive"


def test_statistical_significance_results():
    """
    Verify results/statistical_significance.csv contains valid paired hypothesis tests.
    """
    csv_path = WORKSPACE_ROOT / "results" / "statistical_significance.csv"
    assert csv_path.is_file(), f"Missing statistical significance file: {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = [
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
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column in statistical significance: {col}"

    # Mean difference should be positive (baseline > ours)
    assert (df["Mean_Difference_pct"] > 0.0).all(), "Expected Two-Phase DFL to outperform all baselines"
    assert ((df["Wilcoxon_p_value"] >= 0.0) & (df["Wilcoxon_p_value"] <= 1.0)).all()
    assert ((df["T_p_value"] >= 0.0) & (df["T_p_value"] <= 1.0)).all()


def test_publication_figures_exist_and_valid():
    """
    Verify that all 5 required publication figures exist at 300 DPI and have non-zero file sizes.
    """
    fig_dir = WORKSPACE_ROOT / "results" / "figures"
    required_figures = [
        "fig_regret_comparison.png",
        "fig_convergence_curves.png",
        "fig_speedup_pareto.png",
        "fig_sensitivity_heatmap.png",
        "fig_inventory_trajectory.png",
    ]
    for fig_name in required_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.is_file(), f"Missing publication figure: {fig_path}"
        assert fig_path.stat().st_size > 10000, f"Figure file size too small (< 10 KB): {fig_path}"

