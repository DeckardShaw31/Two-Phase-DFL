"""
Tests for statistical inference rigor, Holm-Bonferroni correction, and bootstrap CIs.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_statistical_tests import (
    compute_bootstrap_ci,
    holm_bonferroni_adjustment,
    run_statistical_tests,
)


def test_holm_bonferroni_monotonicity_and_step_down():
    """Verifies that step-down Holm-Bonferroni adjustment preserves ranking and bounds."""
    # Test cases with known properties
    raw_p = np.array([0.001, 0.04, 0.02, 0.005, 0.5])
    # Sorted: p_(1)=0.001 (m=5 -> *5 = 0.005)
    #         p_(2)=0.005 (m=4 -> *4 = 0.02)
    #         p_(3)=0.02  (m=3 -> *3 = 0.06)
    #         p_(4)=0.04  (m=2 -> *2 = 0.08)
    #         p_(5)=0.5   (m=1 -> *1 = 0.5)

    adj_p = holm_bonferroni_adjustment(raw_p)

    # 1. Bounds check
    assert (adj_p >= 0.0).all()
    assert (adj_p <= 1.0).all()

    # 2. Adjusted p-values must be >= raw p-values
    assert (adj_p >= raw_p).all()

    # 3. Monotonicity when sorted
    sorted_indices = np.argsort(raw_p)
    sorted_adj = adj_p[sorted_indices]
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-9

    # Check exact expected adjusted values
    expected_sorted = np.array([0.005, 0.02, 0.06, 0.08, 0.5])
    np.testing.assert_allclose(sorted_adj, expected_sorted, atol=1e-6)


def test_bootstrap_ci_coverage():
    """Verifies that compute_bootstrap_ci produces valid bounds containing the mean."""
    rng = np.random.default_rng(123)
    diffs = rng.normal(loc=5.0, scale=2.0, size=30)

    ci_lower, ci_upper = compute_bootstrap_ci(diffs, alpha=0.05, num_resamples=5000, seed=42)

    assert ci_lower < ci_upper
    sample_mean = float(np.mean(diffs))
    assert ci_lower <= sample_mean <= ci_upper


def test_statistical_tests_dataframe_rigor(tmp_path: Path):
    """Verifies statistical inference on synthetic multi-seed instance data."""
    # Create 3 models, 5 seeds, 30 instances
    records = []
    rng = np.random.default_rng(42)

    for inst in range(1, 31):
        for seed in [42, 1042, 2042, 3042, 4042]:
            # Ours: lower regret
            ours_reg = 20.0 + rng.normal(0, 2)
            # Baseline 1: significantly worse
            b1_reg = 35.0 + rng.normal(0, 3)
            # Baseline 2: slightly worse
            b2_reg = 26.0 + rng.normal(0, 2)

            records.append({"instance_id": inst, "seed": seed, "model": "Two-Phase_DFL (Ours)", "normalized_regret_pct": ours_reg})
            records.append({"instance_id": inst, "seed": seed, "model": "Baseline_1", "normalized_regret_pct": b1_reg})
            records.append({"instance_id": inst, "seed": seed, "model": "Baseline_2", "normalized_regret_pct": b2_reg})

    df_synth = pd.DataFrame(records)
    out_csv = str(tmp_path / "stat_test_synth.csv")

    res_df = run_statistical_tests(
        instance_metrics_df=df_synth,
        target_model="Two-Phase_DFL (Ours)",
        output_path=out_csv,
        save_results=True,
    )

    assert len(res_df) == 2
    assert "Baseline_1" in res_df["Baseline_Model"].values
    assert "Baseline_2" in res_df["Baseline_Model"].values

    # Check N_Instances is strictly 30 (seed-averaged independent units)
    assert (res_df["N_Instances"] == 30).all()

    # Check Cohen's d_z is positive
    assert (res_df["Cohen_dz"] > 0).all()

    # Check Holm adjusted p-values are valid
    assert (res_df["Wilcoxon_p_Holm"] >= 0.0).all()
    assert (res_df["Wilcoxon_p_Holm"] <= 1.0).all()


def test_production_statistical_tests_file():
    """If results/statistical_tests.csv exists, verifies its scientific rigor."""
    path = "results/statistical_tests.csv"
    if not os.path.isfile(path):
        pytest.skip(f"`{path}` not generated yet.")

    df = pd.read_csv(path)
    assert len(df) == 13
    assert (df["N_Instances"] == 30).all()
    assert (df["Mean_Difference_pct"] > 0).all()
    assert (df["Cohen_dz"] > 0).all()

    # Verify that Two-Phase DFL (Ours) achieves statistically significant outperformance
    # over standard PTO baselines under Holm-Bonferroni correction
    sig_models = df[df["Significant_Holm_05"]]["Baseline_Model"].tolist()
    assert "SafetyStock_PTO" in sig_models
    assert "Ridge_PTO" in sig_models
