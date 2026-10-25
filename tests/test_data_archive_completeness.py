"""
Tests for raw instance-level data archive completeness and schema integrity.
"""

import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def instance_metrics_df() -> pd.DataFrame:
    path = "results/instance_level_metrics.csv"
    if not os.path.isfile(path):
        pytest.skip(f"Data archive `{path}` not found.")
    return pd.read_csv(path)


@pytest.fixture
def decisions_npz_archive() -> dict[str, np.ndarray]:
    path = "results/instance_predictions_and_decisions.npz"
    if not os.path.isfile(path):
        pytest.skip(f"Data archive `{path}` not found.")
    return np.load(path)


def test_instance_metrics_row_count_and_schema(instance_metrics_df: pd.DataFrame):
    """
    Verifies that instance_level_metrics.csv contains exactly 2,100 records
    (14 models x 5 seeds x 30 instances) with zero missing values.
    """
    df = instance_metrics_df
    expected_rows = 14 * 5 * 30  # 2,100
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}."

    expected_cols = [
        "instance_id",
        "seed",
        "model",
        "oracle_cost",
        "routing_cost",
        "holding_cost",
        "spoilage_cost",
        "stockout_cost",
        "total_cost",
        "normalized_regret_pct",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column `{col}`."

    # Zero nulls
    assert df.isnull().sum().sum() == 0

    # Check non-negative costs and regrets
    assert (df["normalized_regret_pct"] >= -1e-5).all()
    assert (df["total_cost"] >= 0.0).all()
    assert (df["oracle_cost"] > 0.0).all()

    # Check that average realized total cost for all models exceeds the clairvoyant oracle
    model_costs = df.groupby("model")["total_cost"].mean()
    oracle_mean = df["oracle_cost"].mean()
    assert (model_costs > oracle_mean).all(), "Every model's mean cost must exceed the oracle cost."

    # Across all instances, whenever total_cost <= oracle_cost due to MIP solver gap, regret is strictly clamped to 0
    below_oracle = df["total_cost"] <= df["oracle_cost"]
    if below_oracle.any():
        assert (df.loc[below_oracle, "normalized_regret_pct"] == 0.0).all()


def test_decisions_npz_dimensions_and_keys(decisions_npz_archive: dict[str, np.ndarray]):
    """
    Verifies that instance_predictions_and_decisions.npz contains complete 5D/3D tensors.
    """
    archive = decisions_npz_archive
    required_keys = [
        "predictions",
        "deliveries",
        "true_demand",
        "oracle_costs",
        "costs",
        "regrets",
        "model_names",
        "seeds",
        "instance_ids",
    ]
    for k in required_keys:
        assert k in archive, f"Missing array `{k}` in NPZ archive."

    # Dimensions: 14 models, 5 seeds, 30 instances, 10 stores, 5 periods
    assert archive["predictions"].shape == (14, 5, 30, 10, 5)
    assert archive["deliveries"].shape == (14, 5, 30, 10, 5)
    assert archive["true_demand"].shape == (30, 10, 5)
    assert archive["oracle_costs"].shape == (30,)
    assert archive["costs"].shape == (14, 5, 30)
    assert archive["regrets"].shape == (14, 5, 30)
    assert len(archive["model_names"]) == 14
    assert len(archive["seeds"]) == 5
    assert len(archive["instance_ids"]) == 30

    # Non-negativity of physical quantities
    assert (archive["deliveries"] >= -1e-5).all()
    assert (archive["predictions"] >= -1e-5).all()
    assert (archive["true_demand"] >= 0.0).all()
