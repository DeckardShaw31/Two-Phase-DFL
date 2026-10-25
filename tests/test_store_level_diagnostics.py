"""
Tests for period-level and store-level capacity externality diagnostics.
"""

import os

import pandas as pd
import pytest


@pytest.fixture
def store_diagnostics_df() -> pd.DataFrame:
    path = "results/store_level_diagnostics.csv"
    if not os.path.isfile(path):
        pytest.skip(f"Diagnostic file `{path}` not found.")
    return pd.read_csv(path)


@pytest.fixture
def capacity_diagnostics_df() -> pd.DataFrame:
    path = "results/capacity_diagnostics.csv"
    if not os.path.isfile(path):
        pytest.skip(f"Diagnostic file `{path}` not found.")
    return pd.read_csv(path)


def test_capacity_diagnostics_structure(capacity_diagnostics_df: pd.DataFrame):
    """Verifies that capacity_diagnostics.csv contains 25 rows (5 models x 5 periods)."""
    df = capacity_diagnostics_df
    assert len(df) == 25
    expected_models = {
        "Two-Phase_DFL (Ours)",
        "LightGBM_PTO",
        "Ridge_PTO",
        "LightGBM_Quantile_PTO",
        "SafetyStock_PTO",
    }
    assert set(df["Model"].unique()) == expected_models
    assert set(df["Period_t"].unique()) == {1, 2, 3, 4, 5}

    required_cols = [
        "Model",
        "Period_t",
        "Delivered_Quantity",
        "Carryover_Inventory",
        "Realized_Stockout",
        "Realized_Spoilage",
    ]
    for col in required_cols:
        assert col in df.columns


def test_store_level_diagnostics_structure(store_diagnostics_df: pd.DataFrame):
    """Verifies that store_level_diagnostics.csv contains 50 rows (5 models x 10 stores)."""
    df = store_diagnostics_df
    assert len(df) == 50
    assert set(df["Store_ID"].unique()) == set(range(1, 11))

    required_cols = [
        "Model",
        "Store_ID",
        "Depot_Distance",
        "Delivery_Frequency",
        "Avg_Delivered_Qty",
        "Avg_Carryover",
        "Avg_Stockout",
        "Avg_Demand",
        "Stockout_Share",
        "Demand_Share",
        "Peripheral_Rationing_Index",
    ]
    for col in required_cols:
        assert col in df.columns


def test_peripheral_rationing_externality(store_diagnostics_df: pd.DataFrame):
    """
    Verifies the economic mechanism:
    Under SafetyStock_PTO and LightGBM_Quantile_PTO, peripheral stores (stores 4, 6, 10)
    suffer a disproportionate stockout share due to vehicle capacity rationing, resulting in
    Peripheral_Rationing_Index > 1.0 (indicating peripheral rationing).
    Under Two-Phase_DFL (Ours), the peripheral rationing is mitigated.
    """
    df = store_diagnostics_df

    # Peripheral rationed stores: 4, 6, 10
    peripheral_stores = [4, 6, 10]

    # Check that SafetyStock_PTO exhibits severe peripheral rationing (> 2.0)
    ss_df = df[(df["Model"] == "SafetyStock_PTO") & (df["Store_ID"].isin(peripheral_stores))]
    assert (ss_df["Peripheral_Rationing_Index"] > 2.0).all()

    # Check that LightGBM_Quantile_PTO also exhibits severe peripheral rationing (> 2.0)
    lgb_q_df = df[(df["Model"] == "LightGBM_Quantile_PTO") & (df["Store_ID"].isin(peripheral_stores))]
    assert (lgb_q_df["Peripheral_Rationing_Index"] > 2.0).all()

    # Compare average stockouts at peripheral stores: Ours is much lower than SafetyStock
    ours_periph_stockout = df[(df["Model"] == "Two-Phase_DFL (Ours)") & (df["Store_ID"].isin(peripheral_stores))]["Avg_Stockout"].mean()
    ss_periph_stockout = ss_df["Avg_Stockout"].mean()
    assert ours_periph_stockout < ss_periph_stockout
