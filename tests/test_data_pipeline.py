"""
tests/test_data_pipeline.py
Automated Verification and Validation Test Suite for PIRP Data Pipeline (Milestone 2).

Tests:
1. File existence and structure of all processed datasets and configurations.
2. Zero missing, NaN, null, or infinite values in features and targets.
3. Feature integrity, value ranges, and monotonic temporal indexing.
4. Metric space properties of Solomon C1, R1, and RC1 distance matrices (zero diagonal,
   strict positivity, exact symmetry, triangle inequality).
5. Exact parameter configuration alignment with Q1 research specification.
6. Bitwise / numerical reproducibility under fixed random seeds and genuine variance under differing seeds.
7. Mathematical correctness of lag and rolling window feature calculations.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import (
    simulate_retail_demand_series,
)

DATA_PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)


class TestProcessedFilesExistence:
    """Verifies that all required dataset and configuration files exist."""

    @pytest.mark.parametrize(
        "filename",
        [
            "demand_features_train.csv",
            "demand_features_val.csv",
            "demand_features_test.csv",
            "network_solomon_c1.json",
            "network_solomon_r1.json",
            "network_solomon_rc1.json",
            "params_config.json",
        ],
    )
    def test_file_exists(self, filename):
        file_path = os.path.join(DATA_PROCESSED_DIR, filename)
        assert os.path.exists(file_path), f"Required file missing: {file_path}"
        assert os.path.getsize(file_path) > 0, f"File is empty: {file_path}"


class TestDatasetSplitsAndIntegrity:
    """Verifies row counts, non-overlapping instances, and schema consistency across splits."""

    @pytest.fixture(scope="class")
    @classmethod
    def datasets(cls):
        train_path = os.path.join(DATA_PROCESSED_DIR, "demand_features_train.csv")
        val_path = os.path.join(DATA_PROCESSED_DIR, "demand_features_val.csv")
        test_path = os.path.join(DATA_PROCESSED_DIR, "demand_features_test.csv")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        return {"train": df_train, "val": df_val, "test": df_test}

    def test_split_row_counts_and_shapes(self, datasets):
        df_train = datasets["train"]
        df_val = datasets["val"]
        df_test = datasets["test"]

        # 140 train instances * 10 stores * 5 time periods = 7000 rows
        assert len(df_train) == 7000, f"Expected 7000 train rows, got {len(df_train)}"
        # 30 val instances * 10 stores * 5 time periods = 1500 rows
        assert len(df_val) == 1500, f"Expected 1500 val rows, got {len(df_val)}"
        # 30 test instances * 10 stores * 5 time periods = 1500 rows
        assert len(df_test) == 1500, f"Expected 1500 test rows, got {len(df_test)}"

        # Total rows = 10,000
        total_rows = len(df_train) + len(df_val) + len(df_test)
        assert total_rows == 10000, f"Expected 10000 total rows, got {total_rows}"

    def test_instance_disjointness_and_temporal_order(self, datasets):
        train_inst = set(datasets["train"]["instance_id"].unique())
        val_inst = set(datasets["val"]["instance_id"].unique())
        test_inst = set(datasets["test"]["instance_id"].unique())

        assert len(train_inst) == 140, f"Expected 140 train instances, got {len(train_inst)}"
        assert len(val_inst) == 30, f"Expected 30 val instances, got {len(val_inst)}"
        assert len(test_inst) == 30, f"Expected 30 test instances, got {len(test_inst)}"

        # Zero leakage / overlap between sets
        assert train_inst.isdisjoint(val_inst), "Train and Val instances overlap!"
        assert train_inst.isdisjoint(test_inst), "Train and Test instances overlap!"
        assert val_inst.isdisjoint(test_inst), "Val and Test instances overlap!"

        # Strict sequential temporal ordering
        assert max(train_inst) < min(val_inst), "Val instances precede train instances!"
        assert max(val_inst) < min(test_inst), "Test instances precede val instances!"

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_zero_missing_nan_inf(self, datasets, split_name):
        df = datasets[split_name]

        # No null/NaN values in any column
        null_counts = df.isnull().sum()
        assert null_counts.sum() == 0, f"Null values found in {split_name}: {null_counts[null_counts > 0]}"

        # No infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, f"Infinite values found in {split_name} column {col}: {inf_count}"

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_schema_and_column_names(self, datasets, split_name):
        df = datasets[split_name]
        expected_cols = [
            "instance_id",
            "t",
            "store_id",
            "global_day",
            "day_of_week",
            "dow_sin",
            "dow_cos",
            "temperature",
            "rain_prob",
            "promo_flag",
            "discount_pct",
            "base_price",
            "net_price",
            "lag_1",
            "lag_2",
            "lag_7",
            "rolling_mean_7",
            "rolling_std_7",
            "demand",
        ]
        assert list(df.columns) == expected_cols, f"Column mismatch in {split_name}"

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_value_bounds_and_physical_validity(self, datasets, split_name):
        df = datasets[split_name]

        # Horizon steps t in 1..5
        assert set(df["t"].unique()) == {1, 2, 3, 4, 5}

        # Store IDs in 1..10
        assert set(df["store_id"].unique()) == set(range(1, 11))

        # Target demand is strictly positive
        assert (df["demand"] > 0).all(), f"Found non-positive demand in {split_name}"
        assert df["demand"].min() >= 1.0, f"Minimum demand should be >= 1.0, got {df['demand'].min()}"

        # Binary promo flag
        assert set(df["promo_flag"].unique()).issubset({0, 1})

        # Discount between 0.0 and 1.0
        assert (df["discount_pct"] >= 0.0).all() and (df["discount_pct"] <= 1.0).all()

        # Net price <= base price
        assert (df["net_price"] <= df["base_price"] + 1e-6).all()
        assert (df["net_price"] > 0).all()

        # Day of week in 0..6
        assert set(df["day_of_week"].unique()).issubset(set(range(7)))

        # Rolling std is non-negative
        assert (df["rolling_std_7"] >= 0).all()


class TestSolomonTopologies:
    """Verifies Solomon C1, R1, and RC1 network topology metric space properties."""

    @pytest.mark.parametrize("topo_name", ["c1", "r1", "rc1"])
    def test_topology_json_structure(self, topo_name):
        file_path = os.path.join(DATA_PROCESSED_DIR, f"network_solomon_{topo_name}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["topology"].lower() == topo_name
        assert data["num_nodes"] == 11  # 1 depot + 10 stores
        assert data["num_stores"] == 10
        assert data["depot_index"] == 0
        assert data["store_indices"] == list(range(1, 11))
        assert len(data["node_coords"]) == 11
        assert len(data["distance_matrix"]) == 11

    @pytest.mark.parametrize("topo_name", ["c1", "r1", "rc1"])
    def test_distance_matrix_metric_properties(self, topo_name):
        file_path = os.path.join(DATA_PROCESSED_DIR, f"network_solomon_{topo_name}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dist_matrix = data["distance_matrix"]
        coords = data["node_coords"]
        n = len(dist_matrix)

        # 1. Zero diagonal
        for i in range(n):
            assert dist_matrix[i][i] == 0.0, f"Diagonal dist[{i}][{i}] != 0 in {topo_name}"

        # 2. Strict positivity off-diagonal
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert dist_matrix[i][j] > 0.0, f"Distance dist[{i}][{j}] <= 0 in {topo_name}"

        # 3. Exact symmetry
        for i in range(n):
            for j in range(n):
                assert dist_matrix[i][j] == dist_matrix[j][i], f"Asymmetry at ({i},{j}) in {topo_name}"

        # 4. Triangle inequality: c_ik + c_kj >= c_ij - 1e-4
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert (
                        dist_matrix[i][k] + dist_matrix[k][j] >= dist_matrix[i][j] - 1e-4
                    ), f"Triangle inequality failed in {topo_name} for ({i}, {k}, {j})"

        # 5. Consistency with node coordinates
        for i in range(n):
            for j in range(n):
                expected_dist = round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]), 4)
                assert abs(dist_matrix[i][j] - expected_dist) <= 1e-3, f"Coordinate distance mismatch at ({i}, {j})"


class TestParamsConfiguration:
    """Verifies that params_config.json contains all required parameters with exact specifications."""

    @pytest.fixture(scope="class")
    @classmethod
    def config(cls):
        config_path = os.path.join(DATA_PROCESSED_DIR, "params_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_parameter_values(self, config):
        assert config["num_stores"] == 10
        assert config["horizon_T"] == 5
        assert config["num_instances"] == 200
        assert config["train_instances"] == 140
        assert config["val_instances"] == 30
        assert config["test_instances"] == 30

        # Physical and cost parameters from R1/R2
        assert config["shelf_life_tau_candidates"] == [2, 3, 5]
        assert config["holding_cost_h"] == 1.0
        assert config["depot_holding_cost_h0"] == 0.5
        assert config["spoilage_penalty_ps"] == 5.0
        assert config["stockout_penalty_pu_candidates"] == [10.0, 20.0, 50.0]

        # Fleet and capacity parameters
        assert config["vehicle_capacity_Q"] == 100
        assert config["fleet_size_K"] == 2
        assert config["store_capacity_Ci"] == 70
        assert config["depot_capacity_C0"] == 1000

        # Topologies
        assert set(config["topologies"]) == {"C1", "R1", "RC1"}
        assert config["random_seed"] == 42


class TestReproducibilityAndFeatureMath:
    """Verifies exact reproducibility under fixed seeds and validates lag/rolling calculations."""

    def test_reproducibility_across_runs(self):
        # Generate with seed 42 twice and confirm identical values
        df1 = simulate_retail_demand_series(num_stores=5, total_days=60, warmup_days=14, seed=42)
        df2 = simulate_retail_demand_series(num_stores=5, total_days=60, warmup_days=14, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

        # Generate with different seed and confirm different values
        df3 = simulate_retail_demand_series(num_stores=5, total_days=60, warmup_days=14, seed=999)
        assert not df1["demand"].equals(df3["demand"]), "Different seeds produced identical demand!"

    def test_feature_lag_and_rolling_math(self):
        # Generate a small series and manually verify lag and rolling calculations
        df = simulate_retail_demand_series(num_stores=2, total_days=50, warmup_days=20, seed=123)

        # Verify for Store 1
        df_store1 = df[df["store_id"] == 1].sort_values("global_day").reset_index(drop=True)

        for idx in range(7, len(df_store1)):
            current_row = df_store1.iloc[idx]
            prev_row = df_store1.iloc[idx - 1]
            prev2_row = df_store1.iloc[idx - 2]
            prev7_row = df_store1.iloc[idx - 7]

            # Lag 1 matches demand at idx - 1
            assert abs(current_row["lag_1"] - prev_row["demand"]) < 1e-4

            # Lag 2 matches demand at idx - 2
            assert abs(current_row["lag_2"] - prev2_row["demand"]) < 1e-4

            # Lag 7 matches demand at idx - 7
            assert abs(current_row["lag_7"] - prev7_row["demand"]) < 1e-4

            # Rolling 7-day mean matches mean of previous 7 days
            expected_mean = np.mean(df_store1.iloc[idx - 7 : idx]["demand"])
            assert abs(current_row["rolling_mean_7"] - expected_mean) < 1e-2
