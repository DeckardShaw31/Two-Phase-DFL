"""
data/generate_dataset.py
Reproducible Data Generation and Preprocessing Pipeline for Multi-Period Perishable Inventory-Routing.

Creates a synthetic evaluation environment with hand-specified demand heterogeneity,
weekly seasonality, promotions, hidden weather shocks, and autoregressive noise. No
M5 observations or empirical demand calibration are used. The network uses ten-store
Solomon-derived spatial layouts; time windows are not modeled. Inventory and cost
parameters are explicit inputs to the generated benchmark.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. SOLOMON-DERIVED SPATIAL LAYOUTS (Depot + 10 Stores; no time windows)
# -----------------------------------------------------------------------------
# Coordinates selected from Solomon benchmark instances:
# Node 0: Central Distribution Depot
# Nodes 1..10: Regional Retail Stores (N=10)

SOLOMON_TOPOLOGIES = {
    "C1": {
        "description": "Clustered customer layout selected from Solomon C101",
        "depot": [40.0, 50.0],
        "customers": [
            [45.0, 68.0],  # Store 1
            [45.0, 70.0],  # Store 2
            [42.0, 66.0],  # Store 3
            [42.0, 68.0],  # Store 4
            [42.0, 65.0],  # Store 5
            [40.0, 69.0],  # Store 6
            [40.0, 66.0],  # Store 7
            [38.0, 68.0],  # Store 8
            [38.0, 70.0],  # Store 9
            [35.0, 66.0],  # Store 10
        ],
    },
    "R1": {
        "description": "Random customer layout selected from Solomon R101",
        "depot": [35.0, 35.0],
        "customers": [
            [41.0, 49.0],  # Store 1
            [35.0, 17.0],  # Store 2
            [55.0, 45.0],  # Store 3
            [55.0, 20.0],  # Store 4
            [15.0, 30.0],  # Store 5
            [25.0, 30.0],  # Store 6
            [20.0, 50.0],  # Store 7
            [10.0, 43.0],  # Store 8
            [55.0, 60.0],  # Store 9
            [30.0, 60.0],  # Store 10
        ],
    },
    "RC1": {
        "description": "Random-clustered customer layout selected from Solomon RC101",
        "depot": [40.0, 50.0],
        "customers": [
            [25.0, 85.0],  # Store 1 (Cluster A)
            [22.0, 75.0],  # Store 2 (Cluster A)
            [22.0, 85.0],  # Store 3 (Cluster A)
            [20.0, 80.0],  # Store 4 (Cluster A)
            [20.0, 85.0],  # Store 5 (Cluster A)
            [18.0, 75.0],  # Store 6 (Cluster A)
            [15.0, 75.0],  # Store 7 (Cluster A)
            [15.0, 80.0],  # Store 8 (Cluster A)
            [30.0, 50.0],  # Store 9 (Random periphery)
            [35.0, 35.0],  # Store 10 (Random periphery)
        ],
    },
}


def compute_euclidean_distance_matrix(coords: list[list[float]]) -> list[list[float]]:
    """
    Computes Euclidean distance matrix for a list of 2D coordinates.
    Ensures symmetry, zero diagonal, and triangle inequality.
    """
    num_nodes = len(coords)
    dist_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                dist_matrix[i][j] = 0.0
            else:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist = math.hypot(dx, dy)
                dist_matrix[i][j] = round(dist, 4)

    # Enforce exact symmetry
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            val = round((dist_matrix[i][j] + dist_matrix[j][i]) / 2.0, 4)
            dist_matrix[i][j] = val
            dist_matrix[j][i] = val

    return dist_matrix


def verify_distance_matrix(dist_matrix: list[list[float]]) -> bool:
    """
    Verifies that the distance matrix satisfies:
    1. Zero diagonal: dist[i][i] == 0
    2. Strictly positive off-diagonal: dist[i][j] > 0 for i != j
    3. Symmetry: dist[i][j] == dist[j][i]
    4. Triangle inequality: dist[i][j] <= dist[i][k] + dist[k][j] + 1e-4
    """
    n = len(dist_matrix)
    for i in range(n):
        if abs(dist_matrix[i][i]) > 1e-6:
            raise ValueError(f"Diagonal element dist[{i}][{i}] = {dist_matrix[i][i]} != 0")
        for j in range(n):
            if i != j and dist_matrix[i][j] <= 0:
                raise ValueError(f"Non-positive distance dist[{i}][{j}] = {dist_matrix[i][j]}")
            if abs(dist_matrix[i][j] - dist_matrix[j][i]) > 1e-6:
                raise ValueError(f"Asymmetry detected at ({i}, {j}): {dist_matrix[i][j]} != {dist_matrix[j][i]}")
            for k in range(n):
                if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j] + 1e-3:
                    raise ValueError(
                        f"Triangle inequality violated: dist[{i}][{j}]={dist_matrix[i][j]} > "
                        f"dist[{i}][{k}]={dist_matrix[i][k]} + dist[{k}][{j}]={dist_matrix[k][j]}"
                    )
    return True


def build_network_topology_dict(topology_name: str) -> dict:
    """
    Builds a spatial-network dictionary from selected Solomon customer coordinates.
    """
    topo_info = SOLOMON_TOPOLOGIES[topology_name]
    depot_coord = topo_info["depot"]
    customer_coords = topo_info["customers"]
    all_coords = [depot_coord] + customer_coords
    num_stores = len(customer_coords)

    dist_matrix = compute_euclidean_distance_matrix(all_coords)
    verify_distance_matrix(dist_matrix)

    node_names = ["Depot"] + [f"Store_{i}" for i in range(1, num_stores + 1)]

    return {
        "topology": topology_name,
        "description": topo_info["description"],
        "num_nodes": len(all_coords),
        "num_stores": num_stores,
        "depot_index": 0,
        "depot_coord": depot_coord,
        "store_indices": list(range(1, num_stores + 1)),
        "node_names": node_names,
        "node_coords": all_coords,
        "distance_matrix": dist_matrix,
        "service_time": [0.0] + [10.0] * num_stores,
    }


# -----------------------------------------------------------------------------
# 2. HAND-SPECIFIED SYNTHETIC RETAIL PERISHABLE DEMAND SIMULATION
# -----------------------------------------------------------------------------

# Illustrative synthetic store volume profiles (not empirically calibrated)
STORE_BASELINES = {
    1: {"base_mean": 22.0, "noise_scale": 2.2},
    2: {"base_mean": 17.5, "noise_scale": 1.9},
    3: {"base_mean": 24.0, "noise_scale": 2.5},
    4: {"base_mean": 15.0, "noise_scale": 1.7},
    5: {"base_mean": 19.5, "noise_scale": 2.1},
    6: {"base_mean": 16.0, "noise_scale": 1.8},
    7: {"base_mean": 21.0, "noise_scale": 2.2},
    8: {"base_mean": 18.0, "noise_scale": 2.0},
    9: {"base_mean": 20.5, "noise_scale": 2.3},
    10: {"base_mean": 15.5, "noise_scale": 1.7},
}

# Day-of-week multipliers: Monday=0 to Sunday=6
# Perishable food demand surges on Friday/Saturday/Sunday
DOW_MULTIPLIERS = [0.90, 0.85, 0.88, 0.92, 1.08, 1.28, 1.15]


def simulate_retail_demand_series(
    num_stores: int = 10,
    total_days: int = 1028,
    warmup_days: int = 28,
    seed: int = 42,
    calendar_start_day: int = 0,
) -> pd.DataFrame:
    """
    Simulates multi-store continuous daily perishable retail demand series.

    Generative components:
    - Base store volume heterogeneity
    - Day-of-week seasonality (weekend grocery surges)
    - Macro-trend and annual temperature cycle
    - Promotions with price discount and demand elasticity uplift
    - Weather precipitation foot traffic shocks
    - Autoregressive persistence AR(1)
    - Zero future lookahead for rolling/lag features
    """
    rng = np.random.RandomState(seed)
    records = []

    # Calendar covariates vary across generated episodes. Weather affects demand
    # generation but is not exposed to the forecaster as an exact future value.
    day_indices = np.arange(total_days)
    calendar_days = calendar_start_day + day_indices
    # Seasonal temperature: mean 18C, amplitude 9C, annual cycle 365 days
    temp_seasonal = 18.0 + 9.0 * np.sin(2 * np.pi * calendar_days / 365.0)
    temp_noise = rng.normal(loc=0.0, scale=2.5, size=total_days)
    daily_temperatures = np.round(temp_seasonal + temp_noise, 2)
    # Daily precipitation probability
    daily_rain_prob = np.round(rng.beta(a=1.5, b=4.0, size=total_days), 3)

    for store_id in range(1, num_stores + 1):
        store_profile = STORE_BASELINES[store_id]
        base_mean = store_profile["base_mean"]
        noise_scale = store_profile["noise_scale"]

        # Autoregressive error sequence
        ar_error = np.zeros(total_days)
        phi = 0.32
        white_noise = rng.normal(loc=0.0, scale=noise_scale, size=total_days)
        for d in range(1, total_days):
            ar_error[d] = phi * ar_error[d - 1] + white_noise[d]

        # Promotional schedule: 18% base probability, higher on Fridays/Saturdays
        dow_array = calendar_days % 7
        weekend_boost = np.where((dow_array == 4) | (dow_array == 5), 0.12, 0.0)
        promo_probs = 0.15 + weekend_boost
        promo_flags = (rng.uniform(size=total_days) < promo_probs).astype(int)

        # Discount depths between 10% and 30% during promotions
        discounts = np.where(
            promo_flags == 1,
            np.round(rng.uniform(low=0.10, high=0.30, size=total_days), 2),
            0.0,
        )

        base_price = 10.0
        net_prices = np.round(base_price * (1.0 - discounts), 2)
        # Price elasticity ~ -1.25 for perishable groceries
        price_uplift = np.where(
            discounts > 0,
            np.round((net_prices / base_price) ** (-1.25), 3),
            1.0,
        )

        # Weather factor: warmer temperature slightly boosts fresh goods (+1.2% per degree above 18C)
        temp_effect = 1.0 + 0.012 * (daily_temperatures - 18.0)
        # Rain shock: heavy rain (>0.55) slightly reduces store visits (-6%)
        rain_shock = np.where(daily_rain_prob > 0.55, 0.94, 1.0)

        # Compute raw daily demand series
        daily_demands = np.zeros(total_days)
        for d in range(total_days):
            dow = dow_array[d]
            dow_mult = DOW_MULTIPLIERS[dow]
            promo_mult = price_uplift[d]
            weather_mult = temp_effect[d] * rain_shock[d]
            # Long-term slight organic volume growth (1.5% per year)
            growth_trend = 1.0 + 0.015 * (calendar_days[d] / 365.0)

            expected_demand = base_mean * dow_mult * promo_mult * weather_mult * growth_trend
            realized = expected_demand + ar_error[d]
            # Truncate strictly at minimum 1.0 unit (non-negative, strictly positive demand)
            daily_demands[d] = max(1.0, round(float(realized), 2))

        # Features for a five-day plan are defined at episode start. Calendar and
        # announced promotion/price fields are known over the planning horizon;
        # demand history is a frozen pre-episode summary. Realized weather and
        # within-horizon demand lags are deliberately omitted from model inputs.
        history = daily_demands[max(0, warmup_days - 7) : warmup_days]
        history_last = float(daily_demands[warmup_days - 1])
        history_mean = float(np.mean(history))
        history_std = float(np.std(history, ddof=1)) if len(history) > 1 else 0.0
        for d in range(warmup_days, total_days):
            dow = int(dow_array[d])
            dow_sin = round(math.sin(2 * math.pi * dow / 7.0), 4)
            dow_cos = round(math.cos(2 * math.pi * dow / 7.0), 4)
            annual_angle = 2.0 * math.pi * (calendar_days[d] % 365) / 365.0

            records.append({
                "global_day": int(calendar_days[d]),
                "store_id": store_id,
                "day_of_week": dow,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "annual_sin": round(math.sin(annual_angle), 4),
                "annual_cos": round(math.cos(annual_angle), 4),
                "history_last_demand": history_last,
                "history_mean_7": round(history_mean, 3),
                "history_std_7": round(history_std, 3),
                "temperature": float(daily_temperatures[d]),
                "rain_prob": float(daily_rain_prob[d]),
                "promo_flag": int(promo_flags[d]),
                "discount_pct": float(discounts[d]),
                "base_price": float(base_price),
                "net_price": float(net_prices[d]),
                "demand": float(daily_demands[d]),
            })

    df = pd.DataFrame(records)
    # Sort strictly by global_day then store_id
    df = df.sort_values(by=["global_day", "store_id"]).reset_index(drop=True)
    return df


def generate_independent_episodes(
    num_instances: int = 200,
    num_stores: int = 10,
    horizon: int = 5,
    warmup_days: int = 28,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Generate independent demand paths with a private historical warmup per episode.

    Each five-day decision episode has its own RNG stream. The warmup is used only
    to construct the episode-start history features and is never included as an
    outcome row. This makes episode-level bootstrap inference defensible and keeps
    future realized demand out of all predictor inputs.
    """
    seed_rng = np.random.default_rng(seed)
    episode_seeds = seed_rng.integers(0, 2**32 - 1, size=num_instances, dtype=np.uint32)
    calendar_starts = seed_rng.integers(0, 365, size=num_instances)
    episodes: list[pd.DataFrame] = []
    episode_manifest: list[dict[str, int]] = []

    for instance_id, (episode_seed, calendar_start) in enumerate(zip(episode_seeds, calendar_starts, strict=True)):
        raw = simulate_retail_demand_series(
            num_stores=num_stores,
            total_days=warmup_days + horizon,
            warmup_days=warmup_days,
            seed=int(episode_seed),
            calendar_start_day=int(calendar_start),
        )
        raw = raw.copy()
        raw["instance_id"] = instance_id
        raw["t"] = np.repeat(np.arange(1, horizon + 1, dtype=np.int64), num_stores)
        raw["episode_seed"] = int(episode_seed)
        raw["episode_calendar_start"] = int(calendar_start)
        episodes.append(raw)
        episode_manifest.append({
            "instance_id": instance_id,
            "episode_seed": int(episode_seed),
            "calendar_start_day": int(calendar_start),
        })

    panel = pd.concat(episodes, ignore_index=True)
    panel = panel.sort_values(["instance_id", "t", "store_id"]).reset_index(drop=True)
    return panel, episode_manifest


def organize_into_instances(
    df: pd.DataFrame,
    horizon: int = 5,
    num_instances: int = 200,
) -> pd.DataFrame:
    """
    Groups continuous daily records into sequential multi-period planning instances.
    Each instance has horizon T (default 5 periods, t=1..5) across all stores.
    """
    min_day = df["global_day"].min()
    required_days = num_instances * horizon
    max_day = min_day + required_days

    # Filter strictly to the required range of days
    df_active = df[(df["global_day"] >= min_day) & (df["global_day"] < max_day)].copy()

    # Assign instance_id and step t
    df_active["instance_id"] = (df_active["global_day"] - min_day) // horizon
    df_active["t"] = ((df_active["global_day"] - min_day) % horizon) + 1  # 1-indexed: 1..T

    # Order columns logically
    ordered_cols = [
        "instance_id",
        "t",
        "store_id",
        "global_day",
        "day_of_week",
        "dow_sin",
        "dow_cos",
        "annual_sin",
        "annual_cos",
        "history_last_demand",
        "history_mean_7",
        "history_std_7",
        "temperature",
        "rain_prob",
        "promo_flag",
        "discount_pct",
        "base_price",
        "net_price",
        "demand",
    ]
    ordered_cols = [col for col in ordered_cols if col in df_active.columns]
    df_active = df_active[ordered_cols].sort_values(by=["instance_id", "t", "store_id"]).reset_index(drop=True)
    return df_active


# -----------------------------------------------------------------------------
# 3. SPLIT GENERATION & PARAMETER CONFIGURATION
# -----------------------------------------------------------------------------

def split_and_save_datasets(
    df_instances: pd.DataFrame,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits independently generated episodes in deterministic ID order into
    train, validation, and test datasets. Episode IDs are not time blocks.
    """
    instances = sorted(df_instances["instance_id"].unique())
    num_instances = len(instances)

    n_train = round(num_instances * train_ratio)
    n_val = round(num_instances * val_ratio)

    train_inst_ids = set(instances[:n_train])
    val_inst_ids = set(instances[n_train : n_train + n_val])
    test_inst_ids = set(instances[n_train + n_val :])

    df_train = df_instances[df_instances["instance_id"].isin(train_inst_ids)].copy().reset_index(drop=True)
    df_val = df_instances[df_instances["instance_id"].isin(val_inst_ids)].copy().reset_index(drop=True)
    df_test = df_instances[df_instances["instance_id"].isin(test_inst_ids)].copy().reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "demand_features_train.csv")
    val_path = os.path.join(output_dir, "demand_features_val.csv")
    test_path = os.path.join(output_dir, "demand_features_test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[Dataset] Train split: {len(df_train)} rows ({len(train_inst_ids)} instances) -> {train_path}")
    print(f"[Dataset] Val split:   {len(df_val)} rows ({len(val_inst_ids)} instances) -> {val_path}")
    print(f"[Dataset] Test split:  {len(df_test)} rows ({len(test_inst_ids)} instances) -> {test_path}")

    return df_train, df_val, df_test


def save_network_topologies(output_dir: str) -> None:
    """
    Builds and writes verified Solomon network topologies to JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for topo_name in ["C1", "R1", "RC1"]:
        network_dict = build_network_topology_dict(topo_name)
        out_filename = f"network_solomon_{topo_name.lower()}.json"
        out_path = os.path.join(output_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(network_dict, f, indent=2)
        print(f"[Network] Saved {topo_name} topology ({network_dict['num_nodes']} nodes) -> {out_path}")


def save_parameters_configuration(
    output_dir: str,
    num_instances: int = 200,
    horizon: int = 5,
    num_stores: int = 10,
    seed: int = 42,
) -> None:
    """
    Generates and saves the system parameters configuration file.
    """
    config = {
        "description": "Perishable Inventory-Routing System and Operational Parameters",
        "demand_source": "Synthetic demand generated by data/generate_dataset.py; no M5 observations or M5 calibration are used.",
        "episode_design": "Independent five-day demand trajectories with a private 28-day historical warmup; the warmup is used only for fixed episode-start context features.",
        "information_timing": "Calendar and announced promotion/price inputs are known at episode start. Realized weather and within-horizon realized demand are excluded from predictor features.",
        "num_stores": num_stores,
        "horizon_T": horizon,
        "num_instances": num_instances,
        "train_instances": int(num_instances * 0.70),
        "val_instances": int(num_instances * 0.15),
        "test_instances": int(num_instances * 0.15),
        "split_ratios": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        },
        "shelf_life_tau_candidates": [2, 3, 5],
        "default_shelf_life_tau": 3,
        "holding_cost_h": 1.0,
        "depot_holding_cost_h0": 0.5,
        "spoilage_penalty_ps": 5.0,
        "stockout_penalty_pu_candidates": [10.0, 20.0, 50.0],
        "default_stockout_penalty_pu": 20.0,
        "vehicle_capacity_Q": 100,
        "fleet_size_K": 2,
        "store_capacity_Ci": 70,
        "depot_capacity_C0": 1000,
        "topologies": ["C1", "R1", "RC1"],
        "random_seed": seed,
        "feature_names": [
            "store_id",
            "day_of_week",
            "dow_sin",
            "dow_cos",
            "annual_sin",
            "annual_cos",
            "history_last_demand",
            "history_mean_7",
            "history_std_7",
            "promo_flag",
            "discount_pct",
            "base_price",
            "net_price",
        ],
        "target_name": "demand",
        "index_cols": ["instance_id", "t", "store_id"],
    }

    out_path = os.path.join(output_dir, "params_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] System parameters configuration -> {out_path}")


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE ENTRY POINT
# -----------------------------------------------------------------------------

def run_pipeline(
    output_dir: str = "data/processed",
    num_stores: int = 10,
    num_instances: int = 200,
    horizon: int = 5,
    seed: int = 42,
) -> None:
    """
    Executes the end-to-end reproducible data generation and verification pipeline.
    """
    print("=" * 70)
    print("EXECUTING REPRODUCIBLE DATA PIPELINE FOR PIRP (Milestone 2)")
    print(f"Stores (N): {num_stores} | Horizon (T): {horizon} | Instances: {num_instances} | Seed: {seed}")
    print("=" * 70)

    # 1. Generate independent demand episodes with a private historical warmup.
    warmup_days = 28
    print(f"[Step 1] Generating {num_instances} independent episodes; each has {warmup_days} warm-up days + {horizon} operational days...")
    df_instances, episode_manifest = generate_independent_episodes(
        num_instances=num_instances,
        num_stores=num_stores,
        horizon=horizon,
        warmup_days=warmup_days,
        seed=seed,
    )
    print(f"[Step 2] Structured instances shape: {df_instances.shape}")

    # 3. Create train/val/test splits
    print("[Step 3] Splitting into Train (70%), Val (15%), Test (15%)...")
    df_train, df_val, df_test = split_and_save_datasets(
        df_instances=df_instances,
        output_dir=output_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # Summary statistics
    print("\n--- Demand Summary Statistics across Splits ---")
    print(f"Train demand: mean={df_train['demand'].mean():.2f}, std={df_train['demand'].std():.2f}, min={df_train['demand'].min():.2f}, max={df_train['demand'].max():.2f}")
    print(f"Val demand:   mean={df_val['demand'].mean():.2f}, std={df_val['demand'].std():.2f}, min={df_val['demand'].min():.2f}, max={df_val['demand'].max():.2f}")
    print(f"Test demand:  mean={df_test['demand'].mean():.2f}, std={df_test['demand'].std():.2f}, min={df_test['demand'].min():.2f}, max={df_test['demand'].max():.2f}")

    # 4. Generate Solomon Network Topologies
    print("\n[Step 4] Generating and verifying Solomon-derived spatial topologies (C1, R1, RC1)...")
    save_network_topologies(output_dir)

    # 5. Save System Parameters Configuration
    print("\n[Step 5] Writing System Parameter Grid Configuration...")
    save_parameters_configuration(
        output_dir=output_dir,
        num_instances=num_instances,
        horizon=horizon,
        num_stores=num_stores,
        seed=seed,
    )

    manifest_path = os.path.join(output_dir, "episode_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "master_seed": seed,
            "num_instances": num_instances,
            "horizon_days": horizon,
            "warmup_days": warmup_days,
            "independent_episode_rng": True,
            "m5_data_used": False,
            "episodes": episode_manifest,
            "train_instance_ids": sorted(df_train["instance_id"].unique().astype(int).tolist()),
            "validation_instance_ids": sorted(df_val["instance_id"].unique().astype(int).tolist()),
            "test_instance_ids": sorted(df_test["instance_id"].unique().astype(int).tolist()),
        }, f, indent=2)
    print(f"[Manifest] Independent episode seeds and split IDs -> {manifest_path}")

    print("\n" + "=" * 70)
    print("DATA PIPELINE GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reproducible PIRP dataset.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output processed data directory.")
    parser.add_argument("--num-stores", type=int, default=10, help="Number of retail stores (default: 10).")
    parser.add_argument("--num-instances", type=int, default=200, help="Number of planning instances (default: 200).")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon T in periods (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()
    run_pipeline(
        output_dir=args.output_dir,
        num_stores=args.num_stores,
        num_instances=args.num_instances,
        horizon=args.horizon,
        seed=args.seed,
    )
