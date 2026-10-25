"""
Unit and Integration Tests for Two-Stage Baselines, DFL Pipelines, and Metrics.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_benchmark import (
    compute_statistical_significance,
    load_dataset,
    run_benchmark,
    structure_instances,
)
from src.models.baselines import (
    MLPPTO,
    ElasticNetPTO,
    GradientBoostingPTO,
    KNNSAAModel,
    LightGBMPTO,
    LightGBMQuantilePTO,
    RandomForestPTO,
    RidgePTO,
    SafetyStockPTO,
    TemporalSeqNetPTO,
)
from src.models.dfl_models import DFLPipeline
from src.utils.metrics import (
    compute_decision_regret,
)


@pytest.fixture
def dummy_train_val_data():
    np.random.seed(42)
    N_samples = 100
    N_features = 14
    X = np.random.randn(N_samples, N_features).astype(np.float32)
    y = np.maximum(5.0, 20.0 + 3.0 * X[:, 0] - 2.0 * X[:, 1] + np.random.randn(N_samples) * 2.0)
    return X, y


def test_ridge_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = RidgePTO(alpha=1.0).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_lightgbm_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = LightGBMPTO(n_estimators=10).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_mlp_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = MLPPTO(in_features=14, epochs=3).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_knn_saa_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = KNNSAAModel(k_neighbors=3).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_elasticnet_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = ElasticNetPTO(alpha=0.1, l1_ratio=0.5).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_randomforest_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = RandomForestPTO(n_estimators=10, max_depth=4).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_gradientboosting_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = GradientBoostingPTO(n_estimators=10, max_depth=3).fit(X[:80], y[:80])
    assert model.is_fitted
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_lightgbm_quantile_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = LightGBMQuantilePTO(
        stockout_penalty_pu=20.0,
        spoilage_penalty_ps=5.0,
        holding_cost_h=1.0,
        shelf_life_tau=3,
        n_estimators=10,
    ).fit(X[:80], y[:80])
    assert model.is_fitted
    assert np.isclose(model.critical_fractile, 20.0 / (20.0 + 1.0 + 5.0 / 3.0), atol=1e-3)
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_safety_stock_pto_baseline(dummy_train_val_data):
    X, y = dummy_train_val_data
    model = SafetyStockPTO(
        alpha_ridge=1.0,
        stockout_penalty_pu=20.0,
        spoilage_penalty_ps=5.0,
        holding_cost_h=1.0,
    ).fit(X[:80], y[:80])
    assert model.is_fitted
    assert model.sigma > 0.0
    assert model.safety_buffer >= 0.0
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0.0)


def test_temporal_seq_net_pto_baseline():
    """Verifies that TemporalSeqNetPTO accepts 2D, 3D, and 4D tensor inputs."""
    np.random.seed(42)
    N, T, F = 10, 5, 14
    X_4d = np.random.randn(4, N, T, F).astype(np.float32)
    y_3d = np.random.uniform(10, 30, size=(4, N, T)).astype(np.float32)

    model = TemporalSeqNetPTO(
        in_features=F,
        num_stores=N,
        horizon_T=T,
        hidden_dim=32,
        epochs=3,
        batch_size=2,
    )
    model.fit(X_4d, y_3d)
    assert model.is_fitted

    # Predict on 4D batch
    preds_4d = model.predict(X_4d)
    assert preds_4d.shape == (4, N, T)
    assert np.all(preds_4d >= 0.0)

    # Predict on 3D single instance
    preds_3d = model.predict(X_4d[0])
    assert preds_3d.shape == (N, T)
    assert np.all(preds_3d >= 0.0)

    # Predict on 2D flat
    X_2d = X_4d.reshape(-1, F)
    preds_2d = model.predict(X_2d)
    assert preds_2d.shape == (X_2d.shape[0],)
    assert np.all(preds_2d >= 0.0)


def test_multidimensional_tensor_handling_base_pto():
    """Verifies that tabular models transparently reshape 3D and 4D tensors."""
    N, T, F = 10, 5, 14
    X_4d = np.random.randn(4, N, T, F).astype(np.float32)
    y_3d = np.random.uniform(10, 30, size=(4, N, T)).astype(np.float32)

    for cls in [RidgePTO, ElasticNetPTO, LightGBMPTO]:
        model = cls().fit(X_4d, y_3d)
        assert model.is_fitted
        preds_batch = model.predict(X_4d)
        assert preds_batch.shape == (4, N, T)
        assert np.all(preds_batch >= 0.0)

        preds_single = model.predict(X_4d[0])
        assert preds_single.shape == (N, T)
        assert np.all(preds_single >= 0.0)


def test_benchmark_dataset_loading_full_splits():
    """Verifies that all three dataset splits (140 train, 30 val, 30 test) load cleanly."""
    df_train, df_val, df_test, config, dist_matrix, feat_cols = load_dataset()

    assert len(df_train["instance_id"].unique()) == 140
    assert len(df_val["instance_id"].unique()) == 30
    assert len(df_test["instance_id"].unique()) == 30
    assert len(feat_cols) == 14
    assert dist_matrix.shape == (11, 11)

    N = config["num_stores"]
    T = config["horizon_T"]
    X_train_3d, d_train_3d = structure_instances(df_train, feat_cols, N=N, T=T)
    X_val_3d, d_val_3d = structure_instances(df_val, feat_cols, N=N, T=T)
    X_test_3d, d_test_3d = structure_instances(df_test, feat_cols, N=N, T=T)

    assert X_train_3d.shape == (140, 10, 5, 14)
    assert d_train_3d.shape == (140, 10, 5)
    assert X_val_3d.shape == (30, 10, 5, 14)
    assert d_val_3d.shape == (30, 10, 5)
    assert X_test_3d.shape == (30, 10, 5, 14)
    assert d_test_3d.shape == (30, 10, 5)


def test_statistical_significance_computation():
    """Verifies paired Wilcoxon and t-test calculation and edge case handling."""
    # Synthetic instance regrets
    inst_ids = list(range(1, 11))
    records = []
    for i in inst_ids:
        # Ours has lower regret
        records.append({
            "instance_id": i,
            "Model": "Two-Phase_DFL (Ours)",
            "NormalizedRegret_pct": 10.0 + float(i),
        })
        # Baseline 1 has higher regret
        records.append({
            "instance_id": i,
            "Model": "Ridge_PTO",
            "NormalizedRegret_pct": 20.0 + float(i) * 1.5,
        })
        # Baseline 2 has identical regret
        records.append({
            "instance_id": i,
            "Model": "Identical_Model",
            "NormalizedRegret_pct": 10.0 + float(i),
        })

    df = pd.DataFrame(records)
    sig_df = compute_statistical_significance(df, target_model="Two-Phase_DFL (Ours)")

    assert len(sig_df) == 2
    assert "Ridge_PTO" in sig_df["Baseline_Model"].values
    assert "Identical_Model" in sig_df["Baseline_Model"].values

    ridge_row = sig_df[sig_df["Baseline_Model"] == "Ridge_PTO"].iloc[0]
    assert ridge_row["Mean_Difference_pct"] > 0
    assert ridge_row["T_p_value"] < 0.05
    assert ridge_row["Significant_p05"]

    identical_row = sig_df[sig_df["Baseline_Model"] == "Identical_Model"].iloc[0]
    assert np.isclose(identical_row["Mean_Difference_pct"], 0.0)
    assert identical_row["Wilcoxon_p_value"] == 1.0


def test_metrics_regret_calculation():
    # Model cost 120, oracle cost 100 -> abs_regret=20, norm_regret=20%
    abs_r, norm_r = compute_decision_regret(model_cost=120.0, oracle_cost=100.0)
    assert np.isclose(abs_r, 20.0)
    assert np.isclose(norm_r, 20.0)

    # Perfect forecast: model matches oracle -> regret=0%
    abs_r0, norm_r0 = compute_decision_regret(model_cost=100.0, oracle_cost=100.0)
    assert np.isclose(abs_r0, 0.0)
    assert np.isclose(norm_r0, 0.0)


def test_dfl_pipeline_two_phase_warmup():
    """Verifies that Two-Phase DFL trains and transitions between LP warmup and MILP."""
    N, T, F = 10, 5, 14
    X_train = np.random.randn(4, N, T, F).astype(np.float32)
    d_train = np.random.uniform(10, 30, size=(4, N, T)).astype(np.float32)

    pipeline = DFLPipeline(loss_type="SPO+", in_features=F, num_stores=N, horizon_T=T)
    history = pipeline.fit_two_phase(
        X_train=X_train,
        d_train=d_train,
        total_epochs=2,
        transition_ratio=0.5,
        verbose=False,
    )
    assert len(history) == 2
    assert history[0]["phase"] == "Phase1_LP_Warmup"
    assert history[1]["phase"] == "Phase2_MILP_FineTune"


def test_dry_run_benchmark_integration(tmp_path):
    """Executes a rapid dry-run benchmark and verifies schema and generated files."""
    out_dir = str(tmp_path / "benchmark_test_out")
    df_results = run_benchmark(
        models=["Ridge_PTO", "Two-Phase_DFL"],
        num_eval_instances=2,
        num_train_instances=4,
        epochs_dfl=2,
        output_dir=out_dir,
        save_results=True,
        dry_run=True,
    )

    expected_cols = [
        "Model",
        "Category",
        "Topology",
        "ShelfLife_tau",
        "StockoutRatio_pu_h",
        "NormalizedRegret_pct",
        "Regret_std",
        "TotalCost",
        "RoutingCost",
        "SpoilageCost",
        "StockoutCost",
        "MSE",
        "MAE",
        "TrainTime_sec",
    ]
    for col in expected_cols:
        assert col in df_results.columns

    assert len(df_results) == 2
    assert os.path.exists(os.path.join(out_dir, "benchmark_metrics.csv"))
    assert os.path.exists(os.path.join(out_dir, "benchmark_metrics.json"))
    assert os.path.exists(os.path.join(out_dir, "instance_regrets.csv"))
    assert os.path.exists(os.path.join(out_dir, "statistical_significance.csv"))
    assert os.path.exists(os.path.join(out_dir, "statistical_significance.json"))
