"""
Unit and integration tests for Master Reproduction Harness (src/experiments/reproduce_all.py).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from src.experiments.reproduce_all import (
    generate_environment_manifest,
)


def test_environment_manifest_structure(tmp_path: Path):
    """Verifies that environment manifest contains all required metadata and parameters."""
    manifest_file = str(tmp_path / "environment_manifest.json")
    manifest = generate_environment_manifest(output_path=manifest_file)

    assert os.path.isfile(manifest_file)
    assert "platform" in manifest
    with open(manifest_file, "r") as f:
        loaded = json.load(f)

    assert "platform" in loaded
    assert "packages" in loaded
    assert "hardware" in loaded
    assert "pirp_parameters" in loaded
    assert "experiment_configuration" in loaded

    # Check PIRP parameter alignment
    pirp_params = loaded["pirp_parameters"]
    assert pirp_params["store_capacity_Ci"] == 70.0
    assert pirp_params["aging_surcharge_eps"] == 0.01
    assert pirp_params["vehicle_capacity_Q"] == 100
    assert pirp_params["shelf_life_tau"] == 3
    assert pirp_params["stockout_penalty_pu"] == 20.0
    assert pirp_params["spoilage_penalty_ps"] == 5.0
    assert pirp_params["holding_cost_h"] == 1.0

    # Check experimental configuration
    exp_config = loaded["experiment_configuration"]
    assert exp_config["seeds"] == [42, 1042, 2042, 3042, 4042]
    assert exp_config["num_test_instances"] == 30
    assert exp_config["num_models"] == 14
    assert exp_config["dfl_epochs"] == 1500


def test_dry_run_execution():
    """Verifies that reproduce_all --dry-run completes successfully in < 15s."""
    res = subprocess.run(
        [sys.executable, "-m", "src.experiments.reproduce_all", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "SMOKE TEST PASSED" in res.stdout


def test_cli_help_flag():
    """Verifies that reproduce_all CLI responds to --help."""
    res = subprocess.run(
        [sys.executable, "-m", "src.experiments.reproduce_all", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "--verify" in res.stdout
    assert "--dry-run" in res.stdout
    assert "--full" in res.stdout
