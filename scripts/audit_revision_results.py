"""Audit a completed revision2 primary run and write a concise JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_METHODS = {
    "LightGBM_PTO",
    "LightGBM_Quantile_PTO",
    "Crama_UL_lambda_adaptation",
    "Two-Phase_DFL",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "episode_metrics.csv"
    summary_path = run_dir / "summary.csv"
    paired_path = run_dir / "paired_primary_tests.csv"
    solver_path = run_dir / "solver_diagnostics.csv"
    training_solver_path = run_dir / "dfl_training_solver_diagnostics.csv"
    required = [manifest_path, metrics_path, summary_path, paired_path, solver_path, training_solver_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run is incomplete; missing: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)
    solver = pd.read_csv(solver_path)
    training_solver = pd.read_csv(training_solver_path)

    method_episodes = metrics.groupby("method")["episode_id"].nunique().astype(int).to_dict()
    methods = set(method_episodes)
    expected_ids = set(map(int, manifest["test_episode_ids"]))
    coverage_errors = {
        method: sorted(expected_ids - set(map(int, group.episode_id.unique())))
        for method, group in metrics.groupby("method")
    }
    dfl_seed_counts = metrics[metrics.method == "Two-Phase_DFL"].groupby("episode_id")["training_seed"].nunique()

    component_total = (
        metrics["routing_cost"]
        + metrics["holding_cost"]
        + metrics["spoilage_cost"]
        + metrics["stockout_cost"]
    )
    decomposition_error = np.abs(component_total - metrics["total_cost"])
    spoilage_component_error = np.abs(
        metrics["expiry_cost"] + metrics["capacity_disposal_cost"] - metrics["spoilage_cost"]
    )
    fill_rate = metrics["fill_rate"].to_numpy(dtype=float)
    parity = metrics["solver_fifo_parity_max_abs_error"].dropna().to_numpy(dtype=float)

    dataset_dir = Path(manifest["dataset_directory"])
    hash_errors = []
    for filename, expected_hash in manifest["input_files_sha256"].items():
        path = dataset_dir / filename
        if not path.exists() or _sha256(path) != expected_hash:
            hash_errors.append(filename)

    fallback_col = solver["solver_used_heuristic_fallback"].astype(str).str.lower().eq("true")
    optimal_col = solver["solver_optimal"].astype(str).str.lower().eq("true")
    training_fallback_col = training_solver["used_heuristic_fallback"].astype(str).str.lower().eq("true")
    training_optimal_col = training_solver["is_optimal"].astype(str).str.lower().eq("true")
    milp_requested = training_solver["solver_name"].astype(str).str.contains("MILP", case=False)
    paired_complete = (
        len(paired) == 2
        and paired["n_independent_episodes"].astype(int).eq(len(expected_ids)).all()
        and np.isfinite(paired["paired_t_p_raw"].to_numpy(dtype=float)).all()
        and np.isfinite(paired["holm_adjusted_p"].to_numpy(dtype=float)).all()
    )

    checks = {
        "expected_methods_present": methods == EXPECTED_METHODS,
        "all_methods_cover_test_episodes": all(not missing_ids for missing_ids in coverage_errors.values()),
        "all_60_test_episodes": len(expected_ids) == 60,
        "dfl_has_three_seeds_per_episode": bool(dfl_seed_counts.eq(3).all()),
        "cost_decomposition_error_below_1e-6": bool(decomposition_error.max() < 1e-6),
        "spoilage_components_reconcile_below_1e-6": bool(spoilage_component_error.max() < 1e-6),
        "fill_rate_in_unit_interval": bool(np.all((fill_rate >= -1e-9) & (fill_rate <= 1.0 + 1e-9))),
        "forecast_plan_fifo_parity_below_1e-4": bool(len(parity) > 0 and np.max(parity) < 1e-4),
        "dataset_hashes_match": not hash_errors,
        "paired_primary_tests_complete": bool(paired_complete),
    }
    report = {
        "run_directory": str(run_dir.resolve()),
        "protocol_version": manifest.get("protocol_version"),
        "checks_passed": all(checks.values()),
        "checks": checks,
        "method_episode_counts": method_episodes,
        "uncovered_test_ids": coverage_errors,
        "cost_decomposition_max_abs_error": float(decomposition_error.max()),
        "spoilage_component_max_abs_error": float(spoilage_component_error.max()),
        "fifo_parity_max_abs_error": float(np.max(parity)) if len(parity) else None,
        "evaluation_solver_calls": int(len(solver)),
        "evaluation_solver_fallback_calls": int(fallback_col.sum()),
        "evaluation_solver_optimal_calls": int(optimal_col.sum()),
        "training_solver_calls": int(len(training_solver)),
        "training_milp_requested_calls": int(milp_requested.sum()),
        "training_solver_fallback_calls": int(training_fallback_col.sum()),
        "training_solver_optimal_calls": int(training_optimal_col.sum()),
        "dataset_hash_mismatches": hash_errors,
        "paired_primary_tests": paired.to_dict(orient="records"),
        "summary_methods": summary["method"].tolist(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="results/revision2_final")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = audit_run(run_dir)
    output = Path(args.output) if args.output else run_dir / "audit_report.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
