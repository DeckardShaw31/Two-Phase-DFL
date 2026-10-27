"""Audit the post-hoc finite-scenario robust comparator outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "episode_metrics.csv")
    candidates = pd.read_json(run_dir / "candidate_diagnostics.jsonl", lines=True)
    comparisons = pd.read_csv(run_dir / "exploratory_paired_comparisons.csv")
    expected_ids = set(map(int, manifest["test_episode_ids"]))
    observed_ids = set(map(int, metrics["episode_id"].unique()))
    # Older output manifests did not record the dataset directory explicitly.
    data_dir = Path(manifest.get(
        "dataset_directory",
        run_dir.resolve().parents[1] / "data" / "revision2_final" / "processed",
    ))
    hash_errors = [
        name
        for name, expected in manifest["input_hashes"].items()
        if not (data_dir / name).exists() or sha256(data_dir / name) != expected
    ]

    candidate_counts = candidates.groupby("episode_id").size()
    component_error = np.abs(
        metrics.routing_cost + metrics.holding_cost + metrics.spoilage_cost + metrics.stockout_cost - metrics.total_cost
    )
    spoilage_error = np.abs(metrics.expiry_cost + metrics.capacity_disposal_cost - metrics.spoilage_cost)
    parity_error = metrics.solver_fifo_parity_max_abs_error.to_numpy(dtype=float)
    candidate_parity_error = candidates.fifo_parity_max_abs_error.to_numpy(dtype=float)
    fallback_count = int(metrics.solver_fallback_candidate_count.sum())
    total_candidate_count = int(metrics.solver_candidate_count.sum())
    checks = {
        "all_manifest_test_episodes_covered_once": len(metrics) == len(expected_ids) and observed_ids == expected_ids,
        "five_candidates_per_test_episode": bool(candidate_counts.eq(5).all() and len(candidate_counts) == len(expected_ids)),
        "candidate_rows_match_manifest_counts": len(candidates) == total_candidate_count == 5 * len(expected_ids),
        "cost_decomposition_error_below_1e-6": bool(component_error.max() < 1e-6),
        "spoilage_components_reconcile_below_1e-6": bool(spoilage_error.max() < 1e-6),
        "fifo_state_parity_below_1e-4": bool(np.isfinite(parity_error).all() and parity_error.max() < 1e-4),
        "all_candidate_fifo_state_parity_below_1e-4": bool(
            np.isfinite(candidate_parity_error).all() and candidate_parity_error.max() < 1e-4
        ),
        "dataset_hashes_match": not hash_errors,
        "exploratory_pairs_cover_all_episodes": bool(
            len(comparisons) == 2 and comparisons.n_paired_test_episodes.eq(len(expected_ids)).all()
        ),
        "method_not_claimed_as_source_reproduction": "not a source-faithful" in manifest["status"]
        or "not a reproduction" in manifest["status"]
        or "not implement" in manifest["source_scope"],
    }
    report = {
        "run_directory": str(run_dir.resolve()),
        "protocol_version": manifest.get("protocol_version"),
        "checks_passed": all(checks.values()),
        "checks": checks,
        "n_test_episodes": len(metrics),
        "candidate_plan_rows": len(candidates),
        "solver_fallback_candidates": fallback_count,
        "solver_total_candidates": total_candidate_count,
        "max_fifo_parity_error": float(parity_error.max()) if len(parity_error) else None,
        "max_candidate_fifo_parity_error": float(candidate_parity_error.max()) if len(candidate_parity_error) else None,
        "max_cost_decomposition_error": float(component_error.max()),
        "max_spoilage_reconciliation_error": float(spoilage_error.max()),
        "dataset_hash_mismatches": hash_errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="results/revision2_robust_adaptation_exploratory")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = audit(run_dir)
    output = Path(args.output) if args.output else run_dir / "audit_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
