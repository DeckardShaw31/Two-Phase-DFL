"""
Script to bundle an anonymized replication package for journal peer review.

Excludes:
- .git/
- .venv/
- __pycache__/ and *.pyc
- .pytest_cache/
- Local user environment logs and machine-specific cache files.
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_ROOT / "supplementary_material"
OUTPUT_ZIP = OUTPUT_DIR / "replication_package.zip"


def create_anonymized_readme(dest_path: Path):
    """Creates a blinded, double-blind compliant README for peer reviewers."""
    content = """# Replication Package: Decision-Focused Learning for Perishable Inventory-Routing

## Overview
This replication package contains the complete, self-contained source code, optimization models, preprocessed benchmark instances, and evaluation suites supporting the double-blind submission:

**"End-to-End Decision-Focused Learning for Perishable Inventory-Routing under Asymmetric Spoilage and Stockout Costs"**

## Directory Structure
- `src/`: Complete optimization model (`PIRPModel`), DFL pipelines (`SPO+`, `IntOpt`, `DPO`, `Two-Phase_DFL`), baselines, and evaluation metrics.
- `data/processed/`: Calibrated empirical demand series, Solomon spatial network topologies (`C1`, `R1`, `RC1`), and parameter configurations.
- `results/`: Benchmarks, multi-seed metrics (1,500 epochs), 81-configuration sensitivity grid, capacity diagnostics, and publication figures.
- `tests/`: Full automated test suite (150 tests covering formulation, FIFO dynamics, complete recourse, and numerical parity).
- `requirements.txt`: Python package dependencies.

## Quick Replication Instructions

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows:
.venv\\Scripts\\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify All Numerical Results and Manuscript Parity
To verify that all reported metrics, tables, and figures match the experimental outputs to the exact decimal:
```bash
python -m src.experiments.audit_manuscript
```

### 3. Run the Full Test Suite
To execute the complete 150-test verification suite:
```bash
pytest tests -v
```

### 4. End-to-End Pipeline Verification
```bash
python -m src.experiments.reproduce_all --verify
```
"""
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_replication_package():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.is_file():
        OUTPUT_ZIP.unlink()

    include_dirs = ["src", "data/processed", "results", "tests", "scripts"]
    include_files = ["requirements.txt", "pyproject.toml", "LICENSE"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir) / "Two-Phase-DFL-Replication"
        tmp_root.mkdir()

        # Copy directories
        for d in include_dirs:
            src_dir = WORKSPACE_ROOT / d
            if src_dir.is_dir():
                dest_dir = tmp_root / d
                shutil.copytree(
                    src_dir,
                    dest_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp", "*.log"),
                )

        # Copy individual files
        for f in include_files:
            src_file = WORKSPACE_ROOT / f
            if src_file.is_file():
                shutil.copy2(src_file, tmp_root / f)

        # Create anonymized README
        create_anonymized_readme(tmp_root / "README.md")

        # Create zip archive
        print(f"Creating zip archive at: {OUTPUT_ZIP}")
        with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp_root):
                for file in files:
                    full_path = Path(root) / file
                    arcname = full_path.relative_to(tmp_root)
                    zf.write(full_path, arcname)

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Replication package created successfully: {OUTPUT_ZIP} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    build_replication_package()
