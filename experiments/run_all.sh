#!/usr/bin/env bash
# Reproduce every result in the revised manuscript (about 6-8 hours on a 16-core desktop).
# Usage: bash experiments/run_all.sh [python-executable]
set -euo pipefail
PY=${1:-python}
$PY -m pytest -q tests
$PY -m experiments.verify_theory --out results/theory
$PY -m experiments.run_config --config base --out results/base
for c in m2 m4 pu10 pu40 fleet_tight fleet_loose highvar route3; do
  $PY -m experiments.run_config --config "$c" --out "results/$c"
done
# issuing-policy robustness: policies tuned/trained under FIFO, evaluated with 30% LIFO customers
$PY -m experiments.run_config --config lifo --out results/lifo --reuse results/base
$PY -m experiments.make_figures --results results --out revision_1/figures
