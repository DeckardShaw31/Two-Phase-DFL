#!/usr/bin/env bash
# One-factor-at-a-time variants after the base configuration (see run_all.sh for the full pipeline).
set -uo pipefail
PY=${1:-python}
for c in route3 fleet_tight pu40 m2 m4 pu10 fleet_loose highvar; do
  if [ -f "results/$c/test_summary.csv" ]; then echo "skip $c"; continue; fi
  $PY -u -m experiments.run_config --config "$c" --out "results/$c" > "results/${c}_stdout.log" 2>&1 \
    || $PY -u -m experiments.run_config --config "$c" --out "results/$c" >> "results/${c}_stdout.log" 2>&1
done
$PY -u -m experiments.run_config --config lifo --out results/lifo --reuse results/base > results/lifo_stdout.log 2>&1
echo ALL_VARIANTS_DONE
