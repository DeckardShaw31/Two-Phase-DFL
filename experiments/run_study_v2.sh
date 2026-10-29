#!/usr/bin/env bash
# Expanded study (200/50/100 train/val/test episodes, three network layouts, UL-DFL with three
# seeds everywhere, LA-DFL with three seeds in the base case). Unattended: finished
# configurations are skipped and a failed run is resumed once.
# Usage: bash experiments/run_study_v2.sh [python-executable]
set -uo pipefail
PY=${1:-python}
OUT=results_v2
mkdir -p $OUT
run() {  # config, extra args...
  c=$1; shift
  if [ -f "$OUT/$c/test_summary.csv" ]; then echo "skip $c"; return; fi
  echo "start $c $(date +%T)"
  $PY -u -m experiments.run_config --config "$c" --out "$OUT/$c" "$@" > "$OUT/${c}_stdout.log" 2>&1 \
    || $PY -u -m experiments.run_config --config "$c" --out "$OUT/$c" "$@" >> "$OUT/${c}_stdout.log" 2>&1
  echo "done $c $(date +%T)"
}
run base --dfl-seeds 0,1,2
run net_c1
run net_rc1
for c in pu40 route3 fleet_tight m2 m4 pu10 fleet_loose highvar; do run "$c"; done
run lifo --reuse $OUT/base
$PY -u -m experiments.solver_sensitivity --results $OUT > $OUT/solver_sensitivity.log 2>&1
echo ALL_V2_DONE
