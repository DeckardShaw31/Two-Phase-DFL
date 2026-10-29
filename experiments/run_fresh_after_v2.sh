#!/usr/bin/env bash
# Real-data case (FreshRetailNet-50K), queued behind run_study_v2.sh so the two do not compete
# for cores; afterwards the solver-sensitivity check is repeated to include it.
set -uo pipefail
PY=${1:-python}
OUT=results_v2
until grep -q ALL_V2_DONE results_v2_study.log 2>/dev/null; do sleep 120; done
if [ ! -f "$OUT/fresh/test_summary.csv" ]; then
  echo "start fresh $(date +%T)"
  $PY -u -m experiments.run_config --config fresh --out "$OUT/fresh" > "$OUT/fresh_stdout.log" 2>&1 \
    || $PY -u -m experiments.run_config --config fresh --out "$OUT/fresh" >> "$OUT/fresh_stdout.log" 2>&1
  echo "done fresh $(date +%T)"
fi
$PY -u -m experiments.solver_sensitivity --results $OUT > $OUT/solver_sensitivity.log 2>&1
echo ALL_FRESH_DONE
