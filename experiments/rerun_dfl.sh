#!/usr/bin/env bash
# Re-run the DFL stage (training, early stopping, test evaluation) of every configuration,
# reusing the tuned policies in chosen_pre_dfl.joblib; used after the float64 target fix.
set -uo pipefail
PY=${1:-python}
for c in base m2 m4 pu10 pu40 fleet_tight fleet_loose highvar route3; do
  echo "start $c $(date +%T)"
  $PY -u -m experiments.run_config --config "$c" --out "results/$c" > "results/${c}_rerun_stdout.log" 2>&1 \
    || $PY -u -m experiments.run_config --config "$c" --out "results/$c" >> "results/${c}_rerun_stdout.log" 2>&1
  echo "done $c $(date +%T)"
done
$PY -u -m experiments.run_config --config lifo --out results/lifo --reuse results/base > results/lifo_rerun_stdout.log 2>&1
echo ALL_RERUNS_DONE
