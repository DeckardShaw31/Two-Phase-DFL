#!/usr/bin/env bash
# Runs the expanded study (results_v2) in chunks so the machine can cool down in between:
# each call works through the queue for about HOURS hours and stops; a setting is only started
# if its estimated run time still fits. Finished settings are skipped; a finished setting that
# lacks the NN-DFL ablation is extended (NN-DFL trained and evaluated, nothing else re-run), and
# an interrupted setting resumes from its saved tuning/learning stage.
# `touch results_v2/PAUSE` stops the chunk after the setting that is currently running.
# FIRST="fresh m2" moves those settings to the front of the queue; the solver-sensitivity
# check always runs last, once every setting is finished.
# Usage: [FIRST="..."] bash experiments/run_chunk.sh [hours=4] [workers=13] [python-executable]
set -uo pipefail
HOURS=${1:-4}
W=${2:-13}
PY=${3:-python}
OUT=results_v2
LOG=results_v2_chunks.log
ALL="base net_c1 net_rc1 pu40 route3 fleet_tight fresh m2 m4 pu10 fleet_loose highvar lifo"
FIRST=${FIRST:-}
QUEUE="$FIRST $(for c in $ALL; do case " $FIRST " in *" $c "*) ;; *) echo -n "$c ";; esac; done)"
mkdir -p $OUT
rm -f $OUT/PAUSE
T0=$(date +%s)
args() {  # extra arguments per setting
  case $1 in
    base) echo "--dfl-seeds 0,1,2" ;;
    lifo) echo "--reuse $OUT/base" ;;
  esac
}
est() {  # rough minutes with 13 workers: extension of a finished setting, or a full run
  if [ -f "$OUT/$1/test_summary.csv" ]; then echo 20; return; fi
  case $1 in lifo) echo 30 ;; fresh) echo 120 ;; solver_sensitivity) echo 60 ;; *) echo 105 ;; esac
}
fits() {  # does a step of $1 minutes still fit into the chunk?
  [ ! -f "$OUT/PAUSE" ] && [ $(( ($(date +%s) - T0) / 60 + $1 )) -le $(( HOURS * 60 )) ]
}
for c in $QUEUE; do
  if [ -f "$OUT/$c/test_summary.csv" ] && grep -q '^NN-DFL,' "$OUT/$c/test_summary.csv"; then continue; fi
  if ! fits "$(est $c)"; then echo "CHUNK_DONE next=$c $(date '+%F %T')" >> $LOG; exit 0; fi
  echo "start $c $(date '+%F %T') workers=$W" >> $LOG
  $PY -u -m experiments.run_config --config "$c" --out "$OUT/$c" --workers "$W" $(args $c) >> "$OUT/${c}_stdout.log" 2>&1 \
    || $PY -u -m experiments.run_config --config "$c" --out "$OUT/$c" --workers "$W" $(args $c) >> "$OUT/${c}_stdout.log" 2>&1
  echo "done $c $(date '+%F %T')" >> $LOG
done
if [ ! -f "$OUT/solver_sensitivity.csv" ]; then
  if ! fits "$(est solver_sensitivity)"; then echo "CHUNK_DONE next=solver_sensitivity $(date '+%F %T')" >> $LOG; exit 0; fi
  echo "start solver_sensitivity $(date '+%F %T')" >> $LOG
  $PY -u -m experiments.solver_sensitivity --results $OUT --workers "$W" > $OUT/solver_sensitivity.log 2>&1
fi
echo "ALL_DONE $(date '+%F %T')" >> $LOG
