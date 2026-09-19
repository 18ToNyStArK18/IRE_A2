#!/usr/bin/env bash
# Regenerate every re-ranker artifact after enabling the history-content features
# (config.HISTORY_CONTENT_FEATURES, src/history_content.py).
#
#   bash scripts/rerun_history_content.sh            # from the repo root
#   PY=python bash scripts/rerun_history_content.sh  # if the venv lives elsewhere
#
# Assumes the processed data, candidates and embeddings already exist:
#   python build_pipeline.py && python -m src.candidates --method all
# NRMS, the Codabench files, paired_bootstrap and age_signal do not use the
# re-ranker and are not touched.
#
# Sequential, one process per step so memory is released in between, and stops
# at the first failure. Progress: log/rerun_history_content/steps.log.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python} -u"
L=log/rerun_history_content
B=data/backup_pre_history_content
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$L"
: > "$L/steps.log"

step() {  # step <name> <command...>
  local name=$1; shift
  local started=$(date +%s)
  echo "$(date '+%F %T') START $name | free MiB: $(free -m 2>/dev/null | awk '/Mem:/{print $7}')" | tee -a "$L/steps.log"
  "$@" > "$L/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc $(( ($(date +%s)-started)/60 )) min" | tee -a "$L/steps.log"
  if [ $rc -ne 0 ]; then echo "FAILED at $name -- see $L/$name.log" | tee -a "$L/steps.log"; exit $rc; fi
}

$PY -c "from src import config; assert config.HISTORY_CONTENT_FEATURES, 'switch is off'" || exit 1

# 0. Back up what gets overwritten (once; a rerun keeps the original backup).
if [ ! -d "$B" ]; then
  mkdir -p "$B"
  for d in mind ebnerd; do mkdir -p "$B/$d" && cp -r "data/processed/$d/reranker" "$B/$d/"; done
  cp data/processed/reranker_summary_test.csv "$B/" 2>/dev/null
  cp -r results "$B/"
fi

# 1. Re-rankers: every stage-1 method, both datasets; keep each summary row.
for d in ebnerd mind; do for m in popular bm25_fresh bm25 semantic; do
  step reranker_${d}_${m} $PY -m src.reranker --dataset $d --method $m
  cp data/processed/reranker_summary_test.csv "$L/summary_${d}_${m}.csv"
done; done
step reranker_summary $PY -c "
import glob, pandas as pd
pd.concat([pd.read_csv(f) for f in sorted(glob.glob('$L/summary_*.csv'))]).to_csv('data/processed/reranker_summary_test.csv', index=False)"

# 2. Q9 ablation, now with the no_history_content arm (the new features' paired CI).
for d in ebnerd mind; do step q9_ablation_$d $PY scripts/serving_features_ablation.py --dataset $d; done

# 3. Q5 extended evaluation.
for d in ebnerd mind; do for m in popular bm25; do
  step extended_eval_${d}_${m} $PY scripts/extended_eval.py --dataset $d --method $m
done; done

# 4. Q4 serving benchmark (now times the history-content features per request).
for d in ebnerd mind; do step serving_benchmark_$d $PY scripts/serving_benchmark.py --dataset $d; done

# 5. Tests against the rebuilt artifacts.
step pytest $PY -m pytest -q tests/

echo "DONE. Send back: results/*.json, data/processed/*/reranker/report_*_test.json," \
     "data/processed/reranker_summary_test.csv and $L/" | tee -a "$L/steps.log"
