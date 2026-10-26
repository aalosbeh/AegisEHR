#!/bin/bash
# Part A, leave-one-profile-out, analysis, drift, noise/pairwise/latency (CPU only).
set -e
PY=${PY:-python}
W=${W:-6}
cd "$(dirname "$0")/.."
rm -f results/partA_records.jsonl results/loso_*_records.jsonl
$PY scripts/02_run_partA.py --workers $W
for s in Site-A Site-B Site-C; do $PY scripts/02_run_partA.py --workers $W --holdout $s; done
$PY scripts/04_analyze_partA.py > results/analysis_partA.log
$PY scripts/05_drift.py --workers $W > results/drift.log
$PY scripts/07_noise_pairwise_latency.py --workers $W > results/noise.log
