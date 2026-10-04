#!/usr/bin/env bash
# Re-evaluate the toy K-scan on the paper's ruler (r2_ladder.pulls / width_of).
# Training is identical to kscan.toy_scan (same seeds, same schedule); only the estimator
# differs. One process per (K, seed) because the box is CPU-contended; each is single-threaded.
# Produces output/_kruler/K*_s*.json, then output/kscan_toy_ruler.json via the merge step.
cd .
PY=python
mkdir -p output/_kruler
for K in 1 2 4 8 16 32; do
  for SD in 0 1; do
    KSCAN_DEV=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      $PY -u kscan_toy_ruler.py --ks $K --seedlist $SD \
      --out output/_kruler/K${K}_s${SD}.json > output/_kruler/K${K}_s${SD}.log 2>&1 &
  done
done
wait
$PY -u kscan_toy_ruler_merge.py
echo "KSCAN TOY RULER SCAN DONE"
