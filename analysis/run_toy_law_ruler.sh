#!/usr/bin/env bash
# Re-score the toy subsection's dense-coverage comparison (split_scan.py) and accuracy law
# (budget_law.py) on the paper's ruler. Training is identical to the originals; only the
# estimator changes. One process per job because the box is CPU-contended.
cd .
PY=python
mkdir -p output/_law
run() { KSCAN_DEV=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        $PY -u toy_law_ruler.py "$@"; }
# dense-coverage: the five splits of split_scan.py at fixed M*E = 60000, two seeds each
for ME in "12 5000" "60 1000" "600 100" "6000 10" "60000 1"; do
  set -- $ME
  for SD in 0 1; do
    run --mode split --M $1 --E $2 --seed $SD --out output/_law/split_M$1_E$2_s${SD}.json \
        > output/_law/split_M$1_E$2_s${SD}.log 2>&1 &
  done
done
# accuracy law: the three seeds of budget_law.py
for SD in 0 1 2; do
  run --mode law --seed $SD --out output/_law/law_s${SD}.json \
      > output/_law/law_s${SD}.log 2>&1 &
done
wait
$PY -u toy_law_ruler_merge.py
echo "TOY LAW RULER DONE"
