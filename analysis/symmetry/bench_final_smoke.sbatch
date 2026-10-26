#!/bin/bash
#SBATCH -p debugq
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 1:00:00
#SBATCH -J benchsmk
#SBATCH -o logs/bench_final_smoke_%j.out
# Smoke test of the CPU path of bench_cost_final.py (directlib's logit and fit step, the late
# network) without the concatenation network and the one-time passes, one repeat each.
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RIVET_REF=$MAXENT_DIR/rivet_ref
export BENCH_DEVICE=cpu BENCH_THREADS=4 BENCH_REPS=1 BENCH_SKIP=early,c BENCH_OUT=output/bench_final_smoke_cpu.json
python3 -u bench_cost_final.py || { echo "SMOKE FAILED"; exit 3; }
echo "SMOKE DONE $(date)"
