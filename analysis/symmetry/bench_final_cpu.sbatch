#!/bin/bash
#SBATCH -p defq
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -J benchcpu
#SBATCH -o logs/bench_final_cpu_%j.out
# Cost benchmark of Sec. 3.7 (bench_cost_final.py) on one CPU process with four threads, on a node of
# its own so that no other job shares its memory bandwidth. The concatenation network is timed once
# without warm-up (tens of minutes per call), and its full fit step is left out (it is its network
# step plus the design-independent part, both measured).
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RIVET_REF=$MAXENT_DIR/rivet_ref
export BENCH_DEVICE=cpu BENCH_THREADS=4 BENCH_EARLY_REPS=1 BENCH_EARLY_WARMUP=0 BENCH_SKIP=b_early
echo "=== bench cpu on $(hostname) $(date)"; lscpu | grep -i "model name"
python3 -u bench_cost_final.py || { echo "BENCH FAILED"; exit 3; }
echo "BENCH CPU DONE $(date)"
