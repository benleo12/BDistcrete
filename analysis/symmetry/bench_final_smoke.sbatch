#!/bin/bash
#SBATCH -p debugq
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 1:00:00
#SBATCH -J benchsmk
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/bench_final_smoke_%j.out
# Smoke test of the CPU path of bench_cost_final.py (directlib's logit and fit step, the late
# network) without the concatenation network and the one-time passes, one repeat each.
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RIVET_REF=$R/maxent_code/rivet_ref
export BENCH_DEVICE=cpu BENCH_THREADS=4 BENCH_REPS=1 BENCH_SKIP=early,c BENCH_OUT=output/bench_final_smoke_cpu.json
python3 -u bench_cost_final.py || { echo "SMOKE FAILED"; exit 3; }
echo "SMOKE DONE $(date)"
