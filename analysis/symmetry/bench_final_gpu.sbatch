#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH -t 2:00:00
#SBATCH -J benchgpu
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/bench_final_gpu_%j.out
# Cost benchmark of Sec. 3.6 (bench_cost_final.py) on one H200, with the gradient check.
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
export OMP_NUM_THREADS=12 RIVET_REF=$R/maxent_code/rivet_ref
export BENCH_DEVICE=cuda BENCH_THREADS=12
echo "=== bench gpu on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -u bench_cost_final.py || { echo "BENCH FAILED"; exit 3; }
echo "BENCH GPU DONE $(date)"
