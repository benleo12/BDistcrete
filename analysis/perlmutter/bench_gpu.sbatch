#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -J bench_gpu
#SBATCH -o logs/bench_gpu_%j.out
# The cost benchmark (bench_cost.py) on one A100: (a), (n) and (c); the fit step (b) is CPU code.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
# the Rivet reference tables in the working directory unless RIVET_REF is set
export RIVET_REF=${RIVET_REF:-rivet_ref} TARGETS_GRID=output/thrust_targets_grid_ext.npz
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OMP_NUM_THREADS=32 BENCH_DEVICE=cuda BENCH_THREADS=32 python3 -u bench_cost.py output/profile_MIX17ext_central.json
echo "BENCH GPU DONE $(date)"
