#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J bench_cpu
#SBATCH -o logs/bench_cpu_%j.out
# The cost benchmark (bench_cost.py) on one CPU node: with 4 threads, the configuration of every
# process of the Sec. 6 fits, and with the whole node.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
# the Rivet reference tables in the working directory unless RIVET_REF is set
export RIVET_REF=${RIVET_REF:-rivet_ref} TARGETS_GRID=output/thrust_targets_grid_ext.npz
lscpu | grep -E "Model name|^CPU\(s\)"
for th in ${THREADS_LIST:-4 128}; do
  OMP_NUM_THREADS=$th BENCH_DEVICE=cpu BENCH_THREADS=$th python3 -u bench_cost.py output/profile_MIX17ext_central.json
done
echo "BENCH CPU DONE $(date)"
