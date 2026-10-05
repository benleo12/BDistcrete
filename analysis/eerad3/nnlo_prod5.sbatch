#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=49
#SBATCH -t 06:00:00
#SBATCH -J eerad3_nnlo5_y8x
#SBATCH -o %x-%j.out
# EERAD3 NNLO thrust, y0=1e-8, SAME settings as prod2 (nshot5=50000, itmax2=2, two-sweep grids),
# 196 NEW seeds (indices 504..699) to add to the 504 of prod2: cross-rank error x 0.85.
set -uo pipefail
R=$SCRATCH/eerad3_thrust
mkdir -p $R/prod4 && cd $R/prod4
cp $R/prod2/eerad3.input eerad3.input
date; echo "196 production ranks, seeds 504..699, nshot5=50000 itmax2=2, y0=1d-8"
srun -n 196 --cpu-bind=cores $R/perl2/prod_rank7.sh
date; echo "completed ranks: $(grep -l 'only \[NNLO\]' prod_*.log 2>/dev/null | wc -l)/196"
echo EERAD3_PROD5_DONE
