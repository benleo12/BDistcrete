#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 2
#SBATCH -J closure_atfit
#SBATCH -o logs/closure_atfit_%j.out
#SBATCH -e logs/closure_atfit_%j.err
# Pseudo-data with every truth at the real-data fitted point (pseudo_rows_atfit.py), forty sets. Their
# scatter, 0.0052 in the coupling and 0.062 in alpha_0 with correlation +0.92, is the experimental
# error of the paper. The sets used in the paper are in output/pseudo_atfit_sets/.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
export RIVET_REF=$SCRATCH/maxent_code/rivet_ref OMP_NUM_THREADS=6 TORCH_THREADS=6
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001 ROW_HALF=0.012 SEED_BASE=9000
FIT=output/profile_MIX17ext_central.json; NSETS=${NSETS:-40}; tag=$(basename ${FIT%.json})
srun -N $SLURM_NNODES -n $NSETS -c 12 --cpu-bind=cores bash -c \
  "PSEUDO_SET=\$SLURM_PROCID PSEUDO_OUT=output/pseudo_atfit_${tag}_set\$(printf %03d \$SLURM_PROCID).json python3 -u pseudo_rows_atfit.py $FIT > logs/pseudo_atfit_${tag}_\$SLURM_PROCID.log 2>&1"
python3 -u pseudo_rows_atfit.py summary output/pseudo_atfit_${tag}_set*.json
echo "CLOSURE ATFIT DONE $(date)"
