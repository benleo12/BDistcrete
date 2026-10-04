#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 2
#SBATCH -J closure
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/closure_rows_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/closure_rows_%j.err
# Pseudo-data closure of the fit with the estimator of profile_rows.py (pseudo_rows.py), NSETS sets.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
export RIVET_REF=$SCRATCH/maxent_code/rivet_ref OMP_NUM_THREADS=6 TORCH_THREADS=6
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001 ROW_HALF=0.012
FIT=${FIT:?}; NSETS=${NSETS:-40}; tag=$(basename ${FIT%.json})
srun -N $SLURM_NNODES -n $NSETS -c 12 --cpu-bind=cores bash -c \
  "PSEUDO_SET=\$SLURM_PROCID PSEUDO_OUT=output/pseudo_rows_${tag}_set\$(printf %03d \$SLURM_PROCID).json python3 -u pseudo_rows.py $FIT > logs/pseudo_rows_${tag}_\$SLURM_PROCID.log 2>&1"
python3 -u pseudo_rows.py summary output/pseudo_rows_${tag}_set*.json
echo "CLOSURE DONE $(date)"
