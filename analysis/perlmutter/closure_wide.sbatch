#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -J clos_wide
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/closure_wide_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/closure_wide_%j.err
# The pseudo-data sets whose minimum lay at the end of their rows, rerun with rows over twice the
# range (same seeds, so the same truth and pseudo-data), written as *_setNNN_wide.json.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
export RIVET_REF=$SCRATCH/maxent_code/rivet_ref OMP_NUM_THREADS=32 TORCH_THREADS=32
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001 ROW_HALF=0.024
FIT=output/profile_MIX17ext_central.json; tag=$(basename ${FIT%.json}); SETS=${SETS:-"2 22"}
srun -N 1 -n $(echo $SETS | wc -w) -c 64 --cpu-bind=cores bash -c \
  "S=(\$SETS); k=\${S[\$SLURM_PROCID]}; PSEUDO_SET=\$k PSEUDO_OUT=output/pseudo_rows_${tag}_set\$(printf %03d \$k)_wide.json python3 -u pseudo_rows.py $FIT > logs/pseudo_rows_wide_\$k.log 2>&1"
echo "CLOSURE WIDE DONE $(date)"
