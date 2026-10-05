#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -J clos_wide
#SBATCH -o logs/closure_wide_%j.out
#SBATCH -e logs/closure_wide_%j.err
# The pseudo-data sets whose minimum lay at the end of their rows, rerun with rows over twice the
# range (same seeds, so the same truth and pseudo-data), written as *_setNNN_wide.json.
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export RIVET_REF=rivet_ref OMP_NUM_THREADS=32 TORCH_THREADS=32
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001 ROW_HALF=0.024
# SETS is exported because the srun'd shell reads it. The wide files sit next to the sets of
# closure_rows.sbatch, and the summary is redone with them replacing the originals.
FIT=output/profile_MIX17ext_central.json; tag=$(basename ${FIT%.json}); export SETS=${SETS:-"2 22"}
mkdir -p logs output/pseudo_rows_sets
srun -N 1 -n $(echo $SETS | wc -w) -c 64 --cpu-bind=cores bash -c \
  "S=(\$SETS); k=\${S[\$SLURM_PROCID]}; PSEUDO_SET=\$k PSEUDO_OUT=output/pseudo_rows_sets/pseudo_rows_${tag}_set\$(printf %03d \$k)_wide.json python3 -u pseudo_rows.py $FIT > logs/pseudo_rows_wide_\$k.log 2>&1"
python3 -u pseudo_rows.py summary output/pseudo_rows_sets/pseudo_rows_${tag}_set*.json
cp output/pseudo_rows_sets/pseudo_rows_${tag}_summary.json output/
echo "CLOSURE WIDE DONE $(date)"
