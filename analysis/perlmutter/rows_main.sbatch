#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 05:00:00
#SBATCH -N 2
#SBATCH -J rows
#SBATCH -o logs/rows_%j.out
#SBATCH -e logs/rows_%j.err
# The profiles with the other fit parameter continuous (profile_rows.py), every stage followed by
# two neighbour-polish passes:
#   1. the central fit: coupling rows every 0.001 and alpha_0 columns every 0.005
#   2. every theory variation, started from the central rows as well
#   3. the generator parameters limited to the inner 90 percent of the box
#   4. the central rows of the two variant fits (product mixture, window bins only)
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export RIVET_REF=rivet_ref OMP_NUM_THREADS=4 TORCH_THREADS=4
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001
NP=$((SLURM_NNODES*32)); F=output/profile_MIX17ext_central.json; S=${F%.json}
run() {     # run <logtag> <fit>: the part stage over all tasks, with the exported settings
  rm -f ${2%.json}_rows${ROWS_TAG:-}_part*.json
  srun -N $SLURM_NNODES -n $NP -c 8 --cpu-bind=cores bash -c \
    "PART=\$SLURM_PROCID NPART=$NP python3 -u profile_rows.py part $2 > logs/rows_$1_\$SLURM_PROCID.log 2>&1"
  python3 -u profile_rows.py merge $2
}
stage() {   # stage <logtag> <fit> <merged file>: one pass, then two polish passes from the merged file
  run $1 $2
  for p in 1 2; do export POLISH_FROM=$3; run $1_pol$p $2; unset POLISH_FROM; done
}
echo "=== 1 central $(date)"
export COLS=1 COL_STEP=0.005; stage central $F ${S}_rows.json; unset COLS COL_STEP
echo "=== 2 variations $(date)"
export ROWS_WHICH=all ROWS_TAG=_var CENTRAL_ROWS=${S}_rows.json; stage var $F ${S}_rows_var.json; unset ROWS_WHICH ROWS_TAG CENTRAL_ROWS
echo "=== 3 shrunk box $(date)"
export BOX_SHRINK=0.9 CENTRAL_ROWS=${S}_rows.json; stage shrink $F ${S}_rows_shrink0.90.json; unset BOX_SHRINK CENTRAL_ROWS
echo "=== 4 variants $(date)"
for V in output/profile_MIX17geo_central.json output/profile_MIX17win_central.json; do
  stage $(basename ${V%.json}) $V ${V%.json}_rows.json
done
echo "ROWS DONE $(date)"
