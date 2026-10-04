#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 02:30:00
#SBATCH -N 2
#SBATCH -J rows_var2
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/rows_variants_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/rows_variants_%j.err
# The central rows of the two variant fits (product mixture, window bins only), one node each, a pass
# and two polish passes, written as *_rows_sep.json so they cannot collide with stage 4 of rows_main.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
export RIVET_REF=$SCRATCH/maxent_code/rivet_ref OMP_NUM_THREADS=4 TORCH_THREADS=4
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001 ROWS_TAG=_sep
stage_v() {   # stage_v <fit>: one pass and two polish passes on one node
  local F=$1 S=${1%.json} T=$(basename ${1%.json})
  for p in 0 1 2; do
    rm -f ${S}_rows_sep_part*.json
    if [ $p -gt 0 ]; then export POLISH_FROM=${S}_rows_sep.json; else unset POLISH_FROM; fi
    srun --exact -N 1 -n 32 -c 8 --cpu-bind=cores bash -c \
      "PART=\$SLURM_PROCID NPART=32 python3 -u profile_rows.py part $F > logs/rows_sep_${T}_p${p}_\$SLURM_PROCID.log 2>&1"
    python3 -u profile_rows.py merge $F
  done
}
stage_v output/profile_MIX17geo_central.json > logs/rows_sep_geo.out 2>&1 &
stage_v output/profile_MIX17win_central.json > logs/rows_sep_win.out 2>&1 &
wait
echo "VARIANTS DONE $(date)"
