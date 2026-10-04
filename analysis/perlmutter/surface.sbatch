#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -J surface
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/surface_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/surface_%j.err
# The two-parameter chi^2 surface of the fit (surface_rows.py) for the confidence regions of
# Fig. anchored_fit (c): one pass over the coupling rows, then two polish passes over the points.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
export RIVET_REF=$SCRATCH/maxent_code/rivet_ref OMP_NUM_THREADS=4 TORCH_THREADS=4
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32
export SURF_AS=${SURF_AS:-0.108,0.132,0.001} SURF_A0=${SURF_A0:-0.30,0.60,0.01}
FIT=${FIT:?}; stem=${FIT%.json}; tag=$(basename $stem); NP=32
run() {
  rm -f ${stem}_surface_part*.json
  srun -N 1 -n $NP -c 8 --cpu-bind=cores bash -c \
    "PART=\$SLURM_PROCID NPART=$NP python3 -u surface_rows.py part $FIT > logs/surface_${tag}_$1_\$SLURM_PROCID.log 2>&1"
  python3 -u surface_rows.py merge $FIT
}
run pass0
for p in 1 2; do export POLISH_FROM=${stem}_surface.json; run pol$p; unset POLISH_FROM; done
echo "SURFACE DONE $(date)"
