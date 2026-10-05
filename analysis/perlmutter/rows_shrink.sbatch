#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH -J rows_shr
#SBATCH -o logs/rows_shrink_%j.out
#SBATCH -e logs/rows_shrink_%j.err
# The central rows with the generator parameters limited to the inner 90 percent of the box, a pass
# and two polish passes, written as *_rows_shrink0.90sep.json so they cannot collide with stage 3 of
# rows_main.
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export RIVET_REF=rivet_ref OMP_NUM_THREADS=4 TORCH_THREADS=4
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 ROW_STEP=0.001
export BOX_SHRINK=0.9 CENTRAL_ROWS=output/profile_MIX17ext_central_rows.json ROWS_TAG=_shrink0.90sep
F=output/profile_MIX17ext_central.json; S=${F%.json}; NP=$((SLURM_NNODES*32))
for p in 0 1 2; do
  rm -f ${S}_rows${ROWS_TAG}_part*.json
  if [ $p -gt 0 ]; then export POLISH_FROM=${S}_rows${ROWS_TAG}.json; else unset POLISH_FROM; fi
  srun -N $SLURM_NNODES -n $NP -c 8 --cpu-bind=cores bash -c \
    "PART=\$SLURM_PROCID NPART=$NP python3 -u profile_rows.py part $F > logs/rows_shrsep_p${p}_\$SLURM_PROCID.log 2>&1"
  python3 -u profile_rows.py merge $F
done
echo "SHRINK DONE $(date)"
