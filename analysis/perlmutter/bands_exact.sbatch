#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 2
#SBATCH -J bands
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/bands_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/bands_%j.err
# Exact nuisance bands at the fitted point (bands_exact.py): the nuisance parameters minimized once at
# the point of *_rows.json, every band end a constrained minimization split over both nodes, then the
# merge into *_exchange.json and *_exchange_dist.npz.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
export RIVET_REF=$SCRATCH/maxent_code/rivet_ref OMP_NUM_THREADS=4 TORCH_THREADS=4
export TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float64 FIG_EXPS=${FIG_EXPS:-delphi}
FIT=${FIT:?}; NP=$((SLURM_NNODES*32)); stem=${FIT%.json}; tag=$(basename $stem)
rm -f ${stem}_bands_part*.json
python3 -u bands_exact.py point $FIT || exit 1
srun -N $SLURM_NNODES -n $NP -c 8 --cpu-bind=cores bash -c \
  "PART=\$SLURM_PROCID NPART=$NP python3 -u bands_exact.py part $FIT > logs/bands_${tag}_\$SLURM_PROCID.log 2>&1"
python3 -u bands_exact.py merge $FIT
echo "BANDS DONE $(date)"
