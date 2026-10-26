#!/bin/bash
#SBATCH -p preq
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 2:00:00
#SBATCH -J bandchk
#SBATCH -o logs/band_checks_%j.out
# Checks behind the bands of Sec. 4 (band_checks.py): statistical errors, the part of the generator band
# from the mixing fraction, the largest weight. Same model, point and hypercube as three_bands.sbatch.
#   TAG=MIXGEOV3S9 sbatch --export=ALL band_checks.sbatch
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8 RIVET_REF=$MAXENT_DIR/rivet_ref
M=output/models/${TAG:?}
export NODES=200 SEED=11
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
python3 -u band_checks.py MIX ${M}_cond.npz ${M}_ref_v2_slim.npz output/thrust_targets_nlo_point.npz || { echo "BAND CHECKS FAILED"; exit 3; }
echo "BAND CHECKS $TAG DONE $(date)"
