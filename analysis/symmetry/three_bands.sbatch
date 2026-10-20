#!/bin/bash
#SBATCH -p sharedq
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 12:00:00
#SBATCH -J 3bands
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/three_bands_%j.out
# The three uncertainties of Sec. 4 on one trained mixture (three_bands.py, the run called 3B in
# phase2/INVENTORY.md): the calculation imposed at the default tunes of both generators with the
# mixing fraction at 1/2, the generator band over 200 points of a Latin hypercube, the learning band
# from the wifi fit, the theory band from the variations of the calculation, against DELPHI. The
# mean of members is the prediction (WIFI_CENTRAL=0). The LEP tables are the Rivet reference files
# in rivet_ref (copied from the laptop's Rivet install, md5 checked).
#   TAG=MIXGEOV3S9 sbatch --export=ALL three_bands.sbatch
# EXPORT, REF, WFIT and OUT override the files of TAG (used to rerun the published r7 model).
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8 RIVET_REF=$R/maxent_code/rivet_ref
M=output/models/${TAG:?}
EXPORT=${EXPORT:-${M}_cond.npz}; REF=${REF:-${M}_ref_v2_slim.npz}; WFIT=${WFIT:-output/wifi_${TAG}_fit.npz}
export WIFI_CENTRAL=0 NODES=200 SEED=11 OUT=${OUT:-output/three_bands_${TAG}}
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
echo "=== three bands $TAG on $(hostname): $EXPORT $REF $WFIT -> $OUT $(date)"
python3 -u three_bands.py MIX $EXPORT $REF output/thrust_targets_nlo_point.npz $WFIT || { echo "THREE BANDS FAILED"; exit 3; }
echo "THREE BANDS $TAG DONE $(date)"
