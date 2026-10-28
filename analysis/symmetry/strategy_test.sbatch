#!/bin/bash
#SBATCH -p sharedq
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J strattest
#SBATCH -o logs/strategy_test_%j.out
# Strategies for the combined band (strategy_test.py) on the model, files, central point, hypercube (NODES,
# SEED) and wifi fit of three_bands.sbatch: the re-solved generator band checked against the stored
# output/three_bands_${TAG}.npz, linear generator bands at the box centre and at the central point, a width
# scan, linear against re-solved theory, a joint Monte Carlo over (theta, omega, c) and the hybrid.
#   TAG=MIXGEOV3S9 sbatch --export=ALL symmetry/strategy_test.sbatch
# EXPORT, REF, WFIT, STORED and OUT override the files of TAG; MC_N, HYB_N, WIDTHS, MC_NP and MC_CRN the stages;
# LIN_DTYPE (float64) the head of S1-S6, a second AE array in memory (S0 keeps the float32 head of three_bands.py).
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8 RIVET_REF=$MAXENT_DIR/rivet_ref
M=output/models/${TAG:?}
EXPORT=${EXPORT:-${M}_cond.npz}; REF=${REF:-${M}_ref_v2_slim.npz}; WFIT=${WFIT:-output/wifi_${TAG}_fit.npz}
export WIFI_CENTRAL=0 NODES=200 SEED=11 OUT=${OUT:-output/strategy_test_${TAG}} STORED=${STORED:-output/three_bands_${TAG}}
export MC_N=${MC_N:-400} HYB_N=${HYB_N:-50} MC_NP=${MC_NP:-0} MC_CRN=${MC_CRN:-0} LIN_DTYPE=${LIN_DTYPE:-float64}
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
echo "=== strategy test $TAG on $(hostname): $EXPORT $REF $WFIT against $STORED.npz -> $OUT $(date)"
[ -f "$STORED.npz" ] || echo "warning: $STORED.npz not found, S0 is not checked against the stored band"
python3 -u strategy_test.py $TAG $EXPORT $REF output/thrust_targets_nlo_point.npz $WFIT || { echo "STRATEGY TEST FAILED"; exit 3; }
echo "STRATEGY TEST $TAG DONE $(date)"
