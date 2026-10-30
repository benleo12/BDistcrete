#!/bin/bash
#SBATCH -p sharedq
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J envelope
#SBATCH -o logs/envelope_test_%j.out
# The envelope of NDRAW variations against the linear covariance (envelope_test.py) on the model, files and
# central point of strategy_test.sbatch, for one BOX (full, tune, tune_xifull, tune_sherpa, tune_herwig).
#   TAG=MIXGEOV3S9 BOX=full NREF=2000 sbatch --export=ALL symmetry/envelope_test.sbatch
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8 RIVET_REF=$MAXENT_DIR/rivet_ref
M=output/models/${TAG:?}
EXPORT=${EXPORT:-${M}_cond.npz}; REF=${REF:-${M}_ref_v2_slim.npz}; WFIT=${WFIT:-output/wifi_${TAG}_fit.npz}
export BOX=${BOX:-full} NDRAW=${NDRAW:-400} NREF=${NREF:-0} SEED=${SEED:-11} TUNE_SCALE=${TUNE_SCALE:-0.3333333333333333}
export XI_NODES=${XI_NODES:-8} OUT=${OUT:-output/envelope_test_${TAG}_${BOX:-full}}
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
echo "=== envelope test $TAG $BOX on $(hostname): $EXPORT $REF $WFIT -> $OUT $(date)"
python3 -u envelope_test.py $TAG $EXPORT $REF output/thrust_targets_nlo_point.npz $WFIT || { echo "ENVELOPE TEST FAILED"; exit 3; }
echo "ENVELOPE TEST $TAG $BOX DONE $(date)"
