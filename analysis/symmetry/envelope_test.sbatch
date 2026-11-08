#!/bin/bash
#SBATCH -p preq
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J lmx_env
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/envelope_test_%j.out
# The envelope of NDRAW variations against the linear covariance (envelope_test.py) on the model, files and
# central point of strategy_test.sbatch, for one BOX. MIX=linear (the default): BOX is tune, full, tune_sherpa,
# tune_herwig, full_sherpa or full_herwig, 2000 draws, outputs end in _lin; NREF=1000 on the full box adds the
# further draws that measure the sampling noise. MIX=geometric: the earlier draft's boxes (full, tune,
# tune_xifull, tune_sherpa, tune_herwig) and 400 draws.
#   MAXENT_DIR=$RUN TAG=MIXGEOV3S9 BOX=tune sbatch --export=ALL symmetry/envelope_test.sbatch
# It runs only in the rerun's folder RUN (E7): MAXENT_DIR must resolve to it, and the script there must have the MIX
# switch. The rerun's checks are on: the bundle must be one of the beam-photon rule in use (BEAM_PHOTONS=remove,
# bundle_rule.py) and the export must carry the refitted temperature (T_DIR, E2). EXPORT, REF, WFIT and OUT override
# the files of TAG; TORCH_ENV the Python environment (default /mnt/beegfs/bassi/reweighting/env/torch).
set -uo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # E7: the only folder this script runs in
[ "$(realpath "${MAXENT_DIR:-/nonexistent}" 2>/dev/null)" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2; }
grep -q "^MIX = " "$RUN/envelope_test.py" || { echo "$RUN/envelope_test.py has no MIX switch (an earlier draft's script)"; exit 2; }
TORCH_ENV=${TORCH_ENV:-/mnt/beegfs/bassi/reweighting/env/torch}
cd "$RUN" || exit 2; [ -f $TORCH_ENV/bin/activate ] || { echo "no Python environment at $TORCH_ENV"; exit 2; }; source $TORCH_ENV/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8 RIVET_REF=$RUN/rivet_ref PYTHONDONTWRITEBYTECODE=1
export BEAM_PHOTONS=remove T_DIR=$RUN/output/temperature
M=output/models/${TAG:?}
EXPORT=${EXPORT:-${M}_cond.npz}; REF=${REF:-${M}_ref_v2_slim.npz}; WFIT=${WFIT:-output/wifi_${TAG}_fit.npz}
export MIX=${MIX:-linear}; SFX=$([ "$MIX" = linear ] && echo _lin)
export BOX=${BOX:-full} NDRAW=${NDRAW:-$([ "$MIX" = linear ] && echo 2000 || echo 400)} NREF=${NREF:-0} SEED=${SEED:-11} TUNE_SCALE=${TUNE_SCALE:-0.3333333333333333}
export XI_NODES=${XI_NODES:-8} OUT=${OUT:-output/envelope_test_${TAG}_${BOX:-full}${SFX}}
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
echo "=== envelope test $TAG $BOX ($MIX, $NDRAW draws, NREF $NREF) on $(hostname): $EXPORT $REF $WFIT -> $OUT $(date)"
python3 -B -u envelope_test.py $TAG $EXPORT $REF output/thrust_targets_nlo_point.npz $WFIT || { echo "ENVELOPE TEST FAILED"; exit 3; }
echo "ENVELOPE TEST $TAG $BOX DONE $(date)"
