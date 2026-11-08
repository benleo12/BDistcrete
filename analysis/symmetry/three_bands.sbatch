#!/bin/bash
#SBATCH -p preq
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 12:00:00
#SBATCH -J lmx_3b
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/three_bands_%j.out
# The three uncertainties of Sec. 5 on one trained mixture (three_bands.py): the calculation
# imposed at the default tunes of both generators with the mixing fraction at 1/2, the generator band over the points of a Latin hypercube, the learning band
# from the wifi fit, the theory band from the variations of the calculation, against DELPHI. The
# mean of members is the prediction (WIFI_CENTRAL=0). The LEP tables are the Rivet reference files
# in rivet_ref (copied from the laptop's Rivet install, md5 checked). MIX=linear (the default): the linear
# mixture, 2000 points over BOX (full or tune), the comparisons with the factors of conventions.py (D8: CONV_FACTORS
# or its default places; CONVENTIONS=none leaves them out), outputs end in _lin; MIX=geometric: the earlier draft,
# 200 points.
#   MAXENT_DIR=$RUN TAG=MIXGEOV3S9 sbatch --export=ALL three_bands.sbatch
# It runs only in the rerun's folder RUN (E7): MAXENT_DIR must resolve to it, and the scripts there must have the MIX
# switch. The rerun's checks are on: the bundle must be one of the beam-photon rule in use (BEAM_PHOTONS=remove,
# bundle_rule.py) and the export must carry the refitted temperature (T_DIR, E2). EXPORT, REF, WFIT and OUT override
# the files of TAG (REF: the reference bundle with the observables); TORCH_ENV the Python environment (default
# /mnt/beegfs/bassi/reweighting/env/torch).
set -uo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # E7: the only folder this script runs in
[ "$(realpath "${MAXENT_DIR:-/nonexistent}" 2>/dev/null)" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2; }
grep -q "^MIX = " "$RUN/three_bands.py" || { echo "$RUN/three_bands.py has no MIX switch (an earlier draft's script)"; exit 2; }
TORCH_ENV=${TORCH_ENV:-/mnt/beegfs/bassi/reweighting/env/torch}
cd "$RUN" || exit 2; [ -f $TORCH_ENV/bin/activate ] || { echo "no Python environment at $TORCH_ENV"; exit 2; }; source $TORCH_ENV/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8 RIVET_REF=$RUN/rivet_ref PYTHONDONTWRITEBYTECODE=1
export BEAM_PHOTONS=remove T_DIR=$RUN/output/temperature
M=output/models/${TAG:?}
EXPORT=${EXPORT:-${M}_cond.npz}; REF=${REF:-${M}_ref_v2_slim.npz}; WFIT=${WFIT:-output/wifi_${TAG}_fit.npz}
export MIX=${MIX:-linear}; SFX=$([ "$MIX" = linear ] && echo _lin)
export WIFI_CENTRAL=0 NODES=${NODES:-$([ "$MIX" = linear ] && echo 2000 || echo 200)} SEED=11 BOX=${BOX:-full} OUT=${OUT:-output/three_bands_${TAG}${SFX}}
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
echo "=== three bands $TAG ($MIX, $NODES points over $BOX) on $(hostname): $EXPORT $REF $WFIT -> $OUT $(date)"
python3 -B -u three_bands.py MIX $EXPORT $REF output/thrust_targets_nlo_point.npz $WFIT || { echo "THREE BANDS FAILED"; exit 3; }
echo "THREE BANDS $TAG DONE $(date)"
