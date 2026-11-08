#!/bin/bash
#SBATCH -p preq
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J lmx_thvars
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/theory_variations_%j.out
# theory_variations.py on MIXGEOV3S9. MIX=linear (the default) writes output/theory_variations_MIX_lin.json, with the
# comparisons at the hadron level of DELPHI and OPAL (D8, conventions.py; CONVENTIONS=none leaves them out).
#   MAXENT_DIR=$RUN sbatch --export=ALL theory_variations.sbatch
# It runs only in the rerun's folder RUN (E7): MAXENT_DIR must resolve to it, and the script there must have the MIX
# switch. The rerun's checks are on: the bundle must be one of the beam-photon rule in use (BEAM_PHOTONS=remove,
# bundle_rule.py) and the export must carry the refitted temperature (T_DIR, E2). EXPORT, REF and OUT override the
# files; TORCH_ENV the Python environment (default /mnt/beegfs/bassi/reweighting/env/torch).
set -uo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # E7: the only folder this script runs in
[ "$(realpath "${MAXENT_DIR:-/nonexistent}" 2>/dev/null)" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2; }
grep -q "^MIX = " "$RUN/theory_variations.py" || { echo "$RUN/theory_variations.py has no MIX switch (an earlier draft's script)"; exit 2; }
TORCH_ENV=${TORCH_ENV:-/mnt/beegfs/bassi/reweighting/env/torch}
cd "$RUN" || exit 2; [ -f $TORCH_ENV/bin/activate ] || { echo "no Python environment at $TORCH_ENV"; exit 2; }; source $TORCH_ENV/bin/activate
export TORCH_THREADS=4 RIVET_REF=$RUN/rivet_ref MIX=${MIX:-linear} PYTHONDONTWRITEBYTECODE=1
export BEAM_PHOTONS=remove T_DIR=$RUN/output/temperature
EXPORT=${EXPORT:-output/models/MIXGEOV3S9_cond.npz}; REF=${REF:-output/models/MIXGEOV3S9_ref_v2_slim.npz}
export CENTRAL=0.118,0.68,3.9,0.48,0.46,0.17,0.97,0.18,0.102337,0.654714,3.528693,1.849375,0.914156,0.374094,0.33107,0.78,0.5
echo "=== theory variations ($MIX) on $(hostname): $EXPORT $REF -> ${OUT:-default} $(date)"
python3 -B -u theory_variations.py MIX $EXPORT $REF output/thrust_targets_nlo_point.npz || { echo "THVARS FAILED"; exit 3; }
echo "THVARS DONE $(date)"
