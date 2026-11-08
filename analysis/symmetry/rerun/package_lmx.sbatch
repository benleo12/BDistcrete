#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J lmx_package
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# package_r8.sbatch of the rerun, for the release phase (after apply_T_lmx; not submitted with the scoring):
# the networks of Stage D (Sherpa, export E_P2v3fullfac0, public name E), Stage E (Herwig, F_P2v3fullfac0,
# public F) and the mixture (MIXGEOV3S9, public MIX17), on their v3 designs, into RUN/release_lmx. The
# packager copies the export's temperature into the released head, so every export is first checked
# against its refit in output/temperature (score_obs.check_export_T). It reads the reference bundles for
# their particles only, which D4 leaves alone. The mixture's head is packaged as trained (geometric); the
# linear mixture of D1 for the release is an open release item (RELEASE_TODO.md). It reads the exports of
# Stages D and E, so it waits for apply_T_rel_lmx.sbatch (submitted with the scoring) and stops without its record.
#   sbatch --dependency=afterok:<the job id of lmx_applyTrel> --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/package_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb data_stagePURE17_v3 || exit 2
"$PY" -B symmetry/rerun/lmx_fingerprint.py has-record applyTrel || { echo "apply_T_rel_lmx.sbatch has not ended well in this rerun"; exit 2; }
TAGS="E_P2v3fullfac0 F_P2v3fullfac0 MIXGEOV3S9"
for T in $TAGS; do need_file output/models/${T}_cond.npz output/models/${T}_trunk.pt output/models/${T}_ref.npz output/temperature/$T.json || exit 2; done
"$PY" -B - $TAGS <<'PY' || { echo "EXPORT TEMPERATURE CHECK FAILED"; exit 3; }
import sys, numpy as np, score_obs as SO
for t in sys.argv[1:]:
    e = f'output/models/{t}_cond.npz'
    print(f'{t}: the export carries T = {SO.check_export_T(e, float(np.load(e)["temperature"]))}, its refit')
PY
# package_release.py reads gentune from the release folder it writes (sys.path.insert(0, RELEASE_DIR))
mkdir -p release_lmx/models && { [ -d release_lmx/gentune ] || cp -R release/gentune release_lmx/gentune; } || exit 2
export OMP_NUM_THREADS=16 STAGEE_CSV=stageE_design_v3all.csv STAGEF_CSV=stageF_design_v3all.csv DM_DATA=data_stagePURE17_v3 RELEASE_DIR=release_lmx
"$PY" -B -u package_release.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || { echo "PACKAGING FAILED"; exit 3; }
echo "PACKAGING DONE $(date)"
