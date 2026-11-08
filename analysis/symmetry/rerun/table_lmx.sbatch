#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -J lmx_table
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# table_r8.sbatch of the rerun, for the release phase (after package_lmx; not submitted with the scoring),
# on the photon-free observables (D4) at the refitted temperatures (D11):
#   make_stage_table.py  closure of the released networks at the held-out runs of their designs
#                        -> RUN/release_lmx_table/STAGES.md, closure_<stage>.csv, closure_summary.json
#   precision_gap.py     the float16 gap of the packaged trunks -> output/precision_gap_r8.json
# The reference observables come from the refreshed bundles output/models/<tag>_ref.npz, checked against
# the published bundles of the same events (REF_ORIG_DIR, read only), with the multiplicity and the kaon
# fraction without the beam photons of their particles; the held-out runs from the _nb folders. Each export
# is checked against its refit (T_DIR), and each packaged head against its export. C_1M, which the shipped
# release also documents, has no bundle on Symmetry and is left out here (RELEASE_TODO.md). It reads the
# exports of Stages D and E, so it also needs the record of apply_T_rel_lmx.sbatch.
#   sbatch --dependency=afterok:<the job id of lmx_package> --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/table_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb data_stagePURE17_v3_h400 || exit 2
"$PY" -B symmetry/rerun/lmx_fingerprint.py has-record applyTrel || { echo "apply_T_rel_lmx.sbatch has not ended well in this rerun"; exit 2; }
PUBM=/mnt/beegfs/bassi/reweighting/maxent_code/output/models      # the published bundles, read only
need_file release_lmx/models/MANIFEST.json || exit 2
for T in E_P2v3fullfac0 F_P2v3fullfac0 MIXGEOV3S9; do
  need_file output/models/${T}_cond.npz output/models/${T}_ref.npz $PUBM/${T}_ref.npz output/temperature/$T.json || exit 2
done
mkdir -p release_lmx_table/models && cp release_lmx/models/MANIFEST.json release_lmx_table/models/MANIFEST.json
export OMP_NUM_THREADS=16 STAGEE_CSV=stageE_design_v3all.csv STAGEF_CSV=stageF_design_v3all.csv DM_DATA=data_stagePURE17_v3_h400 \
       STAGEE_DATA=data_stagePURE17_v3_h400 STAGEF_DATA=data_stagePURE17_v3_h400 REF_ORIG_DIR=$PUBM
RELEASE_DIR=release_lmx_table "$PY" -B -u make_stage_table.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || { echo "TABLE FAILED"; exit 3; }
RELEASE_DIR=release_lmx "$PY" -B -u precision_gap.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || { echo "GAP FAILED"; exit 3; }
echo "TABLE DONE $(date)"
