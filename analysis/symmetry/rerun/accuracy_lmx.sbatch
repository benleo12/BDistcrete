#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J lmx_accuracy
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# accuracy_r8.sbatch of the rerun, for the release phase (after table_lmx; not submitted with the scoring):
# the per-point closure error against displacement (closure_vs_distance.py: the photon-free reference means
# of the refreshed bundles, checked against the published ones, REF_ORIG_DIR, and the held-out runs of the
# _nb folders), then the floor-plus-slope fit (fit_accuracy_law.py), into RUN/release_lmx_table:
# accuracy_law_points.csv and accuracy_law.json. It reads no export, only the bundles and table_lmx's closure files.
#   sbatch --dependency=afterok:<the job id of lmx_table> --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/accuracy_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb data_stagePURE17_v3_h400 || exit 2
PUBM=/mnt/beegfs/bassi/reweighting/maxent_code/output/models      # the published bundles, read only
need_file release_lmx_table/closure_E.csv release_lmx_table/closure_F.csv release_lmx_table/closure_MIX17.csv || exit 2
for T in E_P2v3fullfac0 F_P2v3fullfac0 MIXGEOV3S9; do need_file output/models/${T}_ref.npz $PUBM/${T}_ref.npz || exit 2; done
export OMP_NUM_THREADS=16 STAGEE_CSV=stageE_design_v3all.csv STAGEF_CSV=stageF_design_v3all.csv DM_DATA=data_stagePURE17_v3_h400 \
       STAGEE_DATA=data_stagePURE17_v3_h400 STAGEF_DATA=data_stagePURE17_v3_h400 RELEASE_DIR=release_lmx_table REF_ORIG_DIR=$PUBM
CLOSURE_DIST_OUT=release_lmx_table/accuracy_law_points.csv "$PY" -B -u closure_vs_distance.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 \
  || { echo "ACC FAILED"; exit 3; }
CLOSURE_DIST_IN=release_lmx_table/accuracy_law_points.csv "$PY" -B -u fit_accuracy_law.py || { echo "ACC FAILED"; exit 3; }
echo "ACC DONE $(date)"
