#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J lmx_closmix
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# closure_mix.sbatch of the rerun (Fig. 3): fig_closure_mix.py on MIXGEOV3S9, photon-free, at its
# refitted temperature, at the point of median closure width of each generator on the new standard
# test (output/widths_v2_MIXGEOV3S9.json of strict_closure_lmx; the published points were 7214, 8211).
# Writes output/fig_closure_mix.npz, removed when the job starts and recorded when it ends well
# (output/lmx_records/closmix.json).
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/closure_mix_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb data_stagePURE17_v3 || exit 2
rm -f output/fig_closure_mix.npz output/lmx_records/closmix.json
need_file output/widths_v2_MIXGEOV3S9.json output/models/MIXGEOV3S9_ref_v2_slim.npz || exit 2
need_rule output/models/MIXGEOV3S9_ref_v2_slim.npz || exit 2
export OMP_NUM_THREADS=8 DM_DATA=data_stagePURE17_v3 LADDER_HEAD_KIND=geometric LADDER_MIX_SPLIT=8,8 MIX_TAG=MIXGEOV3S9
export MIX_REF=output/models/MIXGEOV3S9_ref_v2_slim.npz REF_PARTICLES=output/published/models/MIXGEOV3S9_ref.npz \
       REF_ORIG=output/published/models/MIXGEOV3S9_ref_v2_slim.npz CLOSURE_MEDIAN_FROM=output/widths_v2_MIXGEOV3S9.json \
       CLOSURE_OUT=output/fig_closure_mix.npz
"$PY" -B -u fig_closure_mix.py || { echo "CLOSURE MIX FAILED"; exit 3; }
step_record closmix output/fig_closure_mix.npz || exit 3
echo "CLOSURE MIX DONE $(date)"
