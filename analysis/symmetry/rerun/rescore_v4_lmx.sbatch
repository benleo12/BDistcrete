#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J lmx_rescorev4
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# rescore_v4.sbatch of the rerun: the two 2768-run mixtures against a reference of the 720-run size
# (415 of the 1000 stored events of each run), on the photon-free observables, at their refitted
# temperatures (recal_lc_lmx, apply_T_lmx). Standard test on data_stagePURE17_v3, whose 48 held-out
# runs are the files of data_stagePURE17_v4's (its training runs are deleted, and only the held-out
# runs are read), strict test on data_stagePURE17_v3_h400. Writes output/widths_v2_<TAG>r415.json and
# output/widths_v2_<TAG>h400r415.json, removed when it starts and recorded when it ends well
# (output/lmx_records/rescorev4.json).
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/rescore_v4_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb data_stagePURE17_v3 data_stagePURE17_v3_h400 || exit 2
OUTS=$(for T in MIXGEOV4S9 MIXGEOV4L; do echo output/widths_v2_${T}r415.json output/widths_v2_${T}h400r415.json; done)
rm -f $OUTS output/lmx_records/rescorev4.json
export OMP_NUM_THREADS=20 REF_PER_RUN=415 WIDTHS_SUFFIX=r415
one() {
  local T=$1 REF=output/models/$1_ref_v2_slim.npz
  need_file output/models/${T}_cond.npz $REF output/published/models/${T}_ref.npz output/published/models/${T}_ref_v2_slim.npz || return 2
  need_rule $REF || return 2
  export REF_PARTICLES=output/published/models/${T}_ref.npz REF_ORIG=output/published/models/${T}_ref_v2_slim.npz
  DM_DATA=data_stagePURE17_v3 WIDTHS_OUT=output/widths_v2_${T}r415.json "$PY" -B -u ladder_widths_big.py MIX output/models/${T}_cond.npz $REF || return 3
  ln -sfn ${T}_cond.npz output/models/${T}h400_cond.npz
  DM_DATA=data_stagePURE17_v3_h400 WIDTHS_OUT=output/widths_v2_${T}h400r415.json "$PY" -B -u ladder_widths_big.py MIX output/models/${T}h400_cond.npz $REF || return 3
}
PIDS=(); for T in MIXGEOV4S9 MIXGEOV4L; do one $T > logs/rescore_${T}_${SLURM_JOB_ID}.log 2>&1 & PIDS+=($!); done
FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
grep -h "report-split width" logs/rescore_*_${SLURM_JOB_ID}.log
[ $FAIL = 0 ] && { step_record rescorev4 $OUTS || FAIL=1; }
echo "RESCORE DONE (failures: $FAIL) $(date)"; exit $FAIL
