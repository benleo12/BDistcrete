#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J lmx_widths
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# strict_closure.sbatch of the rerun, with the standard test that train_mixgeo.sbatch wrote after
# training: ladder_widths_big.py for each factorized mixture of the learning curve and of the paper,
# on the photon-free observables, at the refitted temperature its export now carries (apply_T_lmx).
#   standard test  output/widths_v2_<TAG>.json      on data_stagePURE17_v3 (MIXGEOV3E20S9: _v3ev20)
#   strict test    output/widths_v2_<TAG>h400.json  on data_stagePURE17_v3_h400
# The reference is the refreshed bundle output/models/<TAG>_ref_v2_slim.npz (photon-free shapes,
# checked against the published one and against output/bundle_rule.json), with the multiplicity and the
# kaon fraction of its particles. The outputs are removed when the job starts and recorded when every tag
# ended well (output/lmx_records/widths.json; with a TAGS of its own, a test, nothing is recorded).
#   sbatch --export=ALL,MAXENT_DIR=$RUN[,TAGS="..."] symmetry/rerun/strict_closure_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
ALL="MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2 MIXGEOV3M8S9 MIXGEOV3S27F MIXGEOV3E20S9 MIXGEOV3E160S9 MIXGEOV3E320S9"
TAGS=${TAGS:-$ALL}
need_nb data_stagePURE17_v3 data_stagePURE17_v3ev20 data_stagePURE17_v3_h400 || exit 2
OUTS=$(for T in $TAGS; do echo output/widths_v2_${T}.json output/widths_v2_${T}h400.json; done)
rm -f $OUTS; [ "$TAGS" = "$ALL" ] && rm -f output/lmx_records/widths.json
export OMP_NUM_THREADS=10
one() {
  local T=$1 STD=data_stagePURE17_v3; [ $T = MIXGEOV3E20S9 ] && STD=data_stagePURE17_v3ev20
  local REF=output/models/${T}_ref_v2_slim.npz
  need_file output/models/${T}_cond.npz $REF output/published/models/${T}_ref.npz output/published/models/${T}_ref_v2_slim.npz || return 2
  need_rule $REF || return 2
  export REF_PARTICLES=output/published/models/${T}_ref.npz REF_ORIG=output/published/models/${T}_ref_v2_slim.npz
  echo "=== $T standard test on $STD $(date)"
  DM_DATA=$STD WIDTHS_OUT=output/widths_v2_${T}.json "$PY" -B -u ladder_widths_big.py MIX output/models/${T}_cond.npz $REF || return 3
  ln -sfn ${T}_cond.npz output/models/${T}h400_cond.npz
  echo "=== $T strict test $(date)"
  DM_DATA=data_stagePURE17_v3_h400 WIDTHS_OUT=output/widths_v2_${T}h400.json "$PY" -B -u ladder_widths_big.py MIX output/models/${T}h400_cond.npz $REF || return 3
}
PIDS=(); for T in $TAGS; do one $T > logs/widths_${T}_${SLURM_JOB_ID}.log 2>&1 & PIDS+=($!); done
FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
grep -h "report-split width" logs/widths_*_${SLURM_JOB_ID}.log
if [ $FAIL = 0 ]; then
  if [ "$TAGS" = "$ALL" ]; then step_record widths $OUTS || FAIL=1; else echo "TAGS is not every tag: no record"; fi
fi
echo "WIDTHS DONE (failures: $FAIL) $(date)"; exit $FAIL
