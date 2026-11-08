#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -J lmx_recal
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# D11 for the learning-curve mixtures whose training data are gone (MIXGEOV3E160S9, MIXGEOV3E320S9,
# MIXGEOV4S9, MIXGEOV4L): recalibrate_T.py, the training run's objective and grid on the stop halves
# of the 48 held-out runs (data_stagePURE17_v3: their held-out runs are the same files), with the
# refreshed (photon-free) reference bundle and the particles of the published one. Writes
# output/recal_T_<tag>.json and output/temperature/<tag>.json. Also a cross-check for MIXGEOV3S9,
# whose temperature the standard test refits (specs/std_fac.txt): output/recal_T_MIXGEOV3S9_check.json,
# not written to output/temperature. Its outputs are removed when it starts and recorded when it ends well
# (output/lmx_records/recal.json).
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/recal_lc_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb data_stagePURE17_v3 || exit 2
LC="MIXGEOV3E160S9 MIXGEOV3E320S9 MIXGEOV4S9 MIXGEOV4L"
OUTS="$(for T in $LC; do echo output/recal_T_$T.json output/temperature/$T.json; done) output/recal_T_MIXGEOV3S9_check.json"
rm -f $OUTS output/lmx_records/recal.json
export OMP_NUM_THREADS=16 DM_DATA=data_stagePURE17_v3
one() {   # TAG [check]
  local T=$1 REF=output/models/$1_ref_v2_slim.npz
  need_file output/models/${T}_cond.npz $REF output/published/models/${T}_ref.npz output/published/models/${T}_ref_v2_slim.npz || return 2
  need_rule $REF || return 2
  if [ "${2:-}" = check ]; then
    T_DIR= RECAL_EXPORT=output/models/${T}_cond.npz RECAL_REF=$REF RECAL_REF_PARTICLES=output/published/models/${T}_ref.npz \
      RECAL_REF_ORIG=output/published/models/${T}_ref_v2_slim.npz RECAL_OUT=output/recal_T_${T}_check.json "$PY" -B -u recalibrate_T.py $T
  else
    RECAL_EXPORT=output/models/${T}_cond.npz RECAL_REF=$REF RECAL_REF_PARTICLES=output/published/models/${T}_ref.npz \
      RECAL_REF_ORIG=output/published/models/${T}_ref_v2_slim.npz RECAL_OUT=output/recal_T_${T}.json "$PY" -B -u recalibrate_T.py $T
  fi
}
PIDS=()
for T in $LC; do one $T > logs/recal_${T}_${SLURM_JOB_ID}.log 2>&1 & PIDS+=($!); done
one MIXGEOV3S9 check > logs/recal_MIXGEOV3S9_check_${SLURM_JOB_ID}.log 2>&1 & PIDS+=($!)
FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
grep -h "training grid" logs/recal_*_${SLURM_JOB_ID}.log
[ $FAIL = 0 ] && { step_record recal $OUTS || FAIL=1; }
echo "RECAL DONE (failures: $FAIL) $(date)"; exit $FAIL
