#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -J lmx_stagec
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# The Stage C results, photon-free (D4), at the refitted temperature of C_P2augfullfac0 (D11, apply_T_lmx):
#   Fig. 4 and \Map*  eval_valid.py, report halves, the 5 held-out and 12 blind runs, with the x_p
#                      spectrum and the joint (1-T, mult) test   -> output/valid_map_C_P2augfullfac0_report.json
#   \Single*          single_members.py (it fits each member's temperature itself)
#                                                                 -> output/single_members_C_P2augfullfac0.json
#   \XpNullMean       xp_null.py, two halves of the comparison half of runs 6900-6904 -> output/xp_null.json
# (\TruncMaxPercent is counted by trunc_lmx.sbatch, on the original shapes files.)
# The reference observables come from the refreshed bundle output/models/C_P2augfullfac0_ref.npz (listed in
# output/bundle_rule.json), the multiplicity, kaon fraction and x_p of the reference from the particles of the
# published one. The outputs are removed when the job starts and recorded when it ends well
# (output/lmx_records/stagec.json).
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/stagec_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
need_nb $R/gen/data_abc $R/gen/data_abc_blind || exit 2
C=C_P2augfullfac0; REF=output/models/${C}_ref.npz; PUBREF=output/published/models/${C}_ref.npz
OUTS="output/valid_map_${C}_report.json output/single_members_$C.json output/xp_null.json"
rm -f $OUTS output/lmx_records/stagec.json
need_file output/models/${C}_cond.npz $REF $PUBREF output/temperature/$C.json || exit 2
need_rule $REF || exit 2
export OMP_NUM_THREADS=16
( VALID_MODEL=$C VALID_DATA=$R/gen/data_abc VALID_BLIND_DATA=$R/gen/data_abc_blind VALID_HALF=report \
  VALID_OUT=output/valid_map_${C}_report.json VALID_REF=$REF VALID_REF_PARTICLES=$PUBREF VALID_REF_ORIG=$PUBREF \
  "$PY" -B -u eval_valid.py ) > logs/stagec_map_${SLURM_JOB_ID}.log 2>&1 &
P1=$!
( SM_REF=$REF SM_REF_PARTICLES=$PUBREF SM_REF_ORIG=$PUBREF SM_OUT=output/single_members_$C.json \
  "$PY" -B -u single_members.py $C C $R/gen/data_abc ) > logs/stagec_single_${SLURM_JOB_ID}.log 2>&1 &
P2=$!
( XP_NULL_OUT=output/xp_null.json "$PY" -B -u xp_null.py $R/gen/data_abc ) > logs/stagec_xpnull_${SLURM_JOB_ID}.log 2>&1 &
P3=$!
FAIL=0; for p in $P1 $P2 $P3; do wait $p || FAIL=1; done
tail -n 3 logs/stagec_*_${SLURM_JOB_ID}.log
[ $FAIL = 0 ] && { step_record stagec $OUTS || FAIL=1; }
echo "STAGE C DONE (failures: $FAIL) $(date)"; exit $FAIL
