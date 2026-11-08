#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -J lmx_score
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# score_p2.sbatch of the rerun: one trained model scored with the training code's own estimator
# (r2_ladder.py, LADDER_SCORE_FROM), on the photon-free observables (D4, BEAM_PHOTONS=remove), on a CPU
# node of preq (a DCTR network may be given a GPU partition at sbatch time instead, -p and --gres on the
# command line, as submit_scoring_lmx.sh does with LMX_PARTITION and LMX_GRES; r2_ladder.py then uses the
# GPU). The data, reference and split are rebuilt exactly as in training; the light path
# (LADDER_SCORE_LIGHT=1) builds the features of the reference draw alone, bit for bit as the memory
# map would, and writes no memory map. Nothing is trained and nothing is exported.
#   TMODE=refit   the standard test: the temperature is fitted again on the stop split by the training
#                 run's procedure (D11) and written to output/temperature/<model>.json
#   TMODE=table   the strict test, SvdRank and the 48-point scorings: the temperature from that file
# The settings come from the environment, one scoring line of specs/*.txt (make_specs.py):
#   OUT STAGE NET SEED MODEL DATA TMODE [K ENS TRAINED ABC E_CSV F_CSV STAGEW_CSV REF_RUNS NREF_TOTAL
#   SUFFIX DUMP TRUNCATE LOGIT_CACHE NTHREADS]
# Alone:   MAXENT_DIR=$RUN sbatch --export=ALL,<the line> score_p2_lmx.sbatch   (once the start mark exists)
# Several per node: score_p2_array_lmx.sbatch runs this file with bash for each line.
# A finished scoring writes the fingerprint of its code, data record, spec line and start mark into its result
# (lmx_fingerprint.py); a result is kept, and the scoring skipped, only when it carries that fingerprint.
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
# the knobs of the scorers (r2_ladder.py, stage_mixture.py, mixture_cfg.py, score_obs.py) and the python of
# common_lmx.sh, as the submitting shell may pass them (--export=ALL): cleared first, so that only common_lmx.sh
# and the lines below set them. STAGEW_CSV is a key of the spec line (SPEC_KEYS) and stays.
unset $(compgen -e | grep -E '^(LADDER_|DM_)|^STAGE._(DATA|CSV)$|^(STAGED_OUT|PHOTON_FREE_DIR_RE|BEAM_PHOTONS_ALLOW_MIXED|TORCH_PY)$' \
        | grep -vx STAGEW_CSV)
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
: "${OUT:?}" "${STAGE:?}" "${NET:?}" "${SEED:?}" "${MODEL:?}" "${DATA:?}" "${TMODE:?}"
case "$OUT" in output/*.json) ;; *) echo "OUT must be output/<name>.json"; exit 2 ;; esac
case "$TMODE" in refit|table) ;; *) echo "TMODE must be refit or table"; exit 2 ;; esac
need_nb "$DATA" || exit 2
export CUDA_CACHE_PATH=$L/cuda-cache; mkdir -p "$CUDA_CACHE_PATH"     # a GPU's compiled kernels, not in the home folder
# the fingerprint of this scoring (E3): the md5 of the scoring code (beam_photons.py among it) and of
# output/bundle_rule.json, the spec line and the start mark output/.lmx_rerun_start of submit_scoring_lmx.sh.
# preq requeues a preempted job from its start, and RESUME=1 submits again: a result is kept only when it parses
# as JSON and carries this fingerprint; any other result file is removed before the scoring starts, with what
# the scoring writes beside it (its refitted temperature, the weight dumps), so a failed scoring leaves none
read -r FP FPCODE < <("$PY" -B symmetry/rerun/lmx_fingerprint.py fp) || true
[ -n "${FP:-}" ] && [ -n "${FPCODE:-}" ] || { echo "no fingerprint for $OUT"; exit 2; }
"$PY" -B symmetry/rerun/lmx_fingerprint.py keep "$OUT" "$FP" && exit 0
need_gpu || exit 3                  # on h200preq: torch must see the GPU, before anything is removed (common_lmx.sh)
KEY=$(basename "$(realpath "$MODEL")"); KEY=${KEY%_trunk.pt}; KEY=${KEY%_concat.pt}; KEY=${KEY%_cond.npz}
rm -f "$OUT"              # (r2_ladder.py would also merge into an existing result file)
[ "$TMODE" = refit ] && rm -f "$T_DIR/$KEY.json"
[ "${DUMP:-}" = 1 ] && rm -f output/models/${STAGE}${SUFFIX:-}_w_*.npy
need_file "$MODEL" || exit 2
[ "$NET" = cat ] || need_file "${MODEL%_trunk.pt}_cond.npz" || exit 2
[ -n "${TRAINED:-}" ] && { need_file "$TRAINED" || exit 2; }
export OMP_NUM_THREADS=${NTHREADS:-40} LADDER_ENS=${ENS:-4} LADDER_K=${K:-48} LADDER_STEPS=${STEPS:-324000} LADDER_SEED_OFFSET=$((SEED*1000))
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 LADDER_LOWMEM=1 LADDER_MMAP=1
export LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1 LADDER_DEAD=report LADDER_REVIVE=1
export LADDER_SCORE_FROM=$MODEL LADDER_SCORE_T=$TMODE LADDER_SCORE_LIGHT=1
export LADDER_TRAINED_RESULT=${TRAINED:-} LADDER_TRUNCATE=${TRUNCATE:-} LADDER_DUMP_WEIGHTS=${DUMP:-}
# a DCTR network's log ratio at the held-out points, kept for the strict test and SvdRank of the same model, in a
# folder of this mark and of this r2_ladder.py (F2): a new mark or new network code starts an empty cache
LADDER_LOGIT_CACHE=
if [ -n "${LOGIT_CACHE:-}" ]; then
  LADDER_LOGIT_CACHE=$("$PY" -B symmetry/rerun/lmx_fingerprint.py cachedir "$LOGIT_CACHE") || { echo "no cache folder for $OUT"; exit 2; }
fi
export LADDER_LOGIT_CACHE
if [ "${ABC:-}" = 1 ]; then need_file "$R/gen/designs/stageA_design_aug.csv" || exit 2; export LADDER_ABC_DESIGN=$R/gen/designs; fi
export LADDER_MMAP_DIR=/dev/shm/${USER}_mmap_lmxscore_${SLURM_JOB_ID:-x}_$$; trap 'rm -rf $LADDER_MMAP_DIR' EXIT
TAGX=${SLURM_JOB_ID:-$$}_$$
echo "=== score $MODEL ($(realpath $MODEL)) stage $STAGE net $NET on $DATA ($(realpath $DATA)), T $TMODE -> $OUT on $(hostname) $(date)"
if [ "$STAGE" = MIX ]; then
  export LADDER_HEAD_KIND=$([ $NET = cat ] && echo concat_geometric || echo geometric)
  export DM_DATA=$DATA DM_TAG=MIXscore$TAGX LADDER_EMB_STORE=a STAGED_OUT=$OUT
  "$PY" -B -u stage_mixture.py || { echo "SCORE FAILED $OUT"; exit 3; }
else
  export LADDER_HEAD_KIND=$([ $NET = cat ] && echo concat || echo cond)
  # the suffix names only the weight dumps of the wide scan (DUMP=1), which keep the training's names
  export LADDER_STAGES=$STAGE LADDER_SUFFIX=${SUFFIX:-_score$TAGX} LADDER_OUT=$OUT
  export STAGEA_DATA=$DATA STAGEB_DATA=$DATA STAGEC_DATA=$DATA STAGEE_DATA=$DATA STAGEF_DATA=$DATA STAGEW_DATA=$DATA
  export STAGEE_CSV=${E_CSV:-stageE_design.csv} STAGEF_CSV=${F_CSV:-stageF_design_aug.csv}
  if [ "$STAGE" = W ]; then
    need_file "${STAGEW_CSV:?}" || exit 2
    export STAGEW_CSV LADDER_REF_RUNS=${REF_RUNS:-} LADDER_NREF_TOTAL=${NREF_TOTAL:?}
  fi
  "$PY" -B -u r2_ladder.py || { echo "SCORE FAILED $OUT"; exit 3; }
fi
"$PY" -B symmetry/rerun/lmx_fingerprint.py stamp "$OUT" "$FP" "$FPCODE" || { echo "SCORE FAILED $OUT (no fingerprint written)"; exit 3; }
echo "SCORE DONE $OUT $(date)"
