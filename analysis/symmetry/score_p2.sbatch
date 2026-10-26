#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 12
#SBATCH --mem=400G
#SBATCH -t 12:00:00
#SBATCH -J score
#SBATCH -o logs/score_%j.out
# Score an already trained ensemble on a (possibly larger) held-out set with the training code's
# own estimator (r2_ladder.py, LADDER_SCORE_FROM): data, reference and split rebuilt exactly as in
# training, the saved members loaded, the temperature fixed in training.
#   STAGE  MIX or A, B, C, E, F;  NET fac or cat;  MODEL the saved ensemble (..._trunk.pt / ..._concat.pt)
#   DATA   the data set: DM_DATA for MIX, the stage's data directory otherwise;  OUT the result JSON
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=12 LADDER_ENS=4 LADDER_K=${K:-48} LADDER_STEPS=${STEPS:-324000} LADDER_SEED_OFFSET=$((${SEED:-0}*1000))
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 LADDER_LOWMEM=1 LADDER_MMAP=1
export LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1 LADDER_DEAD=report LADDER_REVIVE=1
export LADDER_SCORE_FROM=${MODEL:?}
[ -n "${ABC_DESIGN:-}" ] && export LADDER_ABC_DESIGN=$ABC_DESIGN   # the design the model was trained on
MMAP_BASE=${MMAP_BASE:-/tmp}; mkdir -p $MMAP_BASE
# working memory maps in RAM (MMAP_BASE=/dev/shm) outlive a job that is killed without its EXIT trap:
# remove those of our own jobs that are no longer running before adding one (skipped if squeue fails)
if [ "$MMAP_BASE" = /dev/shm ] && LIVE=" $(squeue -h -u $USER -o %A 2>/dev/null | tr '\n' ' ') " && [ "$LIVE" != "  " ]; then
  for d in /dev/shm/${USER}_mmap_*; do [ -e "$d" ] || continue; j=${d##*_}; case "$LIVE" in *" $j "*) ;; *) rm -rf "$d" ;; esac; done
fi
export LADDER_MMAP_DIR=$MMAP_BASE/${USER}_mmap_score_$SLURM_JOB_ID; trap 'rm -rf $LADDER_MMAP_DIR' EXIT
echo "=== score $MODEL stage ${STAGE:?} net ${NET:?} on ${DATA:?} $(date)"
if [ "$STAGE" = MIX ]; then
  export LADDER_HEAD_KIND=$([ $NET = cat ] && echo concat_geometric || echo geometric)
  export DM_DATA=$DATA DM_TAG=MIXscore$SLURM_JOB_ID LADDER_EMB_STORE=a STAGED_OUT=${OUT:?}
  python3 -u stage_mixture.py || { echo "SCORE FAILED"; exit 3; }
else
  export LADDER_HEAD_KIND=$([ $NET = cat ] && echo concat || echo cond)
  export LADDER_STAGES=$STAGE LADDER_SUFFIX=_score$SLURM_JOB_ID LADDER_OUT=${OUT:?}
  export STAGEA_DATA=$DATA STAGEB_DATA=$DATA STAGEC_DATA=$DATA STAGEE_DATA=$DATA STAGEF_DATA=$DATA
  export STAGEE_CSV=${E_CSV:-stageE_design.csv} STAGEF_CSV=${F_CSV:-stageF_design_aug.csv}
  python3 -u r2_ladder.py || { echo "SCORE FAILED"; exit 3; }
fi
echo "SCORE DONE $(date)"
