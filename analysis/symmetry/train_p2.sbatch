#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 6
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -J p2
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/p2_%A_%a.out
# Phase 2: every stage of the paper with both networks, three trainings each, at the configuration
# the learning curve selects (K, STEPS), in the protocol of the paper: four bootstrap members, a
# fifth of each run held out as the fit set, mean of members, temperature on the stop split,
# closure on the report split. Both networks go through r2_ladder.py and differ only in the
# function class. One array element per "STAGE:NET:SEED" in SPECS.
#   STAGE  A, B, C (ladder), E (Sherpa), F (Herwig), MIX (mixture);  NET  fac or cat;  SEED 0,1,2
#   sbatch --array=0-17%6 --export=ALL,K=48,STEPS=324000,SPECS="A:fac:0 A:cat:0 ..." train_p2.sbatch
# Data: ABC_DATA for A-C (gen/data_abc), E_DATA/E_CSV and F_DATA/F_CSV for the single generators,
# DM_DATA for the mixture.
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
read -ra ALL <<< "${SPECS:?}"; IFS=: read -r STAGE NET SEED <<< "${ALL[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12} LADDER_ENS=4 LADDER_K=${K:?} LADDER_STEPS=${STEPS:?} LADDER_SEED_OFFSET=$((SEED*1000))
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 LADDER_LOWMEM=1 LADDER_MMAP=1
export LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1 LADDER_DEAD=report LADDER_REVIVE=1 LADDER_PATIENCE=${PATIENCE:-0}
export LADDER_GPU_FETCH=${GPU_FETCH:-0}   # 1: standardize each batch on the GPU (bitwise equal, faster)
MMAP_BASE=${MMAP_BASE:-/tmp}; mkdir -p $MMAP_BASE
# working memory maps in RAM (MMAP_BASE=/dev/shm) outlive a job that is killed without its EXIT trap:
# remove those of our own jobs that are no longer running before adding one (skipped if squeue fails)
if [ "$MMAP_BASE" = /dev/shm ] && LIVE=" $(squeue -h -u $USER -o %A 2>/dev/null | tr '\n' ' ') " && [ "$LIVE" != "  " ]; then
  for d in /dev/shm/bassi_mmap_*; do [ -e "$d" ] || continue; j=${d##*_}; case "$LIVE" in *" $j "*) ;; *) rm -rf "$d" ;; esac; done
fi
export LADDER_MMAP_DIR=$MMAP_BASE/bassi_mmap_p2_${STAGE}_${NET}_${SEED}_$SLURM_JOB_ID
trap 'rm -rf $LADDER_MMAP_DIR' EXIT
DT=${DESIGN_TAG:-}   # e.g. aug for the densified ladder designs (ABC_DESIGN); keeps outputs apart
[ -n "${ABC_DESIGN:-}" ] && export LADDER_ABC_DESIGN=$ABC_DESIGN
TAG=P2${DT}${NET}${SEED}
echo "=== P2 stage $STAGE net $NET seed $SEED K $LADDER_K steps $LADDER_STEPS on $(hostname) $(date)"
if [ "$STAGE" = MIX ]; then
  export LADDER_HEAD_KIND=$([ $NET = cat ] && echo concat_geometric || echo geometric)
  export DM_DATA=${DM_DATA:?} DM_TAG=MIX$TAG LADDER_EMB_STORE=a STAGED_OUT=output/p2${DT}_MIX_${NET}_s${SEED}.json
  python3 -u stage_mixture.py || { echo "TRAIN FAILED"; exit 3; }
else
  export LADDER_HEAD_KIND=$([ $NET = cat ] && echo concat || echo cond)
  export LADDER_STAGES=$STAGE LADDER_SUFFIX=_$TAG LADDER_OUT=output/p2${DT}_${STAGE}_${NET}_s${SEED}.json
  export STAGEA_DATA=${ABC_DATA:-$R/gen/data_abc} STAGEB_DATA=${ABC_DATA:-$R/gen/data_abc} STAGEC_DATA=${ABC_DATA:-$R/gen/data_abc}
  export STAGEE_DATA=${E_DATA:-data_stageE} STAGEE_CSV=${E_CSV:-stageE_design.csv}
  export STAGEF_DATA=${F_DATA:-data_stageF} STAGEF_CSV=${F_CSV:-stageF_design_aug.csv}
  python3 -u r2_ladder.py || { echo "TRAIN FAILED"; exit 3; }
fi
echo "P2 $STAGE $NET $SEED DONE $(date)"
