#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -J mixgeo
#SBATCH -o logs/mixgeo_%j.out
# The multiplicative seventeen-parameter mixture on one H200, the same pipeline and settings as
# perlmutter/train_mixgeo_ens.sbatch (train, reference bundle, closure, wifi basis). TAG, DATA,
# STEPS, K and ENS come from the environment:
#   sbatch --export=ALL,TAG=MIXGEOV3S3,DATA=data_stagePURE17_v3,STEPS=108000,ENS=4 train_mixgeo.sbatch
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1
[ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
ENS=${ENS:-4}; TAG=${TAG:?}; DATA=${DATA:?}
# CHAIN=1 saves the whole training state at every checkpoint (LADDER_RESUME) and continues from it,
# so a budget longer than the 1-day limit runs as links submitted with --dependency=afterany. A link
# that finds the trained model skips to the steps after training, one that finds .done exits.
if [ "${CHAIN:-0}" = 1 ]; then
  export LADDER_RESUME=output/models/${TAG}_resume.pt
  [ -f output/models/${TAG}.done ] && { echo "$TAG already done"; exit 0; }
fi
# GPU_FETCH=1: standardize each batch on the GPU (bitwise equal to the CPU path, faster)
export LADDER_GPU_FETCH=${GPU_FETCH:-0}
export DM_DATA=$DATA DM_TAG=$TAG LADDER_ENS=$ENS LADDER_EMB_STORE=a
export LADDER_HEAD_KIND=geometric LADDER_TARGET_PURE=1 LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1
export LADDER_K=${K:-48} LADDER_STEPS=${STEPS:-36000} LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_DEAD=report LADDER_REVIVE=1
# a fresh memory-map directory per job: node-local /tmp by default (about 100 GB free on the H200
# nodes), MMAP_BASE=$R/mmap on beegfs for a set too large for it (the 720-run set maps 86 GB);
# removed on exit whatever happens
MMAP_BASE=${MMAP_BASE:-/tmp}; mkdir -p $MMAP_BASE
# working memory maps in RAM (MMAP_BASE=/dev/shm) outlive a job that is killed without its EXIT trap:
# remove those of our own jobs that are no longer running before adding one (skipped if squeue fails)
if [ "$MMAP_BASE" = /dev/shm ] && LIVE=" $(squeue -h -u $USER -o %A 2>/dev/null | tr '\n' ' ') " && [ "$LIVE" != "  " ]; then
  for d in /dev/shm/${USER}_mmap_*; do [ -e "$d" ] || continue; j=${d##*_}; case "$LIVE" in *" $j "*) ;; *) rm -rf "$d" ;; esac; done
fi
export LADDER_MMAP_DIR=$MMAP_BASE/${USER}_mmap_${TAG}_$SLURM_JOB_ID STAGED_OUT=output/stage_mixture_${TAG}.json
trap 'rm -rf $LADDER_MMAP_DIR' EXIT
echo "=== $TAG on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES:-0}), data $DATA, $ENS members, ${LADDER_STEPS} steps, mmap $LADDER_MMAP_DIR $(date)"
if [ "${CHAIN:-0}" = 1 ] && [ -f $STAGED_OUT ] && [ -f output/models/${TAG}_trunk.pt ] && [ output/models/${TAG}_trunk.pt -nt $LADDER_RESUME ]; then
  echo "=== $TAG trained in an earlier link, continuing after training"
else
  DM_DRYRUN=1 python3 -u stage_mixture.py || { echo "DRYRUN FAILED"; exit 2; }
  python3 -u stage_mixture.py || { echo "TRAIN FAILED"; rm -rf $LADDER_MMAP_DIR; exit 3; }
fi
rm -rf $LADDER_MMAP_DIR
echo "=== reference bundle $(date)"
python3 -u rebuild_ref_lowmem.py $TAG || echo "REF REBUILD FAILED"
echo "=== closure $(date)"
python3 -u ladder_widths_big.py MIX output/models/${TAG}_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "WIDTHS FAILED"
echo "=== wifi basis $(date)"
python3 -u wifi_embed.py MIX output/models/${TAG}_cond.npz output/models/${TAG}_trunk.pt || echo "EMBED FAILED"
[ "${CHAIN:-0}" = 1 ] && touch output/models/${TAG}.done
echo "$TAG DONE $(date)"
