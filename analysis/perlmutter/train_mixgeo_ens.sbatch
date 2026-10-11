#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J mixgeo_ens
#SBATCH -o logs/mixgeo_ens_%j.out
#SBATCH -e logs/mixgeo_ens_%j.err
# The multiplicative seventeen-parameter mixture with ENS ensemble members (the wifi basis has
# ENS^2 cross terms plus the constant), trained on the pure runs with the wifi protocol, then the
# reference bundle, the closure at the held-out pure points and the wifi basis. Submit with
#   sbatch --export=ALL,ENS=8 -t 03:00:00 perlmutter/train_mixgeo_ens.sbatch
# Members train one after another, so allow about 40 minutes per four members plus half an hour.
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export OMP_NUM_THREADS=32
ENS=${ENS:-8}
TAG=MIXGEO${ENS}
[ -f data_stagePURE17/meta.json ] || { python3 -u make_pure_dataset.py || { echo "DATASET FAILED"; exit 1; }; }
export DM_DATA=data_stagePURE17 DM_TAG=$TAG LADDER_ENS=$ENS LADDER_EMB_STORE=a
export LADDER_HEAD_KIND=geometric LADDER_TARGET_PURE=1 LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1
export LADDER_K=48 LADDER_STEPS=36000 LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_DEAD=report LADDER_REVIVE=1
export LADDER_MMAP_DIR=$SCRATCH/mmap_${TAG}_$SLURM_JOB_ID STAGED_OUT=output/stage_mixture_${TAG}.json
DM_DRYRUN=1 python3 -u stage_mixture.py || { echo "DRYRUN FAILED"; exit 2; }
echo "=== train $TAG with $ENS members $(date)"
python3 -u stage_mixture.py || { echo "TRAIN FAILED"; rm -rf $LADDER_MMAP_DIR; exit 3; }
rm -rf $LADDER_MMAP_DIR
echo "=== reference bundle $(date)"
python3 -u rebuild_ref_lowmem.py $TAG || echo "REF REBUILD FAILED"
echo "=== closure $(date)"
python3 -u ladder_widths_big.py MIX output/models/${TAG}_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "WIDTHS FAILED"
echo "=== wifi basis $(date)"
python3 -u wifi_embed.py MIX output/models/${TAG}_cond.npz output/models/${TAG}_trunk.pt || echo "EMBED FAILED"
echo "$TAG DONE $(date)"
