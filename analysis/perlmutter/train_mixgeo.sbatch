#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C gpu
#SBATCH -q interactive
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 03:59:00
#SBATCH -J mixgeo
#SBATCH -o logs/mixgeo_%j.out
#SBATCH -e logs/mixgeo_%j.err
# The multiplicative (geometric) seventeen-parameter mixture, trained on the pure runs of both
# generators with the wifi protocol: four bootstrap members, a fifth of each run held out as the
# fit set of the wifi weights (patch_wifi_geo.py). Then the reference bundle on the exact shapes,
# the closure widths at the held-out pure points, and the wifi basis on the fit set and the
# held-out runs (wifi_embed.py), which wifi_fit.py turns into weights and a covariance.
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export OMP_NUM_THREADS=32
python3 -u make_pure_dataset.py || { echo "DATASET FAILED"; exit 1; }
export DM_DATA=data_stagePURE17 DM_TAG=MIXGEO LADDER_HEAD_KIND=geometric LADDER_MIX_SPLIT=8,8
export LADDER_TARGET_PURE=1 LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1
export LADDER_K=48 LADDER_STEPS=36000 LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_DEAD=report LADDER_REVIVE=1
export LADDER_MMAP_DIR=$SCRATCH/mmap_mixgeo_$SLURM_JOB_ID STAGED_OUT=output/stage_mixture_MIXGEO.json
DM_DRYRUN=1 python3 -u stage_mixture.py || { echo "DRYRUN FAILED"; exit 2; }
echo "=== train $(date)"
python3 -u stage_mixture.py || { echo "TRAIN FAILED"; rm -rf $LADDER_MMAP_DIR; exit 3; }
rm -rf $LADDER_MMAP_DIR
echo "=== reference bundle $(date)"
python3 -u rebuild_ref_lowmem.py MIXGEO || echo "REF REBUILD FAILED"
echo "=== closure $(date)"
python3 -u ladder_widths_big.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_ref_v2_slim.npz || echo "WIDTHS FAILED"
echo "=== wifi basis $(date)"
python3 -u wifi_embed.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_trunk.pt || echo "EMBED FAILED"
echo "MIXGEO DONE $(date)"
