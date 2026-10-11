#!/bin/bash
# The multiplicative seventeen-parameter mixture trained on the laptop GPU (MPS), the same recipe
# as perlmutter/train_mixgeo.sbatch: pure runs of both generators, wifi protocol (four bootstrap
# members, a fifth of each run held out as the fit set), then the reference bundle on the exact
# shapes, the closure at the held-out pure points and the wifi basis. One training job at a time.
#   nohup caffeinate -i ./run_mixgeo_local.sh > logs/mixgeo_local.log 2>&1 &
set -uo pipefail
cd .
source ~/miniconda3/etc/profile.d/conda.sh; conda activate env_ba
export DM_DATA=data_stagePURE17 DM_TAG=MIXGEO LADDER_HEAD_KIND=geometric LADDER_MIX_SPLIT=8,8
export LADDER_TARGET_PURE=1 LADDER_FIT_FRAC=0.2 LADDER_BOOTSTRAP=1
export LADDER_K=48 LADDER_STEPS=36000 LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_DEAD=report LADDER_REVIVE=1
export LADDER_MMAP_DIR=mmap_mixgeo_local STAGED_OUT=output/stage_mixture_MIXGEO.json
rm -rf $LADDER_MMAP_DIR
DM_DRYRUN=1 python -u stage_mixture.py || { echo "DRYRUN FAILED"; exit 2; }
echo "=== train $(date)"
python -u stage_mixture.py || { echo "TRAIN FAILED"; exit 3; }
rm -rf $LADDER_MMAP_DIR
echo "=== reference bundle $(date)"
python -u rebuild_ref_lowmem.py MIXGEO || echo "REF REBUILD FAILED"
echo "=== closure $(date)"
python -u ladder_widths_big.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_ref_v2_slim.npz || echo "WIDTHS FAILED"
echo "=== wifi basis $(date)"
python -u wifi_embed.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_trunk.pt || echo "EMBED FAILED"
echo "MIXGEO LOCAL DONE $(date)"
