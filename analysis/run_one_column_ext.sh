#!/bin/bash
# One alpha_0 column of a direct profile on the widened theory grid (alpha_s 0.100-0.136, alpha_0 0.15-0.85).
# usage: run_one_column_ext.sh TAG OUTTAG A0 [EXPORT REF]
cd .
source ~/miniconda3/etc/profile.d/conda.sh; conda activate env_ba
tag=$1; outtag=$2; a0=$3
ONLY_A0=$a0 TARGETS_GRID=output/thrust_targets_grid_ext.npz NSCAN=${NSCAN:-48} AE_DTYPE=${AE_DTYPE:-float64} TORCH_THREADS=${TORCH_THREADS:-2} \
  EXPORT_PATH=${4:-output/models/${tag}_cond.npz} REF_PATH=${5:-output/models/${tag}_ref_v2.npz} \
  PROFILE_OUT=output/profile_${outtag}_central_a0${a0}.json \
  python -u profile_column.py $tag > logs/profile_column_${outtag}_a0${a0}.log 2>&1
