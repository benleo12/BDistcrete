#!/bin/bash
# Extend each MIX17 alpha_0 column upward in the coupling on the widened theory grid.
cd .
source ~/miniconda3/etc/profile.d/conda.sh; conda activate env_ba
a0=$1
EXTEND_FROM=output/profile_MIX17aug_central_a0$a0.json EXTEND_UP_ONLY=1 ONLY_A0=$a0 \
  TARGETS_GRID=output/thrust_targets_grid_ext.npz AE_DTYPE=float32 TORCH_THREADS=3 \
  EXPORT_PATH=output/models/MIX17aug_cond.npz REF_PATH=output/models/MIX17aug_ref_v2_slim.npz \
  PROFILE_OUT=output/profile_MIX17ext_central_a0$a0.json \
  python -u profile_column.py MIX17aug > logs/profile_column_MIX17ext_a0$a0.log 2>&1
