#!/bin/bash
# One alpha_0 column of the MIX17 direct profile, seeded from the alpha_0 = 0.43 column.
cd .
source ~/miniconda3/etc/profile.d/conda.sh; conda activate env_ba
a0=$1
ONLY_A0=$a0 STARTS_FROM=output/profile_MIX17aug_central_a00.43.json AE_DTYPE=float32 TORCH_THREADS=3 \
  EXPORT_PATH=output/models/MIX17aug_cond.npz REF_PATH=output/models/MIX17aug_ref_v2_slim.npz \
  python -u profile_column.py MIX17aug > logs/profile_column_MIX17aug_a0$a0.log 2>&1
