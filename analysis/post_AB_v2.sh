#!/usr/bin/env bash
# After gen_stageA_v2.sh and gen_stageB_v2.sh: rebuild the ladder A and B reference bundles
# in training order from the corrected shapes, then the tab:ladder widths on v1 and v2 data.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_ba
export OMP_NUM_THREADS=2
cd .
for S in ${STAGES_TO_DO:-A B}; do
  D1=data_stage${S}_full; D2=data_stage${S}_v2
  python rebuild_ref_obs.py $S --shapes-dir $D2 --anchor-dir $D2 --nref-per 12000 --out output/models/${S}_ref_v2.npz 2>&1 | grep -v "^\[stage"
  echo "=== $S widths, v1:";  env STAGE${S}_DATA=$D1 python ladder_widths_v2.py $S output/models/${S}_ref.npz    2>&1 | grep -v "^\[stage"
  echo "=== $S widths, v2:";  env STAGE${S}_DATA=$D2 python ladder_widths_v2.py $S output/models/${S}_ref_v2.npz 2>&1 | grep -v "^\[stage"
done
echo "POST AB DONE"
