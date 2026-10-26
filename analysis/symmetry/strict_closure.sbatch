#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 12:00:00
#SBATCH -J strict
#SBATCH -o logs/strict_%j.out
# The strict closure test: a trained mixture scored at all 48 held-out points with 400k events each
# (the run plus its four extensions, data_stagePURE17_v3_h400), through ladder_widths_big.py.
# The export is linked as <TAG>h400 so the widths land in output/widths_v2_<TAG>h400.json.
#   sbatch --export=ALL,TAGS="MIXGEOV3S9 MIXGEOV3E20S9" strict_closure.sbatch
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=16 DM_DATA=data_stagePURE17_v3_h400 LADDER_HEAD_KIND=geometric LADDER_MIX_SPLIT=8,8
for TAG in ${TAGS:?}; do
  ln -sf ${TAG}_cond.npz output/models/${TAG}h400_cond.npz
  echo "=== $TAG strict closure $(date)"
  python3 -u ladder_widths_big.py MIX output/models/${TAG}h400_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "WIDTHS FAILED $TAG"
done
echo "STRICT DONE $(date)"
