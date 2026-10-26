#!/bin/bash
#SBATCH -p preq
#SBATCH -t 12:00:00
#SBATCH -J rescorev4
#SBATCH -o logs/rescore_v4_%j.out
# The two 2768-run mixtures of the scan of the number of training runs (not in the paper) scored
# again against a reference of the size of the 720-run model's (415 of the 1000 stored events of
# each run, 1 148 720 events, against 1 149 840), on the standard test (data_stagePURE17_v4, the 48
# held-out points at 8e4 events, as at training) and the strict test (data_stagePURE17_v3_h400), so
# that every model of the scan has the same reference size. Writes output/widths_v2_<TAG>r415.json and output/widths_v2_<TAG>h400r415.json.
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=40 LADDER_HEAD_KIND=geometric LADDER_MIX_SPLIT=8,8 REF_PER_RUN=415 WIDTHS_SUFFIX=r415
for TAG in MIXGEOV4S9 MIXGEOV4L; do
  echo "=== $TAG standard test, equal reference $(date)"
  DM_DATA=data_stagePURE17_v4 python3 -u ladder_widths_big.py MIX output/models/${TAG}_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "WIDTHS FAILED $TAG std"
  ln -sf ${TAG}_cond.npz output/models/${TAG}h400_cond.npz
  echo "=== $TAG strict test, equal reference $(date)"
  DM_DATA=data_stagePURE17_v3_h400 python3 -u ladder_widths_big.py MIX output/models/${TAG}h400_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "WIDTHS FAILED $TAG strict"
done
echo "RESCORE DONE $(date)"
