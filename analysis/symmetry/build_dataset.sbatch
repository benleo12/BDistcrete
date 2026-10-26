#!/bin/bash
#SBATCH -p sharedq
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -J build
#SBATCH -o logs/build_%j.out
# Build training sets with make_dataset_v3.py after a generation batch: one per entry of SPECS,
# each "DST:SETS:TRAIN_EVENTS:HELD_EXT[:TRAIN_EXT]".
#   sbatch --dependency=afterok:<gen> --export=ALL,SPECS="data_stagePURE17_v4:release,v3,v4:0:0" build_dataset.sbatch
set -euo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
for spec in ${SPECS:?}; do
  IFS=: read -r DST SETS NEV HEXT TEXT <<< "$spec"
  V3_SETS=$SETS V3_TRAIN_EVENTS=$NEV V3_HELD_EXT=$HEXT V3_TRAIN_EXT=${TEXT:-0} V3_DST=$MAXENT_DIR/$DST python -u make_dataset_v3.py
done
echo "BUILD DONE $(date)"
