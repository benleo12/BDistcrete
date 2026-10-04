#!/bin/bash
# profile_rows.py on this machine: one pass and two polish passes, with NP processes.
# usage: run_rows_local.sh FIT [NP]     (COLS, COL_STEP, ROW_STEP, AE_DTYPE pass through)
cd .
source ~/miniconda3/etc/profile.d/conda.sh; conda activate env_ba
FIT=$1; NP=${2:-4}; S=${FIT%.json}; tag=$(basename $S)
export TARGETS_GRID=output/thrust_targets_grid_ext.npz TORCH_THREADS=${TORCH_THREADS:-3}
run() {
  rm -f ${S}_rows${ROWS_TAG:-}_part*.json
  for p in $(seq 0 $((NP-1))); do PART=$p NPART=$NP python -u profile_rows.py part $FIT > logs/rows_${tag}_$1_$p.log 2>&1 & done; wait
  python -u profile_rows.py merge $FIT
}
run pass0
for p in 1 2; do export POLISH_FROM=${S}_rows${ROWS_TAG:-}.json; run pol$p; unset POLISH_FROM; done
echo "ROWS LOCAL DONE $tag"
