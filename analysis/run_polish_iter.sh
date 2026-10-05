#!/bin/bash
# Neighbour polish of a merged direct profile, repeated until no node improves by more than TOL.
# usage: run_polish_iter.sh TAG [NPART] [MAXPASS]      (TARGETS_GRID, DCHI_MAX, NEIGH, AE_DTYPE pass through)
cd .
source ~/miniconda3/etc/profile.d/conda.sh; conda activate env_ba
tag=$1; npart=${2:-4}; maxpass=${3:-4}; tol=${TOL:-0.01}
for pass in $(seq 1 $maxpass); do
  rm -f output/polish_${tag}_central_part*.json
  from=1; [ $pass -eq 1 ] && [ "${FIRST_FROM_COLUMNS:-1}" = "1" ] && from=0
  for p in $(seq 0 $((npart-1))); do
    FROM_MERGED=$from PART=$p NPART=$npart TORCH_THREADS=${TORCH_THREADS:-3} \
      python -u polish_parallel.py part $tag central > logs/polish_${tag}_pass${pass}_part$p.log 2>&1 &
  done
  wait
  FROM_MERGED=$from python -u polish_parallel.py merge $tag central > logs/polish_${tag}_pass${pass}_merge.log 2>&1
  cat logs/polish_${tag}_pass${pass}_merge.log
  worst=$(grep "nodes improved" logs/polish_${tag}_pass${pass}_merge.log | sed 's/.*largest improvement //')
  echo "pass $pass: largest improvement $worst"
  python -c "import sys; sys.exit(0 if float('$worst') < $tol else 1)" && { echo "CONVERGED after pass $pass"; break; }
done
echo "POLISH DONE $tag"
