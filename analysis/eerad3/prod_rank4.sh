#!/bin/bash
# one production rank in prod2/rank_N with its own read-only grid copy and its own seed block
P=${SLURM_PROCID:-0}
R=$SCRATCH/eerad3_thrust
d=$R/prod2/rank_$P
mkdir -p "$d" && cp $R/grids_2sweep/E.*.T "$d"/ && cd "$d" || exit 1
exec $R/eerad3-1.0-blk/eerad3 -i ../eerad3.input -n $P > ../prod_$P.log 2>&1
