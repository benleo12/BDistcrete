#!/bin/bash
# one production rank in prod4/rank_N with its own read-only grid copy and NEW seed block:
# seed index = 504 + SLURM_PROCID (the patched binary accepts 0..699; prod2 used 0..503), so
# these 196 ranks are statistically independent of prod2 and combine with it 1:1.
P=${SLURM_PROCID:-0}; S=$((P + 504))
R=$SCRATCH/eerad3_thrust
d=$R/prod4/rank_$S
mkdir -p "$d" && cp $R/grids_2sweep/E.*.T "$d"/ && cd "$d" || exit 1
exec $R/eerad3-1.0-blk/eerad3 -i ../eerad3.input -n $S > ../prod_$S.log 2>&1
