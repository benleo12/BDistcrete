#!/bin/bash
# one production rank in prod3/rank_N for the y0=1e-7 cutoff variation (same seed block as prod2).
# EERAD3 names its VEGAS grid files after y0 (E.y1d7.*.T) but the two-sweep warm-up was made at
# y0=1e-8. A VEGAS grid is only an importance-sampling map: any grid gives an unbiased estimate and
# only the variance depends on it, so the y1d8 grids are aliased under the y1d7 names. The first
# prod3 attempt (job 57570767) hit 'End of file' on the missing y1d7 grid in every rank after 14 s.
P=${SLURM_PROCID:-0}
R=$SCRATCH/eerad3_thrust
d=$R/prod3/rank_$P
mkdir -p "$d" && cp $R/grids_2sweep/E.*.T "$d"/ && cd "$d" || exit 1
for f in E.y1d8.*.T; do cp "$f" "${f/y1d8/y1d7}"; done
exec $R/eerad3-1.0-blk/eerad3 -i ../eerad3.input -n $P > ../prod_$P.log 2>&1
