#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 8
#SBATCH --ntasks-per-node=63
#SBATCH -t 06:00:00
#SBATCH -J eerad3_nnlo4_y7
#SBATCH -o %x-%j.out
# EERAD3 NNLO thrust, CUTOFF VARIATION y0=1e-7, second attempt with the y1d8 grids aliased as y1d7 (see prod_rank6.sh) (same seeds as prod2 -> paired cutoff test). The first attempt (nshot5=500000, itmax2=2) did not finish a
# single rank in 6h. Calibration: nshot5=20000 itmax2=1 = 17.5 min/rank, and the 5-parton v5
# channel dominates cost linearly in nshot5. Here nshot5=50000 with itmax2=2 (needed for
# per-bin VEGAS errors; itmax2=1 gives NaN errors) is ~10x cheaper on v5 than the failed run,
# so ~1.5-2.5 h/rank, comfortably inside 6h. 504 ranks (blocks 0-4 + 4) on 8 nodes, 63 tasks
# per node = ~2 cores/task (less contention than the failed 125/node). Read-only grids from the
# two-sweep warmup, each rank in its own dir with its own seed block via the patched -n 0..699.
set -uo pipefail
R=$SCRATCH/eerad3_thrust
mkdir -p $R/prod3 && cd $R/prod3
printf '%s\n' \
  '1d-7      ! y0' '4 ! iaver (1-T)' '1d-5 ! cutvar' '1 ! imom' '1 ! iang' \
  '-2 ! nloop (NNLO only)' '0 ! icol' 'M ! itag' '0 1 ! iwarm iprod' \
  '1 2 ! itmax1 itmax2' '1000000 100000 50000 ! nshot3 nshot4 nshot5' > eerad3.input
date; echo "504 production ranks, seeds 0..503, nshot5=50000 itmax2=2"
srun -n 504 --cpu-bind=cores $R/perl2/prod_rank6.sh
date; echo "completed ranks: $(grep -l 'only \[NNLO\]' prod_*.log 2>/dev/null | wc -l)/504"
echo EERAD3_PROD4_DONE
