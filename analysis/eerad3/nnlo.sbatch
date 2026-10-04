#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 06:00:00
#SBATCH -J eerad3_nnlo
#SBATCH -o %x-%j.out
# EERAD3 NNLO thrust, correct three-phase form. The VEGAS grid filename drops the seed
# digits by construction, so every rank shares one grid: ONE warmup writes it, then the
# production ranks only read it. Running warmup and production together in parallel ranks
# makes them clobber each other and risks a torn read.
set -uo pipefail
NPROD=256
ROOT=$SCRATCH/eerad3_thrust
SRC=$ROOT/eerad3-1.0
mkdir -p "$ROOT/work" && cd "$ROOT/work"

write_card () {   # $1=iwarm $2=iprod
  printf '%s\n' \
    '1d-8      ! y0' \
    '4         ! iaver (1-T)' \
    '1d-5      ! cutvar' \
    '1         ! imom' \
    '1         ! iang' \
    '-2        ! nloop (NNLO only)' \
    '0         ! icol' \
    'M         ! itag' \
    "$1 $2     ! iwarm iprod" \
    '5 5       ! itmax1 itmax2' \
    '20000000 300000 500000  ! nshot3 nshot4 nshot5' > eerad3.input
}

echo "phase 1: single warmup builds the shared VEGAS grids"
write_card 1 0
srun -n 1 "$SRC/eerad3" -i eerad3.input > warmup.log 2>&1 || { echo "WARMUP FAILED"; tail -20 warmup.log; exit 1; }
ls -la E.*v3a.T E.*v4a.T E.*v5a.T E.*v5b.T || { echo "NO GRIDS WRITTEN"; exit 1; }

echo "phase 2: $NPROD production ranks, read-only on the grids"
write_card 0 1
srun -n "$NPROD" --cpu-bind=cores "$SRC/../prod_rank.sh"

echo "phase 3: combine"
printf '%s\n' '4' 'y1d8' 'iM0' 'tx' "0 $((NPROD-1))" '0' > eerad3_combine.input
"$SRC/eerad3_combine" -i eerad3_combine.input | tee combine.log
echo EERAD3_NNLO_DONE
