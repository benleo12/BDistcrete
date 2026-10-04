#!/bin/bash
# EERAD3 NNLO thrust on Perlmutter, in the CORRECT two-phase form.
#
# The VEGAS grid filename that EERAD3 builds deliberately drops the seed digits
# (gridfile = 'E'//fname(4:13)//char//ctype), so every rank would read and write the SAME
# grid. The intended workflow is therefore: ONE warmup job that builds the grids, then many
# production jobs with different -n seeds that only READ them, then eerad3_combine.
# Running warmup and production together in parallel ranks (iwarm=1 iprod=1) makes the ranks
# clobber each other's grid files and can produce a torn read. That is the mistake this
# script exists to avoid.
#
# Usage on a login node:
#   salloc -N 2 -C cpu -q interactive -t 04:00:00 -A <account>
#   ./run_eerad3_nnlo.sh 256          # 256 production ranks
set -euo pipefail
NPROD=${1:-128}
ROOT=$SCRATCH/eerad3_thrust
SRC=$ROOT/eerad3-1.0
mkdir -p $ROOT && cd $ROOT

if [ ! -x $SRC/eerad3 ]; then
  echo "== building EERAD3 =="
  [ -f eerad3-1.0.tar.gz ] || curl -sL -o eerad3-1.0.tar.gz \
      https://eerad3.hepforge.org/downloads/eerad3-1.0.tar.gz
  tar xzf eerad3-1.0.tar.gz
  cd $SRC && mkdir -p obj
  module load PrgEnv-gnu 2>/dev/null || true
  make FC=gfortran FFLAGS="-fno-automatic -O2" eerad3 eerad3_combine
  cd $ROOT
fi

card () {   # $1=nloop $2=iwarm $3=iprod $4=nshot3 $5=nshot4 $6=nshot5
cat <<CARD
1d-8      ! y0
4         ! iaver  (1-T)
1d-5      ! cutvar
1         ! imom
1         ! iang
$1        ! nloop
0         ! icol
M         ! itag
$2 $3     ! iwarm iprod
5 5       ! itmax1 itmax2
$4 $5 $6  ! nshot3 nshot4 nshot5
CARD
}

echo "== phase 1: single warmup, builds the shared VEGAS grids =="
mkdir -p $ROOT/work && cd $ROOT/work
card -2 1 0 20000000 300000 500000 > eerad3.input
srun -n 1 $SRC/eerad3 -i eerad3.input > warmup.log 2>&1
ls -la E.*.v3a.T E.*.v4a.T E.*.v5a.T E.*.v5b.T

echo "== phase 2: $NPROD production ranks, read-only on the grids =="
card -2 0 1 20000000 300000 500000 > eerad3.input
srun -n $NPROD --cpu-bind=cores bash -c \
  '$0/eerad3 -i eerad3.input -n $SLURM_PROCID > prod_$SLURM_PROCID.log 2>&1' $SRC

echo "== phase 3: combine =="
cat > eerad3_combine.input <<CMB
4         ! iaver
y1d8      ! frooty
iM0       ! frooti
tx        ! filetag
0 $((NPROD-1))  ! minfile maxfile
0         ! nvoid
CMB
$SRC/eerad3_combine -i eerad3_combine.input | tee combine.log
echo "DONE: results in $ROOT/work"
