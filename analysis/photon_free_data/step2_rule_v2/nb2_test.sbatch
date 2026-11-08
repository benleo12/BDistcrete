#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb2_test
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:45:00
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseB/observables/logs/test_%j.out
# Test mode of nb2_units.py: the units in UIDS, outputs in WORK/test only (no _nb file is touched).
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd $W || exit 1
echo "=== test on $(hostname) $(date) UIDS=${UIDS:?}"
/usr/bin/time -v /mnt/beegfs/bassi/reweighting/env/torch/bin/python -B -u $W/scripts/nb2_units.py $W 4 test "$UIDS" 2>&1 | grep -v "^\s*\(Page\|Volunt\|Invol\|Swaps\|File sys\|Socket\|Signals\|Average\|Exit\|Minor\|Major\)"
echo "=== end $(date)"
