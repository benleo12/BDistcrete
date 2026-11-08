#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb2
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=170G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseB/observables/logs/nb2_%A_%a.out
# Phase B photon-free CSVs (beam_photons.py mask): dynamic workers, every element takes the next
# unclaimed unit of units.tsv until none is left or its time limit is near. Restart-safe (claims,
# pending records), so a requeue after preemption resumes and other elements can join any time.
#   sbatch --array=0-11 nb2_array.sbatch   (preq gives whole nodes: 78 processes each)
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables
NPROC=${NPROC:-78}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NB_TLIMIT=${NB_TLIMIT:-7200}
cd $W || exit 1
echo "=== start element ${SLURM_ARRAY_TASK_ID:-x} job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date)"
/mnt/beegfs/bassi/reweighting/env/torch/bin/python -B -u $W/scripts/nb2_units.py $W $NPROC run
rc=$?
echo "=== end element ${SLURM_ARRAY_TASK_ID:-x} rc $rc $(date)"
exit $rc
