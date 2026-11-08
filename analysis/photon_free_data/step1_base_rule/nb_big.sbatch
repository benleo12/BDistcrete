#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb_big
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=170G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseA/observables/logs/big_%j.out
# Units of the given tasks (TASKS, comma list or all), one at a time with the events spread over
# 78 processes (nb_bigunit.py), so a preemption loses seconds. NB_MINN: only units this large.
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseA/observables
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/beegfs/bassi/reweighting/maxent_code || exit 1
echo "=== start big job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date) TASKS=${TASKS:?} NB_MINN=${NB_MINN:-0}"
../env/torch/bin/python -u $W/scripts/nb_bigunit.py $W 78 $TASKS
rc=$?; echo "=== end big job rc $rc $(date)"; exit $rc
