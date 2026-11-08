#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb_work
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=170G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseA/observables/logs/work_%j.out
# Dynamic workers (nb_worker.py): one unit per process, claims in claims/, NB_TASKS and NB_MAXN
# restrict the units. A requeue takes back its own claims.
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseA/observables
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NB_TLIMIT=3600
cd /mnt/beegfs/bassi/reweighting/maxent_code || exit 1
echo "=== start work job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date) NB_TASKS=${NB_TASKS:-all} NB_MAXN=${NB_MAXN:-}"
../env/torch/bin/python -u $W/scripts/nb_worker.py $W 78 ""
rc=$?; echo "=== end work job rc $rc $(date)"; exit $rc
