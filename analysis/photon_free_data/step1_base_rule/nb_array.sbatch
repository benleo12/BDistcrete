#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb_obs
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=170G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseA/observables/logs/nb_%A_%a.out
# Photon-free 1_minus_thrust, B_total, rho_heavy (decision D4), one array element per share of
# units.tsv. Restart-safe: finished units are skipped, so a requeue after preemption resumes.
#   sbatch --array=0-15 nb_array.sbatch      (NTASK must match plan.py)
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseA/observables
NTASK=${NTASK:-16}; NPROC=${NPROC:-78}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /mnt/beegfs/bassi/reweighting/maxent_code || exit 1
echo "=== start task $SLURM_ARRAY_TASK_ID job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date)"
../env/torch/bin/python -u $W/scripts/nb_units.py $W $SLURM_ARRAY_TASK_ID $NTASK $NPROC
rc=$?
echo "=== end task $SLURM_ARRAY_TASK_ID rc $rc $(date)"
exit $rc
