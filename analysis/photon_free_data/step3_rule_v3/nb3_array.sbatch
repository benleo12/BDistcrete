#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb3
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=170G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseB/observables_v3/logs/nb3_%A_%a.out
# Beam-photon rule v3 in the _nb CSVs (nb3_units.py run): dynamic workers, every element takes the next
# unclaimed unit of units.tsv until none is left or its time limit is near; only the CSVs with a changed
# row are rewritten (temp file, then rename), and only directly inside the folders of NB_FOLDERS.txt.
# Restart-safe (claims, pending records), so a requeue after preemption resumes and elements can join at
# any time. It does not start once the rerun has (RUN/output/.lmx_rerun_start exists): the rerun reads
# these CSVs.
#   sbatch --array=0-1 nb3_array.sbatch   (preq gives whole nodes: 78 processes each)
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables_v3
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
NPROC=${NPROC:-78}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NB_TLIMIT=${NB_TLIMIT:-7200}
cd $W || exit 1
echo "=== start element ${SLURM_ARRAY_TASK_ID:-x} job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $(date)"
if [ -e $RUN/output/.lmx_rerun_start ]; then
  echo "RERUN STARTED ($RUN/output/.lmx_rerun_start exists): the _nb CSVs are not rewritten under it. Ask first."
  exit 8
fi
/mnt/beegfs/bassi/reweighting/env/torch/bin/python -B -u $W/scripts/nb3_units.py $W $NPROC run
rc=$?
echo "=== end element ${SLURM_ARRAY_TASK_ID:-x} rc $rc $(date)"
exit $rc
