#!/bin/bash
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -J gen_v3
#SBATCH -o gen/logs/batch_%A_%a.out
# Generate the runs listed in a task file, one array element per node, PAR runs at a time.
# Each node mounts CVMFS once (cvmfsexec), enters the EL9 container once, and keeps the HepMC
# on its local disk; only the particle and shape files go to $GEN_DATA on beegfs.
#   sbatch --array=0-3 --export=ALL,TASKS=gen/tasks/v3.txt,GEN_DATA=gen/data_v3 gen/gen_batch.sbatch
# Extensions (more events at the same points): add GEN_SEED_OFFSET=<k>000000 and GEN_DATA=gen/data_ext/k<k>.
# Submit from R, the parent of the analysis folder MAXENT_DIR, which holds gen/ (with gen/logs/),
# tools/cvmfsexec/ and containers/almalinux9.sif. The log, TASKS and GEN_DATA are relative to R.
set -eo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd $R
TASKS=$R/${TASKS:?}; export GEN_DATA=$R/${GEN_DATA:?}; PAR=${PAR:-40}
# NSHARD fixes the split when only some array elements are resubmitted (e.g. --array=0,1,3)
N=${NSHARD:-${SLURM_ARRAY_TASK_COUNT:-1}}; I=${SLURM_ARRAY_TASK_ID:-0}
export GEN_LOCAL=/local1/${USER}_gen_${SLURM_JOB_ID}
mkdir -p "$GEN_LOCAL" "$GEN_DATA" $R/gen/logs
grep -v '^#' "$TASKS" | awk -v n=$N -v i=$I 'NF && (NR-1) % n == i' > "$GEN_LOCAL/tasks"
echo "$(date) $(hostname): $(wc -l < $GEN_LOCAL/tasks) runs, $PAR at a time, into $GEN_DATA"
# cvmfsexec keeps its mount tree inside its own folder, so every node runs a private copy
rsync -a --exclude mnt $R/tools/cvmfsexec/ $GEN_LOCAL/cvmfsexec/
source /etc/profile.d/modules.sh; module load apptainer
$GEN_LOCAL/cvmfsexec/cvmfsexec sft.cern.ch -- apptainer exec --cleanenv -B /cvmfs -B $R -B /local1 \
  --env GEN_DATA=$GEN_DATA,GEN_LOCAL=$GEN_LOCAL,GEN_SEED_OFFSET=${GEN_SEED_OFFSET:-0} $R/containers/almalinux9.sif \
  bash -c "cd $GEN_LOCAL && xargs -P $PAR -L 1 bash $R/gen/gen_one.sh < tasks" | grep -v '^CernVM-FS' || true
rm -rf "$GEN_LOCAL"
echo "$(date) done: $(grep -c '' $R/gen/logs/batch_${SLURM_ARRAY_JOB_ID}_${I}.out) lines of log"
