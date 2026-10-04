#!/bin/bash
# One rank per array task. d=8, so the score expansion predicts a closure cliff at K=d+1=9
# and exhaustion at K2 = 1 + d + d(d+1)/2 = 45; the list brackets both.
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J rank
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/rank_%x_%A_%a.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/rank_%x_%A_%a.err
set -e
STAGE=${STAGE:?set STAGE=E or F}
KLIST=${KLIST:?set KLIST as a comma separated list, one entry per array index}
cd $SCRATCH/maxent_code; mkdir -p logs output
# Split with bash rather than cut. `cut -d, -fN` prints the WHOLE line when the line holds no
# delimiter, so a list that arrived truncated to one entry silently handed every array task the
# same rank instead of failing. Note also that KLIST must NOT be passed inside --export, whose
# own argument is comma separated: export it in the submitting shell and use --export=ALL.
IFS=',' read -ra KS <<< "$KLIST"
echo "KLIST parsed into ${#KS[@]} ranks: ${KS[*]}"
[ "${#KS[@]}" -gt "$SLURM_ARRAY_TASK_ID" ] || {
  echo "BAD array index $SLURM_ARRAY_TASK_ID against a ${#KS[@]} entry list: $KLIST"; exit 2; }
K="${KS[$SLURM_ARRAY_TASK_ID]}"
case "$K" in ''|*[!0-9]*) echo "BAD rank '$K' at index $SLURM_ARRAY_TASK_ID"; exit 2;; esac
module load pytorch/2.6.0
# The same mandatory settings as the ladder: without these the run is not the published recipe.
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export LADDER_LOWMEM=1 LADDER_MMAP=1
# A fresh cache per TASK. SLURM_JOB_ID is distinct for every array task; reusing one dir has
# hung Lustre before, and two tasks sharing one would race on the same float16 files.
export LADDER_MMAP_DIR=$SCRATCH/mmap_rank_${SLURM_JOB_ID}
export RANK_STAGES=$STAGE RANK_KS=$K
RANK_DIR=${RANK_DIR:-output}
mkdir -p "$RANK_DIR"
export RANK_DIR
export RANK_OUT=$RANK_DIR/rank_scan_${STAGE}_K${K}.json
# RANK_ENS is the seed count. The published A, B and C scans used FOUR seeds, so a scan meant
# to share their figure must too, and the merger records whichever was used.
export RANK_ENS=${RANK_ENS:-2}
echo "stage=$STAGE K=$K task=$SLURM_ARRAY_TASK_ID of list $KLIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u r3_rank.py
rc=$?
rm -rf $LADDER_MMAP_DIR
echo "RANK STAGE $STAGE K=$K DONE rc=$rc $(date)"
exit $rc
