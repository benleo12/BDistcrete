#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -J lmx_scorearr
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%A_%a.out
# score_p2_array.sbatch of the rerun: the scorings of a spec file (specs/*.txt, make_specs.py), NPAR
# of them side by side on one preq node (80 threads, 187 GB; whole nodes, so packing saves nodes).
# Element i runs the non-comment lines i*NPAR .. i*NPAR+NPAR-1, each with score_p2_lmx.sbatch and
# 80/NPAR threads (the CPUs of the job over NPAR when -c is given), and fails if any of them failed.
# A preempted element is requeued and runs again from the start; every result is written at the end of
# its scoring only, and a scoring whose result carries its fingerprint is skipped (score_p2_lmx.sbatch).
# The keys of a line are cleared first (and score_p2_lmx.sbatch clears the scorers' other knobs), so a variable
# of the submitting shell does not change a scoring.
# preq by default; submit_scoring_lmx.sh may give the DCTR arrays a GPU partition (-p, --gres, -c, --mem).
#   N=$(grep -v '^#' specs/std_fac.txt | grep -c .); NPAR=2
#   sbatch --array=0-$(( (N+NPAR-1)/NPAR - 1 )) --export=ALL,MAXENT_DIR=$RUN,SPECS=symmetry/rerun/specs/std_fac.txt,NPAR=$NPAR \
#          symmetry/rerun/score_p2_array_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
export CUDA_CACHE_PATH=$L/cuda-cache; mkdir -p "$CUDA_CACHE_PATH"     # a GPU's compiled kernels, not in the home folder
need_gpu || exit 3                  # on h200preq: torch must see the GPU before any line is scored (common_lmx.sh)
need_file "${SPECS:?}" || exit 2
NPAR=${NPAR:-2}; I=${SLURM_ARRAY_TASK_ID:?}
LINES=(); while IFS= read -r l; do LINES+=("$l"); done < <(grep -v '^[[:space:]]*#' "$SPECS" | grep -v '^[[:space:]]*$')
NT=$(( ${SLURM_CPUS_PER_TASK:-80} / NPAR )); [ $NT -ge 1 ] || NT=1
PIDS=(); NAMES=()
for ((j = 0; j < NPAR; j++)); do
  n=$((I*NPAR + j)); [ $n -lt ${#LINES[@]} ] || break
  line=${LINES[$n]}; out=$(sed -E 's/.*OUT=output\/([^ ]+)\.json.*/\1/' <<< "$line")
  echo "=== element $I, line $n: $line"
  ( env $(printf -- '-u %s ' $SPEC_KEYS) $line NTHREADS=$NT bash symmetry/rerun/score_p2_lmx.sbatch ) \
    >> logs/score_${out}_${SLURM_ARRAY_JOB_ID}_$I.log 2>&1 &
  PIDS+=($!); NAMES+=("$out")
done
FAIL=0
for k in "${!PIDS[@]}"; do
  if wait "${PIDS[$k]}"; then echo "done   ${NAMES[$k]}"; else echo "FAILED ${NAMES[$k]} (logs/score_${NAMES[$k]}_${SLURM_ARRAY_JOB_ID}_$I.log)"; FAIL=1; fi
done
exit $FAIL
