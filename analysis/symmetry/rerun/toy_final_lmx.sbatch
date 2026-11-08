#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -J lmx_toy
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# toy_final.sbatch of the rerun (Fig. 13, \Toy*): the Gaussian toy with the paper's network and training,
# three trainings at K = 48 and one at each K of the scan. It trains its own small networks on its own
# events and depends on none of the decisions; it is rerun because every result is regenerated (E5), as a
# job of its own. On preq by default, the nine side by side on one CPU node (eight threads each) instead of
# nine H200 jobs; on a GPU partition only if the authors approve it (-p and --gres at sbatch time, as
# submit_scoring_lmx.sh does with LMX_PARTITION and LMX_GRES: the nine then share the job's GPU and CPUs).
# On another device its numbers can differ from the published ones in the last digits. The step count
# is the paper's (TOY_STEPS of the submitting shell is cleared). The outputs are removed when the job
# starts and recorded when every training ended well (output/lmx_records/toy.json).
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/toy_final_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
unset TOY_STEPS
export CUDA_CACHE_PATH=$L/cuda-cache; mkdir -p "$CUDA_CACHE_PATH"     # a GPU's compiled kernels, not in the home folder
need_gpu || exit 3                  # on h200preq: torch must see the GPU before the outputs are removed (common_lmx.sh)
RUNS="48:0 48:1 48:2 1:0 2:0 3:0 4:0 8:0 16:0"
OUTS=$(for s in $RUNS; do echo output/toy_final_K${s%%:*}_s${s##*:}.json output/toy_final_K${s%%:*}_s${s##*:}.npz; done)
rm -f $OUTS output/lmx_records/toy.json
NT=$(( ${SLURM_CPUS_PER_TASK:-72} / 9 )); [ $NT -ge 1 ] || NT=1
PIDS=()
for s in $RUNS; do
  IFS=: read -r K S <<< "$s"
  ( TOY_K=$K TOY_SEED=$S LADDER_ACT=silu LADDER_EMB_SCALE=auto OMP_NUM_THREADS=$NT TOY_OUT=output/toy_final_K${K}_s$S \
    "$PY" -B -u toy_final.py ) > logs/toy_K${K}_s${S}_${SLURM_JOB_ID}.log 2>&1 &
  PIDS+=($!)
done
FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
tail -qn 1 logs/toy_K*_${SLURM_JOB_ID}.log
[ $FAIL = 0 ] && { step_record toy $OUTS || FAIL=1; }
echo "TOY DONE (failures: $FAIL) $(date)"; exit $FAIL
