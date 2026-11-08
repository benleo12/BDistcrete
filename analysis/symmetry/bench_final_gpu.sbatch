#!/bin/bash
#SBATCH -p preq
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J lmx_bench_gpu
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/bench_final_gpu_%j.out
# Cost benchmark of Sec. 3.6 (bench_cost_final.py) on one H200, with the gradient check. BENCH_MIX=linear (the
# default) times the linear mixture and writes output/bench_final_cuda_12_lin.json; BENCH_MIX=geometric the earlier
# draft's mixture (output/bench_final_cuda_12.json). BENCH_REF and BENCH_PARTICLES override the bundles.
# The partition and the GPU are given on the sbatch command line (E8): the header asks for preq, which has no GPU, and
# the script then stops. Only with the user's approval of a GPU partition (RERUN_PROP step 5b checks it), for example
#   LMX_GPU_APPROVED=1 LMX_PARTITION=h200preq LMX_GRES=gpu:h200:1
#   MAXENT_DIR=$RUN sbatch -p "${LMX_PARTITION:?}" --gres="${LMX_GRES:?}" --export=ALL bench_final_gpu.sbatch
# (the earlier draft's run, 578373 on h200q, peaked at 4.9 GB of host memory). It runs only in the rerun's folder RUN
# (E7): MAXENT_DIR must resolve to it, and bench_cost_final.py there must have the BENCH_MIX switch. The rerun's
# checks are on (BEAM_PHOTONS=remove, bundle_rule.py; T_DIR, E2), and CUDA's kernel cache stays in lmx_runs.
set -uo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # E7: the only folder this script runs in
[ "$(realpath "${MAXENT_DIR:-/nonexistent}" 2>/dev/null)" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2; }
grep -q "^MIX = os.environ.get('BENCH_MIX'" "$RUN/bench_cost_final.py" || { echo "$RUN/bench_cost_final.py has no BENCH_MIX switch (an earlier draft's script)"; exit 2; }
nvidia-smi -L 2>/dev/null | grep -q '^GPU ' || { echo "no GPU in this job: submit with -p \$LMX_PARTITION --gres=\$LMX_GRES (E8), with the user's approval"; exit 2; }
L=$(dirname "$RUN")                                        # lmx_runs
TORCH_ENV=${TORCH_ENV:-/mnt/beegfs/bassi/reweighting/env/torch}
cd "$RUN" || exit 2; [ -f $TORCH_ENV/bin/activate ] || { echo "no Python environment at $TORCH_ENV"; exit 2; }; source $TORCH_ENV/bin/activate
export OMP_NUM_THREADS=12 RIVET_REF=$RUN/rivet_ref PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_PATH=$L/cuda-cache
export BEAM_PHOTONS=remove T_DIR=$RUN/output/temperature
mkdir -p "$CUDA_CACHE_PATH"
export BENCH_DEVICE=cuda BENCH_THREADS=12 BENCH_MIX=${BENCH_MIX:-linear}
echo "=== bench gpu ($BENCH_MIX) on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -B -u bench_cost_final.py || { echo "BENCH FAILED"; exit 3; }
echo "BENCH GPU DONE $(date)"
