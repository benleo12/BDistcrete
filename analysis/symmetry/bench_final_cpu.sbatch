#!/bin/bash
#SBATCH -p preq
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -J lmx_bench_cpu
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/bench_final_cpu_%j.out
# Cost benchmark of Sec. 3.6 (bench_cost_final.py) on one CPU process with four threads, on a node of
# its own so that no other job shares its memory bandwidth. The concatenation network is timed once
# without warm-up (tens of minutes per call), and its full fit step is left out (it is its network
# step plus the design-independent part, both measured). BENCH_MIX=linear (the default) times the linear mixture
# (both generators' weights at a point) and writes output/bench_final_cpu_4_lin.json; BENCH_MIX=geometric the
# earlier draft's mixture (output/bench_final_cpu_4.json). BENCH_REF and BENCH_PARTICLES override the bundles.
#   MAXENT_DIR=$RUN sbatch --export=ALL bench_final_cpu.sbatch
# It runs only in the rerun's folder RUN (E7): MAXENT_DIR must resolve to it, and bench_cost_final.py there must have
# the BENCH_MIX switch. The rerun's checks are on: the bundle of the observables must be one of the beam-photon rule
# in use (BEAM_PHOTONS=remove, bundle_rule.py) and the export must carry the refitted temperature (T_DIR, E2);
# TORCH_ENV the Python environment (default /mnt/beegfs/bassi/reweighting/env/torch).
set -uo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # E7: the only folder this script runs in
[ "$(realpath "${MAXENT_DIR:-/nonexistent}" 2>/dev/null)" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2; }
grep -q "^MIX = os.environ.get('BENCH_MIX'" "$RUN/bench_cost_final.py" || { echo "$RUN/bench_cost_final.py has no BENCH_MIX switch (an earlier draft's script)"; exit 2; }
TORCH_ENV=${TORCH_ENV:-/mnt/beegfs/bassi/reweighting/env/torch}
cd "$RUN" || exit 2; [ -f $TORCH_ENV/bin/activate ] || { echo "no Python environment at $TORCH_ENV"; exit 2; }; source $TORCH_ENV/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RIVET_REF=$RUN/rivet_ref PYTHONDONTWRITEBYTECODE=1
export BEAM_PHOTONS=remove T_DIR=$RUN/output/temperature
export BENCH_DEVICE=cpu BENCH_THREADS=4 BENCH_EARLY_REPS=1 BENCH_EARLY_WARMUP=0 BENCH_SKIP=${BENCH_SKIP:-b_early} BENCH_MIX=${BENCH_MIX:-linear}
echo "=== bench cpu ($BENCH_MIX) on $(hostname) $(date)"; lscpu | grep -i "model name"
python3 -B -u bench_cost_final.py || { echo "BENCH FAILED"; exit 3; }
echo "BENCH CPU DONE $(date)"
