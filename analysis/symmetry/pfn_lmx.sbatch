#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J lmx_pfn
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# The PFN cross-check of App. A.2 in the rerun: the logic of run_crosscheck_seeds.sh, five seeds each of four
# implementations, 20 trainings one at a time, on runs 9000 and 9010 of the wide coupling scan (crosscheck_data.py, as
# published). Three are those of the published run: the energyflow PFN (TensorFlow, crosscheck_ef.py), and with
# crosscheck_torch.py our conditional PFN collapsed to K = 1 (torchK1) and a plain PyTorch PFN with the energyflow
# architecture (torchplain), both with the package's ReLU and the sum over particles not scaled. The fourth, torchK1silu
# (round B15), is the same conditional PFN as the paper builds it: LADDER_ACT=silu and LADDER_EMB_SCALE=auto,
# the sum over particles divided by a scale fixed before training on the first 4096 training events. App. A.2 compares
# that network, with SiLU, against the package's ReLU, and torchK1 is not it. The network inputs do not change in the
# rerun (D4), so neither do the data; every network is trained again. The PyTorch part runs on the CPU
# (CROSSCHECK_DEVICE=cpu; the published run used the laptop's GPU, mps), so the AUCs can move within the spread between
# seeds.
#   CROSSCHECK_DATA=<folder of particles_full_9000.npz and particles_full_9010.npz> MAXENT_DIR=$RUN \
#     sbatch --export=ALL symmetry/pfn_lmx.sbatch
# It runs only in RUN (E7), and only after the scoring side's first submission (the start mark
# output/.lmx_rerun_start must exist: that submission moves aside every earlier output of this job). CROSSCHECK_DATA
# is required and only read; the md5 of both particle files goes into the log. The energyflow part uses the venv
# EF_VENV (default lmx_runs/pfn_venv, which round B7 built; RERUN_PROP says how to check it) without PYTHONPATH (pylib
# is for the torch Python 3.12), the PyTorch part TORCH_ENV with lmx_runs/pylib on its path (for scikit-learn). No
# LADDER_* or STAGE*_ setting of the submitting shell reaches the trainings (r2_ladder reads them at import); the
# fourth sets its two on its own command line. Writes output/crosscheck_seeds/<impl>_s<seed>.json (each with its AUC,
# its parameter count, params, and for PyTorch the network trained: act, and for the conditional PFN emb_scale and
# emb_scale_mode), one log per training in logs/, and output/crosscheck_seeds_summary.json in the published format
# with the counts of App. A.2 and torchK1silu added (crosscheck_summary.py; make_numbers.py reads them). A requeued job
# keeps the trainings an earlier attempt of the same job finished; a new submission starts from nothing.
set -uo pipefail
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # E7: the only folder this script runs in
[ "$(realpath "${MAXENT_DIR:-/nonexistent}" 2>/dev/null)" = "$RUN" ] || { echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2; }
L=$(dirname "$RUN")                                        # lmx_runs
: "${CROSSCHECK_DATA:?set CROSSCHECK_DATA to the folder of particles_full_9000.npz and particles_full_9010.npz (read only)}"
for r in 9000 9010; do
  [ -r "$CROSSCHECK_DATA/particles_full_$r.npz" ] || { echo "no $CROSSCHECK_DATA/particles_full_$r.npz"; exit 2; }
done
EF_VENV=${EF_VENV:-$L/pfn_venv}; TORCH_ENV=${TORCH_ENV:-/mnt/beegfs/bassi/reweighting/env/torch}
[ -x "$EF_VENV/bin/python" ] || { echo "no energyflow venv at $EF_VENV (lmx_runs/pfn_venv, built in round B7; RERUN_PROP section 0)"; exit 2; }
[ -x "$TORCH_ENV/bin/python" ] || { echo "no Python environment at $TORCH_ENV"; exit 2; }
cd "$RUN" || exit 2
[ -e output/.lmx_rerun_start ] || { echo "no output/.lmx_rerun_start: submit the scoring side first (its park would move this job's outputs aside)"; exit 2; }
for f in crosscheck_data.py crosscheck_ef.py crosscheck_torch.py crosscheck_summary.py r2_ladder.py score_obs.py; do
  [ -f "$f" ] || { echo "missing $RUN/$f (sync_run.sh copies it)"; exit 2; }
done
# the published task, file names and seeds only (the scripts read these knobs), and r2_ladder's network as published
# (torchK1silu sets LADDER_ACT and LADDER_EMB_SCALE on its own line below, and only there)
unset CROSSCHECK_PAIR CROSSCHECK_SEED CROSSCHECK_OUT CROSSCHECK_PLAIN
for v in $(compgen -e | grep -E '^(LADDER|STAGE[A-Z]*)_' || true); do unset "$v"; done
NT=${SLURM_CPUS_PER_TASK:-16}
export CROSSCHECK_DATA CROSSCHECK_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
export TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=2 KERAS_HOME=$L/keras-home XDG_CACHE_HOME=$L/xdg-cache
O=output/crosscheck_seeds; mkdir -p logs $O "$KERAS_HOME" "$XDG_CACHE_HOME"
if [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then      # a new submission: nothing of an earlier one is kept
  rm -f $O/ef_s*.json $O/torchK1_s*.json $O/torchplain_s*.json $O/torchK1silu_s*.json $O/*.json.tmp output/crosscheck_seeds_summary.json
fi
echo "=== PFN cross-check on $(hostname), $NT threads, data $CROSSCHECK_DATA ($(realpath "$CROSSCHECK_DATA")) $(date)"
md5sum "$CROSSCHECK_DATA"/particles_full_9000.npz "$CROSSCHECK_DATA"/particles_full_9010.npz     # the events trained on
env -u PYTHONPATH "$EF_VENV/bin/python" -B -c 'import sys, tensorflow, energyflow, sklearn, numpy; print("venv", sys.version.split()[0], "tensorflow", tensorflow.__version__, "energyflow", energyflow.__version__, "sklearn", sklearn.__version__, "numpy", numpy.__version__)' 2>/dev/null || echo "the venv $EF_VENV does not import tensorflow, energyflow and sklearn"
bad=
train() {   # <name> <command...>: writes $O/<name>.json through a temporary file, its log in logs/
  local n=$1; shift
  if [ -s $O/$n.json ]; then echo "=== $n: finished by an earlier attempt of this job, kept"; return 0; fi
  local log=logs/lmx_pfn_${n}_${SLURM_JOB_ID:-0}.log
  echo "=== $n $(date)"
  CROSSCHECK_OUT=$O/$n.json.tmp "$@" > $log 2>&1; local rc=$?
  grep -E "AUC" $log
  if [ $rc = 0 ] && "$TORCH_ENV/bin/python" -B -c 'import json, sys; d = json.load(open(sys.argv[1])); assert "auc" in d and "params" in d' $O/$n.json.tmp 2>/dev/null; then
    mv $O/$n.json.tmp $O/$n.json
  else
    echo "$n FAILED (exit $rc, log $log)"; tail -3 $log; rm -f $O/$n.json.tmp; bad="$bad $n"
  fi
}
for s in 1 2 3 4 5; do
  train ef_s$s env -u PYTHONPATH TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 CROSSCHECK_SEED=$s "$EF_VENV/bin/python" -B -u crosscheck_ef.py
  train torchK1_s$s env PYTHONPATH=$L/pylib CROSSCHECK_SEED=$s "$TORCH_ENV/bin/python" -B -u crosscheck_torch.py
  train torchplain_s$s env PYTHONPATH=$L/pylib CROSSCHECK_SEED=$s CROSSCHECK_PLAIN=1 "$TORCH_ENV/bin/python" -B -u crosscheck_torch.py
  train torchK1silu_s$s env PYTHONPATH=$L/pylib LADDER_ACT=silu LADDER_EMB_SCALE=auto CROSSCHECK_SEED=$s "$TORCH_ENV/bin/python" -B -u crosscheck_torch.py
done
[ -z "$bad" ] && { "$TORCH_ENV/bin/python" -B -u crosscheck_summary.py $O output/crosscheck_seeds_summary.json || bad="$bad summary"; }   # only when every training ended well
if [ -n "$bad" ]; then echo "PFN CROSSCHECK FAILED:$bad"; exit 1; fi
echo "PFN CROSSCHECK DONE $(date)"
