# Sourced by every batch script of the scoring side of the rerun (analysis/symmetry/rerun/).
# The rerun runs in RUN only: MAXENT_DIR must be RUN, the scripts are copied there (sync_run.sh), the data
# folders data_* and ../gen/data_* are links to the photon-free (_nb) folders, and every result is
# written under RUN/output. Every python runs without writing bytecode.
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
if [ "${MAXENT_DIR:-}" != "$RUN" ]; then
  echo "MAXENT_DIR must be $RUN (it is '${MAXENT_DIR:-unset}')"; exit 2
fi
cd "$RUN" || exit 2
[ "$(realpath "$RUN")" = "$RUN" ] || { echo "$RUN is a link to $(realpath "$RUN")"; exit 2; }
# the keys of a scoring line (specs/*.txt, score_p2_lmx.sbatch): score_p2_array_lmx.sbatch clears them before it
# sets a line, and the fingerprint of a scoring holds them (symmetry/rerun/lmx_fingerprint.py)
export SPEC_KEYS="OUT STAGE NET SEED K ENS STEPS MODEL DATA TMODE TRAINED ABC E_CSV F_CSV STAGEW_CSV REF_RUNS NREF_TOTAL SUFFIX DUMP TRUNCATE LOGIT_CACHE"
# the knobs the scripts of the scoring side read from the environment, which the submitting shell may pass on
# (--export=ALL): cleared before anything below or in the job sets what the job uses. The keys of a spec line
# (REF_RUNS and STAGEW_CSV match these names) stay, since score_p2_array_lmx.sbatch sets them.
KNOBS='^(LADDER_|DM_|VALID_|SM_|WIFI_|REF_|RECAL_|WIDTHS_|CLOSURE_|MIX_|XP_|KSCAN_|COVERAGE_|GENSPLIT_|TOY_|RELEASE_)|^STAGE._(DATA|CSV)$|^(STAGED_OUT|STAGE_TABLE_SUBSET|PHOTON_FREE_DIR_RE|BEAM_PHOTONS_ALLOW_MIXED|TORCH_PY|NDEN|NDEN_PER|NFIT_PER|DROP|OUTLIER_SIGMA|LMX_CHECK)$'
for _v in $(compgen -e | grep -E "$KNOBS"); do case " $SPEC_KEYS " in *" $_v "*) ;; *) unset "$_v" ;; esac; done; unset _v
R=$(dirname "$RUN")                                     # /mnt/beegfs/bassi/reweighting/lmx_runs
PY=/mnt/beegfs/bassi/reweighting/env/torch/bin/python
[ -x "$PY" ] || { echo "no python at $PY"; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
# D4 at score time, and the refitted temperatures (D11) and the per-bundle particle caches
export BEAM_PHOTONS=remove T_DIR=output/temperature POBS_DIR=output/pobs
# the temperature scans reuse numpy's own block sorting of each reference observable: the same
# histograms bit for bit, about twenty times faster (r2_ladder.register_sorted)
export LADDER_FAST_HIST=1
mkdir -p logs output/models output/temperature output/pobs output/published
# the published bundles (old observables, read only for their particles and for the check that a
# bundle in output/models is the photon-free one) and the published results (the training records)
PUB=output/published
# a data folder of the rerun must resolve to a photon-free (_nb) folder
need_nb() {
  local d r; for d in "$@"; do
    r=$(realpath "$d" 2>/dev/null) || { echo "missing data folder $d"; return 2; }
    case "$r" in *_nb|*_nb/*) ;; *) echo "$d resolves to $r, not a photon-free (_nb) folder"; return 2 ;; esac
  done
}
need_file() { local f; for f in "$@"; do [ -e "$f" ] || { echo "missing input $f"; return 2; }; done; }
# a refreshed bundle must be listed in output/bundle_rule.json with the beam-photon rule of RUN (I1, I2:
# bundle_rule.check); score_obs.py checks the same where it reads one
need_rule() {
  "$PY" -B -c 'import sys, bundle_rule; [bundle_rule.check(p) for p in sys.argv[1:]]' "$@" \
    || { echo "not a refreshed bundle of the beam-photon rule in RUN (output/bundle_rule.json): $*"; return 2; }
}
# a step of the scoring side records its outputs when it ends well (output/lmx_records/<step>.json, with the
# mark and the code), after removing them when it starts, so a failed step leaves none of them
step_record() { "$PY" -B symmetry/rerun/lmx_fingerprint.py record "$@" || { echo "RECORD FAILED $1"; return 3; }; }
# A job on h200preq, or one that Slurm gave a GPU, must see that GPU before it scores or trains: r2_ladder.py and
# toy_final.py fall back to the CPU in silence, and a DCTR scoring on the CPU would put CPU passes into a logit cache
# that the other elements fill on the GPU (runbook 6.6). need_gpu prints the device ("GPU: ..."), or returns 3 with
# "GPU CHECK FAILED" when torch sees none. On preq, with no GPU, it does nothing. LMX_GRES is no sign of a GPU here:
# --export=ALL passes it from the submitting shell to the preq jobs too.
need_gpu() {
  [ "${SLURM_JOB_PARTITION:-}" = h200preq ] || [ -n "${SLURM_JOB_GPUS:-}" ] || return 0
  "$PY" -B -c 'import os, sys, torch
v = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
if not torch.cuda.is_available():
    print(f"torch {torch.__version__} sees no GPU (CUDA_VISIBLE_DEVICES={v})"); sys.exit(3)
print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} visible (torch {torch.__version__}, CUDA_VISIBLE_DEVICES={v})")' \
    || { echo "GPU CHECK FAILED on $(hostname -s) (partition ${SLURM_JOB_PARTITION:-none}, job ${SLURM_JOB_ID:-none}): nothing was scored or removed"; return 3; }
}
