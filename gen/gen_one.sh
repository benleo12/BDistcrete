#!/bin/bash
# One generator run inside the EL9 container with CVMFS mounted (called by gen_batch.sbatch).
#   gen_one.sh <gen_stageE.sh|gen_stageF.sh> <rid> <8 parameters> <events>
# Skips a run whose particle file already exists, so a batch can simply be resubmitted.
R=/mnt/beegfs/bassi/reweighting
script=$1; rid=$2
# GEN_SEED_OFFSET > 0 makes an extension: the same parameters with seed rid + offset (written to
# its own GEN_DATA directory, so the file name stays particles_full_<rid>.npz)
off=${GEN_SEED_OFFSET:-0}; seed=$(( rid + off )); tagrun=$rid; [ "$off" -gt 0 ] && tagrun=${rid}_s$seed
OUT=${GEN_DATA:?}; LOCAL=${GEN_LOCAL:?}
[ -f "$OUT/particles_full_${rid}.npz" ] && [ -f "$OUT/shapes_run_${rid}.csv" ] && { echo "skip $rid"; exit 0; }
t0=$(date +%s)
B=$LOCAL/s$seed
if GEN_SEED=$seed STAGEE_BASE=$B STAGEF_BASE=$B STAGEABC_BASE=$B STAGEE_DATA=$OUT STAGEF_DATA=$OUT STAGEABC_DATA=$OUT \
     bash "$R/gen/$script" "${@:2}" > "$R/gen/logs/run_${tagrun}.log" 2>&1; then
  echo "ok $tagrun $(( $(date +%s) - t0 ))s"; rm -rf "$B"
else
  echo "FAIL $tagrun (logs kept in $R/gen/logs/run_${tagrun}_*)"
  for f in "$B"/runs/run_$rid/*.log; do [ -f "$f" ] && cp "$f" "$R/gen/logs/run_${tagrun}_$(basename $f)"; done
  rm -f "$B"/runs/run_$rid/events.hepmc*; exit 1
fi
