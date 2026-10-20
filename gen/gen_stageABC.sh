#!/bin/bash
# One Stage A, B or C run (the one-, two- and three-parameter ladder of the paper): Sherpa 3.0.4
# from CVMFS with the pinned build of cvmfs_env.sh, the CSS shower and Ahadic, at the Z pole.
# Only ALPHAS(MZ) and the listed Ahadic parameters move; '-' leaves a parameter at its default.
# usage: gen_stageABC.sh <rid> <alphas> <baryon_fraction|-> <strange_fraction|-> <kt_0|-> <events>
set -e
rid=$1; as=$2; bf=$3; sf=$4; kt=$5; ev=$6
[ $# -eq 6 ] || { echo "BAD ARGC $# (want 6) for rid=$rid"; exit 2; }
seed=${GEN_SEED:-$rid}
chk() { [ "$2" = "-" ] && return 0; awk -v v="$2" -v lo="$3" -v hi="$4" -v n="$1" \
  'BEGIN{ if (v+0<lo || v+0>hi) { printf "BAD %s=%s, expected [%g,%g]\n", n, v, lo, hi; exit 1 } }' || exit 2; }
chk ALPHAS "$as" 0.05 0.25 ; chk BARYON "$bf" 0.01 0.8 ; chk STRANGE "$sf" 0.05 0.95
chk KT_0 "$kt" 0.1 5 ; chk EVENTS "$ev" 100 5000000
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cvmfs_env.sh" sherpa || exit 1
BASE=${STAGEABC_BASE:-$PWD}; DATA=${STAGEABC_DATA:-$BASE/data}; rd=$BASE/runs/run_$rid
mkdir -p "$rd" "$DATA"; cd "$rd"
ahadic=""
[ "$bf" != "-" ] && ahadic="$ahadic  BARYON_FRACTION: $bf"$'\n'
[ "$sf" != "-" ] && ahadic="$ahadic  STRANGE_FRACTION: $sf"$'\n'
[ "$kt" != "-" ] && ahadic="$ahadic  KT_0: $kt"$'\n'
{
cat <<YAML
EVENTS: $ev
EVENT_OUTPUT: HepMC3_GenEvent[events]
RANDOM_SEED: $seed
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
ALPHAS(MZ): $as
ORDER_ALPHAS: 1
SHOWER_GENERATOR: CSS
FRAGMENTATION: Ahadic
YAML
[ -n "$ahadic" ] && printf 'AHADIC:\n%s' "$ahadic"
cat <<YAML
PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
YAML
} > Sherpa.yaml
Sherpa > sherpa.log 2>&1 || { echo "SHERPA FAIL $rid"; exit 1; }
for c in events.hepmc3 events; do [ -f "$c" ] && mv "$c" events.hepmc && break; done
[ -f events.hepmc ] || { echo "NO HEPMC $rid"; exit 1; }
python3 "$HERE/extract_particles_full_flavor.py" --input events.hepmc --output $DATA/particles_full_${rid}.npz > extract.log 2>&1 || { echo "EXTRACT FAIL $rid"; tail -5 extract.log; exit 1; }
python3 "$HERE/compute_shapes_only.py" --input events.hepmc --output-dir $DATA --run-id $rid > shapes.log 2>&1 || { echo "SHAPES FAIL $rid"; tail -5 shapes.log; exit 1; }
rm -f events.hepmc; rm -rf Process Results.zip 2>/dev/null
echo "done $rid (as=$as bf=$bf sf=$sf kt=$kt seed=$seed)"
