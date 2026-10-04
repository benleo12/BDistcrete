#!/usr/bin/env bash
# Dense validation set for the "any point of the box" claim: 12 fresh runs at interior and
# near-edge points of the Stage C box, never seen in training or the original held-out set.
# Identical recipe to gen_stageC.sh (Sherpa 3.0.4, CSS, Ahadic, 80k events).
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_stageC"; DATA="$SD/data_stageC"; mkdir -p "$RUNS" "$DATA"
gen () { local rid=$1 a=$2 sf=$3 kt=$4 ev=$5 rd="$RUNS/run_$1"
  local full="$DATA/particles_full_$1.npz" shp="$DATA/shapes_run_$1.csv"
  [ -f "$full" ] && [ -f "$shp" ] && { echo "skip $rid"; return; }
  mkdir -p "$rd"
  cat > "$rd/Sherpa.yaml" <<EOF
EVENTS: $ev
EVENT_OUTPUT: HepMC3_GenEvent[events]

BEAMS: [11, -11]
BEAM_ENERGIES: 45.6

ALPHAS(MZ): $a
ORDER_ALPHAS: 1

SHOWER_GENERATOR: CSS
FRAGMENTATION: Ahadic
AHADIC:
  STRANGE_FRACTION: $sf
  KT_0: $kt

PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
EOF
  ( cd "$rd" && "$SHERPA3" ) >/dev/null 2>&1 || { echo "FAIL $rid"; return; }
  for cand in "$rd/events.hepmc3" "$rd/events" "$rd/events.hepmc"; do
    [ -f "$cand" ] && [ "$cand" != "$rd/events.hepmc" ] && { mv "$cand" "$rd/events.hepmc"; break; }; done
  [ -f "$rd/events.hepmc" ] || { echo "no hepmc $rid"; return; }
  "$PY" "$SD/extract_particles_full_flavor.py" --input "$rd/events.hepmc" --output "$full" >/dev/null 2>&1
  "$PY" "$SD/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$DATA" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid (a=$a sf=$sf kt=$kt)"
}
# interior spread + near-edge stress points (none coincide with training or held points)
gen 6950 0.1130 0.33 0.90 80000
gen 6951 0.1150 0.52 1.65 80000
gen 6952 0.1170 0.62 1.10 80000
gen 6953 0.1190 0.36 1.40 80000
gen 6954 0.1210 0.58 0.85 80000
gen 6955 0.1230 0.42 1.70 80000
gen 6956 0.1250 0.50 1.05 80000
gen 6957 0.1270 0.32 1.30 80000
gen 6958 0.1135 0.63 1.75 80000
gen 6959 0.1275 0.64 0.82 80000
gen 6960 0.1180 0.48 0.95 80000
gen 6961 0.1220 0.40 1.55 80000
echo "VALIDATION SET DONE"
