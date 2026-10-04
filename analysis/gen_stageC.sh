#!/usr/bin/env bash
# Stage C: reweight THREE parameters at once from different physics sectors -- alpha_s
# (shower), STRANGE_FRACTION (flavor: kaon/strange yields) and KT_0 (kinematic: primordial
# kT, moves multiplicity). Sherpa 3, Ahadic, CSS. 3x3x3 = 27 training + 5 held-out interior.
# Full-event FLAVOR representation only (extract_particles_full_flavor.py), the representation
# Stages A/B/D showed is the right one. 80k events per run.
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
ASV=(0.112 0.120 0.128)
SFV=(0.30 0.46 0.65)
KTV=(0.80 1.21 1.80)
rid=6800
for a in "${ASV[@]}"; do for s in "${SFV[@]}"; do for k in "${KTV[@]}"; do gen $rid $a $s $k 80000; rid=$((rid+1)); done; done; done
# held-out interior (alpha_s, strange, kt)
gen 6900 0.116 0.38 1.00 80000
gen 6901 0.124 0.55 1.50 80000
gen 6902 0.120 0.46 1.21 80000
gen 6903 0.114 0.60 0.95 80000
gen 6904 0.126 0.35 1.60 80000
echo "STAGE C GENERATION DONE"