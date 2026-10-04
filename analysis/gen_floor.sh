#!/usr/bin/env bash
# Precision-floor runs: 800k fresh events at one central and one off-centre held-out point,
# with EXPLICIT new random seeds (the default seed is fixed, so an unseeded rerun at the same
# parameters would reproduce the earlier events and fake the test).
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_stageC"; DATA="$SD/data_stageC"
gen () { local rid=$1 a=$2 sf=$3 kt=$4 ev=$5 seed=$6 rd="$RUNS/run_$1"
  local full="$DATA/particles_full_$1.npz" shp="$DATA/shapes_run_$1.csv"
  [ -f "$full" ] && [ -f "$shp" ] && { echo "skip $rid"; return; }
  mkdir -p "$rd"
  cat > "$rd/Sherpa.yaml" <<EOF
EVENTS: $ev
RANDOM_SEED: $seed
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
  rm -f "$rd/events.hepmc"; echo "done $rid (a=$a sf=$sf kt=$kt seed=$seed ev=$ev)"
}
gen 6970 0.120 0.46 1.21 800000 77001    # centre, same params as held 6902, new seed
gen 6971 0.124 0.55 1.50 800000 77002    # off-centre, same params as held 6901, new seed
echo "FLOOR RUNS DONE"
