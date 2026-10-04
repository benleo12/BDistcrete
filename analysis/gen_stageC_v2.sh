#!/usr/bin/env bash
# Stage C: reweight THREE parameters at once from different physics sectors -- alpha_s
# (shower), STRANGE_FRACTION (flavor: kaon/strange yields) and KT_0 (kinematic: primordial
# kT, moves multiplicity). Sherpa 3, Ahadic, CSS. 3x3x3 = 27 training + 5 held-out interior.
# Full-event FLAVOR representation only (extract_particles_full_flavor.py), the representation
# Stages A/B/D showed is the right one. 80k events per run.
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_stageC_v2"; DATA="$SD/data_stageC_v2"; mkdir -p "$RUNS" "$DATA"
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
  "$PY" "$SD/release/generators/extract_particles_full_flavor.py" --input "$rd/events.hepmc" --output "$full" >/dev/null 2>&1
  "$PY" "$SD/release/generators/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$DATA" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid (a=$a sf=$sf kt=$kt)"
}
if [ "${1:-}" = "one" ]; then shift; gen "$@"; exit 0; fi
NPAR=${NPAR:-6}
printf "%s\n" "6800 0.112 0.30 0.80 80000" "6801 0.112 0.30 1.21 80000" "6802 0.112 0.30 1.80 80000" "6803 0.112 0.46 0.80 80000" "6804 0.112 0.46 1.21 80000" "6805 0.112 0.46 1.80 80000" "6806 0.112 0.65 0.80 80000" "6807 0.112 0.65 1.21 80000" "6808 0.112 0.65 1.80 80000" "6809 0.120 0.30 0.80 80000" "6810 0.120 0.30 1.21 80000" "6811 0.120 0.30 1.80 80000" "6812 0.120 0.46 0.80 80000" "6813 0.120 0.46 1.21 80000" "6814 0.120 0.46 1.80 80000" "6815 0.120 0.65 0.80 80000" "6816 0.120 0.65 1.21 80000" "6817 0.120 0.65 1.80 80000" "6818 0.128 0.30 0.80 80000" "6819 0.128 0.30 1.21 80000" "6820 0.128 0.30 1.80 80000" "6821 0.128 0.46 0.80 80000" "6822 0.128 0.46 1.21 80000" "6823 0.128 0.46 1.80 80000" "6824 0.128 0.65 0.80 80000" "6825 0.128 0.65 1.21 80000" "6826 0.128 0.65 1.80 80000" "6900 0.116 0.38 1.00 80000" "6901 0.124 0.55 1.50 80000" "6902 0.120 0.46 1.21 80000" "6903 0.114 0.60 0.95 80000" "6904 0.126 0.35 1.60 80000" "6950 0.1130 0.33 0.90 80000" "6951 0.1150 0.52 1.65 80000" "6952 0.1170 0.62 1.10 80000" "6953 0.1190 0.36 1.40 80000" "6954 0.1210 0.58 0.85 80000" "6955 0.1230 0.42 1.70 80000" "6956 0.1250 0.50 1.05 80000" "6957 0.1270 0.32 1.30 80000" "6958 0.1135 0.63 1.75 80000" "6959 0.1275 0.64 0.82 80000" "6960 0.1180 0.48 0.95 80000" "6961 0.1220 0.40 1.55 80000" | xargs -P $NPAR -L 1 bash "$0" one
echo "STAGE C v2 GENERATION DONE"
