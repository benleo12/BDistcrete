#!/usr/bin/env bash
# Stage B full-vs-hemisphere (flavor) regeneration. Moderate 2D grid (7 alpha_s x 4 baryon
# = 28 training + 5 held-out), Sherpa 3, Ahadic, CSS. Extract BOTH flavor representations
# from the same HepMC: heavy-hemisphere flavor (extract_particles_flavor.py) and full-event
# flavor (extract_particles_full_flavor.py) + shapes. Tests whether the full-event
# representation closes the 2D box, including B_total/rho_heavy, to the noise floor.
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_stageB_full"; DATA="$SD/data_stageB_full"; mkdir -p "$RUNS" "$DATA"
gen () { local rid=$1 a=$2 bf=$3 ev=$4 rd="$RUNS/run_$1"
  local hemi="$DATA/particles_hemi_$1.npz" full="$DATA/particles_full_$1.npz" shp="$DATA/shapes_run_$1.csv"
  [ -f "$hemi" ] && [ -f "$full" ] && [ -f "$shp" ] && { echo "skip $rid"; return; }
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
  BARYON_FRACTION: $bf

PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
EOF
  ( cd "$rd" && "$SHERPA3" ) >/dev/null 2>&1 || { echo "FAIL $rid"; return; }
  for cand in "$rd/events.hepmc3" "$rd/events" "$rd/events.hepmc"; do
    [ -f "$cand" ] && [ "$cand" != "$rd/events.hepmc" ] && { mv "$cand" "$rd/events.hepmc"; break; }; done
  [ -f "$rd/events.hepmc" ] || { echo "no hepmc $rid"; return; }
  "$PY" "$SD/extract_particles_flavor.py"      --input "$rd/events.hepmc" --output "$hemi" >/dev/null 2>&1
  "$PY" "$SD/extract_particles_full_flavor.py" --input "$rd/events.hepmc" --output "$full" >/dev/null 2>&1
  "$PY" "$SD/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$DATA" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid (a=$a, bf=$bf)"
}
ASV=(0.1100 0.1133 0.1167 0.1200 0.1233 0.1267 0.1300)
BFV=(0.05 0.15 0.25 0.35)
rid=6600
for a in "${ASV[@]}"; do for b in "${BFV[@]}"; do gen $rid $a $b 80000; rid=$((rid+1)); done; done
# held-out interior (alpha_s, baryon)
gen 6700 0.1175 0.10 80000; gen 6701 0.1225 0.30 80000; gen 6702 0.1150 0.20 80000
gen 6703 0.1280 0.30 80000; gen 6704 0.1200 0.15 80000
echo "STAGE B FULL GENERATION DONE"