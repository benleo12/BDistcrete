#!/usr/bin/env bash
# Stage A full-vs-hemisphere regeneration. Same alpha_s grid as the original Stage A
# (13 training 0.110-0.130, 6 held-out interior, 2 out-of-box), Sherpa 3, Ahadic, CSS.
# Extract BOTH representations from the same HepMC (heavy-hemisphere + full-event thrust
# frame) + shapes, so the conditional can be trained on each and the B_total / rho_heavy
# residuals compared. 100k events per run.
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_stageA_full"; DATA="$SD/data_stageA_full"; mkdir -p "$RUNS" "$DATA"
gen () { local rid=$1 a=$2 ev=$3 rd="$RUNS/run_$1"
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

PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
EOF
  ( cd "$rd" && "$SHERPA3" ) >/dev/null 2>&1 || { echo "FAIL $rid"; return; }
  for cand in "$rd/events.hepmc3" "$rd/events" "$rd/events.hepmc"; do
    [ -f "$cand" ] && [ "$cand" != "$rd/events.hepmc" ] && { mv "$cand" "$rd/events.hepmc"; break; }; done
  [ -f "$rd/events.hepmc" ] || { echo "no hepmc $rid"; return; }
  "$PY" "$SD/extract_particles.py"      --input "$rd/events.hepmc" --output "$hemi" >/dev/null 2>&1
  "$PY" "$SD/extract_particles_full.py" --input "$rd/events.hepmc" --output "$full" >/dev/null 2>&1
  "$PY" "$SD/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$DATA" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid (a=$a)"
}
# training grid 0.110-0.130
TR_A=(0.11000 0.11167 0.11333 0.11500 0.11667 0.11833 0.12000 0.12167 0.12333 0.12500 0.12667 0.12833 0.13000)
rid=6100; for a in "${TR_A[@]}"; do gen $rid $a 100000; rid=$((rid+1)); done
# held-out interior
gen 6200 0.11250 100000; gen 6201 0.11583 100000; gen 6202 0.11917 100000
gen 6203 0.12250 100000; gen 6204 0.12583 100000; gen 6205 0.12917 100000
# out of box
gen 6206 0.10800 100000; gen 6207 0.13200 100000
echo "STAGE A FULL GENERATION DONE"