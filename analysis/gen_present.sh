#!/usr/bin/env bash
# Generation for the presentation-grade maxent checks. Sequential, guarded against any
# other Sherpa on the machine.
# 1) Dire pseudo-data with FLAVOR extraction (rid 8201, data_maxent): the mismodeled case
#    for the maxent fit (CSS (alpha_s,baryon) family cannot reach a Dire shower).
# 2) Two alpha_s MARGIN runs for Stage A (rid 6113 a=0.10667, rid 6114 a=0.13333,
#    data_stageA_full): measure whether a training margin removes the edge residual.
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.
# wait for a clear machine (no other Sherpa) before generating
clear=0
for i in $(seq 1 120); do
  if pgrep -f "bin/Sherpa" >/dev/null 2>&1; then clear=0; else clear=$((clear+1)); fi
  [ "$clear" -ge 2 ] && break
  sleep 30
done

gen () { # rid alpha shower dir extractor ahadic_baryon("" for default) ev
  local rid=$1 a=$2 shower=$3 data=$4 extr=$5 bf=$6 ev=$7 rd="$SD/runs_present/run_$1"
  local parts="$data/particles_full_$1.npz" shp="$data/shapes_run_$1.csv"
  [ -f "$parts" ] && [ -f "$shp" ] && { echo "skip $rid"; return; }
  mkdir -p "$rd" "$data"
  { echo "EVENTS: $ev"; echo "EVENT_OUTPUT: HepMC3_GenEvent[events]"; echo "";
    echo "BEAMS: [11, -11]"; echo "BEAM_ENERGIES: 45.6"; echo "";
    echo "ALPHAS(MZ): $a"; echo "ORDER_ALPHAS: 1"; echo "";
    echo "SHOWER_GENERATOR: $shower"; echo "FRAGMENTATION: Ahadic";
    if [ -n "$bf" ]; then echo "AHADIC:"; echo "  BARYON_FRACTION: $bf"; fi
    echo ""; echo "PROCESSES:"; echo "- 11 -11 -> 93 93:"; echo "    Order: {QCD: 0, EW: 2}";
  } > "$rd/Sherpa.yaml"
  ( cd "$rd" && "$SHERPA3" ) >/dev/null 2>&1 || { echo "FAIL $rid"; return; }
  for cand in "$rd/events.hepmc3" "$rd/events" "$rd/events.hepmc"; do
    [ -f "$cand" ] && [ "$cand" != "$rd/events.hepmc" ] && { mv "$cand" "$rd/events.hepmc"; break; }; done
  [ -f "$rd/events.hepmc" ] || { echo "no hepmc $rid"; return; }
  "$PY" "$SD/$extr" --input "$rd/events.hepmc" --output "$parts" >/dev/null 2>&1
  "$PY" "$SD/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$data" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid ($shower a=$a)"
}
# Dire pseudo-data, default hadronization, flavor extraction
gen 8201 0.118 Dire "$SD/data_maxent" extract_particles_full_flavor.py "" 80000
# Stage A margin runs, CSS, kinematic full extraction
gen 6113 0.10667 CSS "$SD/data_stageA_full" extract_particles_full.py "" 80000
gen 6114 0.13333 CSS "$SD/data_stageA_full" extract_particles_full.py "" 80000
echo "PRESENT GENERATION DONE"