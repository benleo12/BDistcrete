#!/usr/bin/env bash
# R1: wide-box regeneration on the FROZEN pipeline (Sherpa 3, full-event representation).
# One grid, three later analyses (pool-all / local-union / single-centre; the centre run
# alpha_s=0.140 is grid point rid 9010). 21 training runs over [0.08,0.20] (spacing 0.006)
# + 6 held-out off-grid, 80k events each. Guarded against any other Sherpa on the machine.
set -uo pipefail
PY=python
SHERPA3=${SHERPA3_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_widebox"; DATA="$SD/data_widebox"; mkdir -p "$RUNS" "$DATA"
clear=0
for i in $(seq 1 240); do
  if pgrep -f "bin/Sherpa" >/dev/null 2>&1; then clear=0; else clear=$((clear+1)); fi
  [ "$clear" -ge 2 ] && break
  sleep 30
done
gen () { local rid=$1 a=$2 ev=$3 rd="$RUNS/run_$1"
  local parts="$DATA/particles_full_$1.npz" shp="$DATA/shapes_run_$1.csv"
  [ -f "$parts" ] && [ -f "$shp" ] && { echo "skip $rid"; return; }
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
  "$PY" "$SD/extract_particles_full.py" --input "$rd/events.hepmc" --output "$parts" >/dev/null 2>&1
  "$PY" "$SD/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$DATA" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid (a=$a)"
}
ASV=(0.0800 0.0860 0.0920 0.0980 0.1040 0.1100 0.1160 0.1220 0.1280 0.1340 0.1400 0.1460 0.1520 0.1580 0.1640 0.1700 0.1760 0.1820 0.1880 0.1940 0.2000)
rid=9000; for a in "${ASV[@]}"; do gen $rid $a 80000; rid=$((rid+1)); done
gen 9100 0.083 80000; gen 9101 0.107 80000; gen 9102 0.131 80000
gen 9103 0.155 80000; gen 9104 0.179 80000; gen 9105 0.197 80000
echo "WIDEBOX GENERATION DONE"