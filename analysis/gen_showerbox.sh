#!/usr/bin/env bash
# A FULL Stage C box per parton shower: the same 27-point grid plus 5 held-out interior
# points in (alpha_s, STRANGE_FRACTION, KT_0), generated separately for CSS, Dire and Alaric.
#
# All three boxes use the SAME alaric-merge binary. The existing Stage C box is Sherpa 3.0.4
# and Alaric only exists on the development branch, so reusing the old CSS box would confound
# a shower change with a generator-version change. That is why CSS is regenerated here rather
# than reused (the same reasoning already written into gen_stageD3.sh).
#
# Because the box varies the Ahadic parameters, each shower's own preferred hadronization
# point lies INSIDE its own box, so the joint shower-plus-hadronization retune that the paper
# currently lists as a missing prerequisite becomes something the conditional model can locate
# rather than a separate exercise.
#
# rids: CSS 8300-8331, Dire 8400-8431, Alaric 8500-8531.
set -uo pipefail
PY=python
SHERPA=${SHERPA31_PREFIX}/bin/Sherpa
SD=.; RUNS="$SD/runs_showerbox"; DATA="$SD/data_showerbox"
mkdir -p "$RUNS" "$DATA"
EV=${EV:-40000}
NPAR=${NPAR:-5}          # leave cores free for the EERAD3 jobs

gen () { local rid=$1 shower=$2 a=$3 sf=$4 kt=$5 rd="$RUNS/run_$rid"
  local full="$DATA/particles_full_$rid.npz" shp="$DATA/shapes_run_$rid.csv"
  [ -f "$full" ] && [ -f "$shp" ] && { echo "skip $rid"; return; }
  mkdir -p "$rd"
  cat > "$rd/Sherpa.yaml" <<YAML
EVENTS: $EV
EVENT_OUTPUT: HepMC3_GenEvent[events]

BEAMS: [11, -11]
BEAM_ENERGIES: 45.6

ALPHAS(MZ): $a
ORDER_ALPHAS: 1

SHOWER_GENERATOR: $shower
FRAGMENTATION: Ahadic
AHADIC:
  STRANGE_FRACTION: $sf
  KT_0: $kt

PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
YAML
  ( cd "$rd" && "$SHERPA" ) > "$rd/sherpa.log" 2>&1 || { echo "FAIL $rid ($shower)"; return; }
  grep -q "Jet_Evolution:$shower" "$rd/sherpa.log" || { echo "WRONG SHOWER $rid (wanted $shower)"; return; }
  for c in "$rd/events.hepmc3" "$rd/events" "$rd/events.hepmc"; do
    [ -f "$c" ] && [ "$c" != "$rd/events.hepmc" ] && { mv "$c" "$rd/events.hepmc"; break; }; done
  [ -f "$rd/events.hepmc" ] || { echo "no hepmc $rid"; return; }
  "$PY" "$SD/extract_particles_full_flavor.py" --input "$rd/events.hepmc" --output "$full" >/dev/null 2>&1
  "$PY" "$SD/compute_shapes_only.py" --input "$rd/events.hepmc" --output-dir "$DATA" --run-id "$rid" >/dev/null 2>&1
  rm -f "$rd/events.hepmc"; echo "done $rid ($shower a=$a sf=$sf kt=$kt)"
}

ASV=(0.112 0.120 0.128); SFV=(0.30 0.46 0.65); KTV=(0.80 1.21 1.80)
HELD=("0.116 0.38 1.00" "0.124 0.55 1.50" "0.120 0.46 1.21" "0.114 0.60 0.95" "0.126 0.35 1.60")
i=0
for sh_base in "CSS 8300" "Dire 8400" "Alaric 8500"; do
  set -- $sh_base; shower=$1; base=$2; rid=$base
  for a in "${ASV[@]}"; do for s in "${SFV[@]}"; do for k in "${KTV[@]}"; do
    gen $rid $shower $a $s $k & rid=$((rid+1)); i=$((i+1))
    (( i % NPAR == 0 )) && wait
  done; done; done
  for h in "${HELD[@]}"; do set -- $h
    gen $rid $shower $1 $2 $3 & rid=$((rid+1)); i=$((i+1))
    (( i % NPAR == 0 )) && wait
  done
done
wait
echo "SHOWERBOX GENERATION DONE ($i runs, $EV events each)"
