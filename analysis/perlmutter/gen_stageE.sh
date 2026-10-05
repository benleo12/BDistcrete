#!/bin/bash
# One Stage E run: write the card, run Sherpa 3.0.4 from CVMFS, extract, drop the HepMC.
# usage: gen_stageE.sh <rid> <alphas> <pt_max> <alpha_l> <gamma_l> <strange> <baryon> <alpha_g> <beta_l> <events>
set -e
rid=$1; as=$2; ptmax=$3; al=$4; gl=$5; sf=$6; bf=$7; ag=$8; bl=$9; ev=${10}
# Argument sanity, so a column-misalignment fails loudly instead of running for hours.
# (An earlier version lost one column to a bad `shift`, putting the event count into BETA_L,
#  which made cluster fragmentation grind at 4 events per 90 minutes with no error message.)
[ $# -eq 10 ] || { echo "BAD ARGC $# (want 10) for rid=$rid"; exit 2; }
chk() { awk -v v="$2" -v lo="$3" -v hi="$4" -v n="$1" \
  'BEGIN{ if (v+0<lo || v+0>hi) { printf "BAD %s=%s, expected [%g,%g]\n", n, v, lo, hi; exit 1 } }' || exit 2; }
chk ALPHAS "$as" 0.05 0.25 ; chk PT_MAX "$ptmax" 0.1 3.5 ; chk ALPHA_L "$al" 0.5 8
chk GAMMA_L "$gl" 0.05 2  ; chk STRANGE "$sf" 0.05 0.95 ; chk BARYON "$bf" 0.01 0.8
chk ALPHA_G "$ag" 0.1 3   ; chk BETA_L "$bl" 0.001 1.5  ; chk EVENTS "$ev" 100 5000000
S=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/sherpa/3.0.4-22c71/x86_64-el9-gcc13-opt
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt
source $V/setup.sh 2>/dev/null
export LD_LIBRARY_PATH=$S/lib64/SHERPA-MC:$S/lib64:$S/lib:$LD_LIBRARY_PATH
export PATH=$S/bin:$PATH
BASE=$SCRATCH/stageE; DATA=$BASE/data; rd=$BASE/runs/run_$rid
mkdir -p "$rd" "$DATA"
[ -d "$BASE/Process" ] && cp -r "$BASE/Process" "$rd/" 2>/dev/null || true
cd "$rd"
cat > Sherpa.yaml <<YAML
EVENTS: $ev
EVENT_OUTPUT: HepMC3_GenEvent[events]
RANDOM_SEED: $rid
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
ALPHAS(MZ): $as
ORDER_ALPHAS: 1
SHOWER_GENERATOR: CSS
FRAGMENTATION: Ahadic
AHADIC:
  PT_MAX: $ptmax
  ALPHA_L: $al
  GAMMA_L: $gl
  STRANGE_FRACTION: $sf
  BARYON_FRACTION: $bf
  ALPHA_G: $ag
  BETA_L: $bl
PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
YAML
Sherpa > sherpa.log 2>&1 || { echo "SHERPA FAIL $rid"; exit 1; }
for c in events.hepmc3 events events.hepmc; do
  [ -f "$c" ] && [ "$c" != "events.hepmc" ] && mv "$c" events.hepmc && break
done
[ -f events.hepmc ] || { echo "NO HEPMC $rid"; exit 1; }
python3 $BASE/code/extract_particles_full_flavor.py --input events.hepmc --output $DATA/particles_full_${rid}.npz > /dev/null 2>&1
python3 $BASE/code/compute_shapes_only.py --input events.hepmc --output-dir $DATA --run-id $rid > /dev/null 2>&1
rm -f events.hepmc
rm -rf Process Results.zip Results.zip~ 2>/dev/null
echo "done $rid (as=$as ptmax=$ptmax al=$al gl=$gl sf=$sf bf=$bf ag=$ag bl=$bl)"
