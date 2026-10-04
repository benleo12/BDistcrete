#!/bin/bash
# One Stage F run: Herwig 7.3.0p1 from CVMFS at the Z pole, extract, drop the HepMC.
# usage: gen_stageF.sh <rid> <alpha_fsr> <ptmin> <clmax> <clpow> <psplit> <pwtsq> <pwtdiq> <clsmr> <events>
set -e
rid=$1; af=$2; pt=$3; cm=$4; cp_=$5; ps=$6; sq=$7; dq=$8; sm=$9; ev=${10}
[ $# -eq 10 ] || { echo "BAD ARGC $# (want 10) for rid=$rid"; exit 2; }
chk() { awk -v v="$2" -v lo="$3" -v hi="$4" -v n="$1" \
  'BEGIN{ if (v+0<lo || v+0>hi) { printf "BAD %s=%s, expected [%g,%g]\n", n, v, lo, hi; exit 1 } }' || exit 2; }
# Herwig hard-limits AlphaQCDFSR:AlphaIn at 0.10.
chk ALPHA_FSR "$af" 0.10 0.2 ; chk PTMIN "$pt" 0.2 2.5 ; chk CLMAX "$cm" 1.5 7
chk CLPOW "$cp_" 0.3 5      ; chk PSPLIT "$ps" 0.3 2   ; chk PWTSQ "$sq" 0.05 1
chk PWTDIQ "$dq" 0.05 1     ; chk CLSMR "$sm" 0.1 2.5  ; chk EVENTS "$ev" 100 5000000
# Generator environment, pinned. Sourcing this rather than setting the paths here is what
# removes the readline shim the Perlmutter version needed: see README.md.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cvmfs_env.sh" herwig || exit 1
# Where runs and extracted data go. Defaults to the working directory so the script is not tied
# to one machine's scratch layout.
BASE=${STAGEF_BASE:-$PWD}; DATA=${STAGEF_DATA:-$BASE/data}; rd=$BASE/runs/run_$rid
mkdir -p "$rd" "$DATA"; cd "$rd"
cat > r$rid.in <<CARD
read snippets/EECollider.in
cd /Herwig/MatrixElements
insert SubProcess:MatrixElements 0 MEee2gZ2qq
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 91.2
set /Herwig/Shower/AlphaQCDFSR:AlphaIn $af
set /Herwig/Shower/PTCutOff:pTmin $pt*GeV
set /Herwig/Hadronization/ClusterFissioner:ClMaxLight $cm
set /Herwig/Hadronization/ClusterFissioner:ClPowLight $cp_
set /Herwig/Hadronization/ClusterFissioner:PSplitLight $ps
set /Herwig/Hadronization/HadronSelector:PwtSquark $sq
set /Herwig/Hadronization/HadronSelector:PwtDIquark $dq
set /Herwig/Hadronization/ClusterDecayer:ClSmrLight $sm
# same stable-particle convention as the Sherpa side: every hadron with c*tau > 10 mm undecayed
set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm
set /Herwig/Decays/DecayHandler:LifeTimeOption 0
read snippets/HepMC.in
set /Herwig/Analysis/HepMC:PrintEvent 100000000
set /Herwig/Analysis/HepMC:Filename events.hepmc
saverun r$rid EventGenerator
CARD
# the CVMFS build hardcodes its Jenkins build path for HerwigDefaults.rpo, so point --repo at the real one
Herwig read -i $HERWIGPATH --repo "$HERWIG_REPO" r$rid.in > read.log 2>&1 || { echo "READ FAIL $rid"; exit 1; }
Herwig run r$rid.run -N $ev --seed $rid --repo "$HERWIG_REPO" > run.log 2>&1 || { echo "RUN FAIL $rid"; exit 1; }
[ -f events.hepmc ] || { echo "NO HEPMC $rid"; exit 1; }
python3 "$HERE/extract_particles_full_flavor.py" --input events.hepmc --output $DATA/particles_full_${rid}.npz --fmt herwig > extract.log 2>&1 || { echo "EXTRACT FAIL $rid"; tail -5 extract.log; exit 1; }
python3 "$HERE/compute_shapes_only.py" --input events.hepmc --output-dir $DATA --run-id $rid --fmt herwig > shapes.log 2>&1 || { echo "SHAPES FAIL $rid"; tail -5 shapes.log; exit 1; }
rm -f events.hepmc r$rid.run
echo "done $rid (af=$af pt=$pt cm=$cm cp=$cp_ ps=$ps sq=$sq dq=$dq sm=$sm)"
