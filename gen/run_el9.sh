#!/bin/bash
# Run one published generator script (gen_stageE.sh = Sherpa, gen_stageF.sh = Herwig) on Symmetry:
# mount CVMFS without root (cvmfsexec), enter the EL9 container (the userspace the CVMFS builds
# need), and run the script from gen/ with its output under $GEN_BASE (default gen/).
#   usage: run_el9.sh gen_stageE.sh <rid> <8 parameters> <events>
set -eo pipefail   # no -u: the module scripts reference unset variables
R=/mnt/beegfs/bassi/reweighting
GEN_BASE=${GEN_BASE:-$R/gen}
CVMFSEXEC=$R/tools/cvmfsexec/cvmfsexec
SIF=$R/containers/almalinux9.sif
script=$1; shift
[ -x "$R/gen/$script" ] || { echo "run_el9.sh: no script $R/gen/$script"; exit 2; }
source /etc/profile.d/modules.sh 2>/dev/null || true
module load apptainer >/dev/null 2>&1 || true
mkdir -p "$GEN_BASE"
exec "$CVMFSEXEC" sft.cern.ch -- apptainer exec --cleanenv -B /cvmfs -B "$R" "$SIF" \
  env STAGEE_BASE="$GEN_BASE" STAGEF_BASE="$GEN_BASE" bash -c "cd '$GEN_BASE' && '$R/gen/$script' $*"
