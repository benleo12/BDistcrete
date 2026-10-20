#!/bin/bash
# Set up Herwig or Sherpa from CVMFS with the builds this sample was generated with.
#
#   source cvmfs_env.sh herwig
#   source cvmfs_env.sh sherpa
#
# Run it inside an el9 container (see README.md). On a host that is not el9 it will appear to
# work and then fail on a library version, which is what the original Perlmutter scripts worked
# around with a readline shim.
#
# The build hashes below are PINNED. CVMFS carries 88 separate builds of Herwig 7.3.0p1, and the
# original script selected one with `ls -d .../7.3.0p1-*/... | head -1`, which is not a pin: a
# build whose hash sorts earlier would silently replace it.
#
# The hash pinned here is NOT the one that glob resolved to. The released sample was generated
# with 18d2e, which wants GSL 2.7 and took it from the host while LCG's 2.8 was also loaded.
# c7a43 is the build the LCG_109 view itself selects and it loads GSL 2.8 alone. The two were
# checked against each other rather than assumed equivalent: 2000 events on the Stage F card at
# the box centre with seed 8000 give byte-identical HepMC. See README.md.
LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt
HERWIG_BUILD=7.3.0p1-c7a43
SHERPA_BUILD=3.0.4-22c71

[ -d /cvmfs/sft.cern.ch ] || { echo "cvmfs_env.sh: /cvmfs/sft.cern.ch is not mounted"; return 1 2>/dev/null || exit 1; }
[ -d "$LCG_VIEW" ] || { echo "cvmfs_env.sh: no LCG view at $LCG_VIEW"; return 1 2>/dev/null || exit 1; }

# Source the view FIRST and prepend to what it sets. The view puts ThePEG on LD_LIBRARY_PATH,
# and replacing the variable instead of prepending leaves Herwig unable to find libThePEG.
source "$LCG_VIEW/setup.sh" 2>/dev/null
_VIEW_LLP=$LD_LIBRARY_PATH

case "${1:-}" in
  herwig)
    H=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/herwig++/$HERWIG_BUILD/x86_64-el9-gcc13-opt
    [ -d "$H" ] || { echo "cvmfs_env.sh: no Herwig build at $H"; return 1 2>/dev/null || exit 1; }
    export LD_LIBRARY_PATH=$H/lib/Herwig:$H/lib:$LCG_VIEW/lib64:$LCG_VIEW/lib:$_VIEW_LLP
    export PATH=$H/bin:$PATH
    # The CVMFS build hardcodes its Jenkins build path for HerwigDefaults.rpo, so every Herwig
    # call needs --repo pointing at the real one. HERWIG_REPO is that path.
    export HERWIGPATH=$H/share/Herwig
    export HERWIG_REPO=$HERWIGPATH/HerwigDefaults.rpo
    ;;
  sherpa)
    S=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/sherpa/$SHERPA_BUILD/x86_64-el9-gcc13-opt
    [ -d "$S" ] || { echo "cvmfs_env.sh: no Sherpa build at $S"; return 1 2>/dev/null || exit 1; }
    export LD_LIBRARY_PATH=$S/lib64/SHERPA-MC:$S/lib64:$S/lib:$_VIEW_LLP
    export PATH=$S/bin:$PATH
    ;;
  *)
    echo "usage: source cvmfs_env.sh {herwig|sherpa}"; return 2 2>/dev/null || exit 2 ;;
esac

# Fail now rather than after the integration step.
_missing=$(ldd "$(command -v "$([ "$1" = herwig ] && echo Herwig || echo Sherpa)")" 2>/dev/null | grep -c 'not found')
[ "${_missing:-0}" -eq 0 ] || { echo "cvmfs_env.sh: $_missing unresolved libraries, are you in an el9 container?"; ldd "$(command -v "$([ "$1" = herwig ] && echo Herwig || echo Sherpa)")" | grep 'not found'; return 1 2>/dev/null || exit 1; }
echo "cvmfs_env.sh: $1 ready, $(  [ "$1" = herwig ] && Herwig --version 2>/dev/null | head -1 || Sherpa --version 2>/dev/null | head -1 )"
