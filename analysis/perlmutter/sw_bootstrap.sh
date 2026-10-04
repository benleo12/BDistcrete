#!/bin/bash
# Bootstrap Sherpa builds on Perlmutter for the 1M anchor campaigns.
# Builds into ~/sw: HepMC3 3.2.7, Sherpa v3.0.4 (reference), Sherpa almerge
# c74190c5a (MEPS/alaric target). Logs in ~/sw/logs.
set -uo pipefail
mkdir -p ~/sw/logs ~/sw/src
cd ~/sw/src
module load cmake gcc-native/13 2>/dev/null || module load cmake gcc-native/12
echo "compilers: $(which gcc) $(gcc --version | head -1); $(which gfortran)"

# ---------------- HepMC3 (same 3.2.7 as the local Mac builds) ----------------
if [ ! -f ~/sw/hepmc3/lib64/libHepMC3.so ] && [ ! -f ~/sw/hepmc3/lib/libHepMC3.so ]; then
  curl -sL https://hepmc.web.cern.ch/hepmc/releases/HepMC3-3.2.7.tar.gz -o hepmc3.tgz
  tar xzf hepmc3.tgz
  cmake -S HepMC3-3.2.7 -B hepmc3-build \
    -DCMAKE_INSTALL_PREFIX=$HOME/sw/hepmc3 -DCMAKE_BUILD_TYPE=Release \
    -DHEPMC3_ENABLE_ROOTIO=OFF -DHEPMC3_ENABLE_PYTHON=OFF \
    > ~/sw/logs/hepmc3-cmake.log 2>&1
  cmake --build hepmc3-build -j 16 >> ~/sw/logs/hepmc3-cmake.log 2>&1
  cmake --install hepmc3-build >> ~/sw/logs/hepmc3-cmake.log 2>&1
fi
echo "HEPMC3 DONE $(ls ~/sw/hepmc3/lib*/libHepMC3.so 2>/dev/null | head -1)"

# ---------------- Sherpa sources: one clone, two checkouts ----------------
if [ ! -d ~/sw/src/sherpa/.git ]; then
  git clone https://gitlab.com/sherpa-team/sherpa.git sherpa > ~/sw/logs/clone.log 2>&1
fi
cd ~/sw/src/sherpa
git fetch --all --tags >> ~/sw/logs/clone.log 2>&1
[ -d ~/sw/src/sherpa-v304 ]   || git worktree add ~/sw/src/sherpa-v304 v3.0.4 >> ~/sw/logs/clone.log 2>&1
[ -d ~/sw/src/sherpa-alaric ] || git worktree add ~/sw/src/sherpa-alaric c74190c5a376d060fd62b5e6bca5a135065f2b2a >> ~/sw/logs/clone.log 2>&1
echo "SOURCES: v304=$(git -C ~/sw/src/sherpa-v304 rev-parse --short HEAD) alaric=$(git -C ~/sw/src/sherpa-alaric rev-parse --short HEAD)"

build_sherpa () {  # $1 src  $2 prefix  $3 log
  cmake -S "$1" -B "$1-build" \
    -DCMAKE_INSTALL_PREFIX="$2" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$HOME/sw/hepmc3 -DSHERPA_ENABLE_HEPMC3=ON \
    -DSHERPA_ENABLE_LHAPDF=OFF -DSHERPA_ENABLE_RIVET=OFF \
    > "$3" 2>&1 && \
  cmake --build "$1-build" -j 24 >> "$3" 2>&1 && \
  cmake --install "$1-build" >> "$3" 2>&1
}
build_sherpa ~/sw/src/sherpa-v304   ~/sw/sherpa-3.0.4  ~/sw/logs/sherpa-v304.log \
  && echo "SHERPA v3.0.4 DONE: $(~/sw/sherpa-3.0.4/bin/Sherpa-config --version 2>/dev/null)" \
  || echo "SHERPA v3.0.4 FAILED (see ~/sw/logs/sherpa-v304.log)"
build_sherpa ~/sw/src/sherpa-alaric ~/sw/sherpa-alaric ~/sw/logs/sherpa-alaric.log \
  && echo "SHERPA alaric DONE: $(~/sw/sherpa-alaric/bin/Sherpa-config --version 2>/dev/null)" \
  || echo "SHERPA alaric FAILED (see ~/sw/logs/sherpa-alaric.log)"
echo "BOOTSTRAP FINISHED"
