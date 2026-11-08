#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb3_test
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseB/observables_v3/logs/test_%j.out
# Test of nb3_units.py on the units in UIDS (default: the six units that hold the events the B0 skeptic
# named). First the folder check (nb3_folders.py: NB_FOLDERS.txt is what nb_links.py sees, and every
# unit's _nb CSV lies directly inside a listed folder). Then the delta from the current (v2) _nb CSV and
# the full recomputation from the original CSV (NB_FULL=1), both into WORK/test, nothing replaced; they
# must agree (nb3_equal.py: the changed rows byte for byte, the others up to rounding of 1e-14 in the
# three shapes). Then the spot check on the test outputs and the named events by hand (show_events.py).
# Exit 1 if the folder check fails or a pair differs.
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables_v3
P="/mnt/beegfs/bassi/reweighting/env/torch/bin/python -B -u"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
UIDS=${UIDS:-data_release__8001,data_release__8003,data_release__8206,data_trainext_k1__8445,data_trainext_k2__8485,data_abc__30021}
cd $W || exit 1
echo "=== test on $(hostname) $(date) UIDS=$UIDS"
$P scripts/nb3_folders.py $W || { echo "FOLDER CHECK FAILED"; exit 1; }
$P scripts/nb3_units.py $W 6 test "$UIDS"
NB_FULL=1 $P scripts/nb3_units.py $W 6 test "$UIDS"
$P scripts/nb3_equal.py $W "$UIDS"; bad=$?
$P scripts/nb3_spot.py $W 6 1 $W/test | tail -1
$P scripts/show_events.py $W data_trainext_k1__8445:43546 data_release__8003:21629 data_abc__30021:42599 \
   data_trainext_k2__8485:24784 data_release__8206:68028 data_release__8001:20153 data_release__8001:53717 \
   data_release__8001:78531 data_release__8001:12668
echo "=== end $(date) bad=$bad"
exit $bad
