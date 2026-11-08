#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J lmx_wificov
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# wifi_cov.sbatch of the rerun (Fig. 5, Table 5, \Cov*, \CovEight*, \Wifi*): for one trained mixture,
# every step runs, whatever exists (nothing is skipped; the outputs below are removed when the job starts
# and recorded when it ends well, output/lmx_records/wifi_<TAG>.json):
#   1. the basis with photon-free held-out observables: the published basis functions (they depend on the
#      network inputs only, unchanged by D4) and the held-out observables recomputed (wifi_embed.py
#      WIFI_HELD_ONLY)                                  -> output/wifi_<TAG>_basis_rebuilt.npz
#      (the observables task's refreshed basis output/wifi_<TAG>_basis.npz, if there, must be listed in
#      output/bundle_rule.json and agree with this rebuild array by array, else the step stops)
#   2. the fitted basis weights and their covariance (wifi_fit.py), at the refitted temperature
#                                                       -> output/wifi_<TAG>_fit.{npz,json}
#   3. their diagnostics against the mean of members out of sample (wifi_diag.py) -> output/wifi_<TAG>_diag.json
#   4. the coverage at the 48 strict held-out points (wifi_coverage.py)            -> output/wifi_coverage_<TAG>.json
#   5. for MIXGEOV3S9 also the split of the loss gain by generator (wifi_gensplit.py) -> output/wifi_gensplit_<TAG>.json
#   sbatch --export=ALL,MAXENT_DIR=$RUN,TAG=MIXGEOV3S9 symmetry/rerun/wifi_cov_lmx.sbatch   (and s1, s2, MIXGEOV3M8S9)
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
: "${TAG:?}"
need_nb data_stagePURE17_v3 data_stagePURE17_v3_h400 || exit 2
M=output/models/$TAG; REF=${M}_ref_v2_slim.npz; REFRESHED=output/wifi_${TAG}_basis.npz; BAS=output/wifi_${TAG}_basis_rebuilt.npz
OUTS="$BAS output/wifi_${TAG}_fit.npz output/wifi_${TAG}_fit.json output/wifi_${TAG}_diag.json output/wifi_coverage_${TAG}.json"
[ "$TAG" = MIXGEOV3S9 ] && OUTS="$OUTS output/wifi_gensplit_${TAG}.json"
rm -f $OUTS output/lmx_records/wifi_${TAG}.json
need_file ${M}_cond.npz ${M}_trunk.pt $REF output/published/wifi_${TAG}_basis.npz output/thrust_targets_nlo_point.npz \
          output/published/models/${TAG}_ref.npz output/published/models/${TAG}_ref_v2_slim.npz output/temperature/$TAG.json || exit 2
need_rule $REF || exit 2
export OMP_NUM_THREADS=16 TORCH_THREADS=16 REF_PARTICLES=output/published/models/${TAG}_ref.npz REF_ORIG=output/published/models/${TAG}_ref_v2_slim.npz
echo "=== $TAG basis with photon-free held-out observables $(date)"
DM_DATA=data_stagePURE17_v3 WIFI_HELD_ONLY=output/published/wifi_${TAG}_basis.npz WIFI_OUT=$BAS \
  "$PY" -B -u wifi_embed.py MIX ${M}_cond.npz ${M}_trunk.pt || { echo "WIFI BASIS FAILED"; exit 3; }
if [ -e $REFRESHED ]; then
  # the observables task's refreshed basis must hold the same arrays as this rebuild, which the steps below read
  need_rule $REFRESHED || exit 2
  "$PY" -B - $REFRESHED $BAS output/published/wifi_${TAG}_basis.npz <<'PY' || { echo "WIFI BASIS DIFFERS FROM THE REFRESHED ONE"; exit 3; }
import sys, numpy as np
a, b, p = (np.load(f) for f in sys.argv[1:4])
keys = [k for k in p.files]                       # every array of the published basis
bad = [k for k in keys if k not in a.files or not np.array_equal(a[k], b[k])]
held = [k for k in keys if k.startswith('held_') and not k.startswith('held_f_') and k not in ('held_rids', 'held_theta')]
same_pub = [k for k in held if np.array_equal(b[k], p[k])]
print(f'{sys.argv[1]}: {len(keys) - len(bad)} of {len(keys)} arrays equal to the rebuilt basis; held-out observables '
      f'unchanged from the published basis in {len(same_pub)} of {len(held)} (the baryon count does not see the beam photons)')
sys.exit(1 if bad else 0)
PY
fi
echo "=== $TAG fit $(date)"
WIFI_FIT_OUT=output/wifi_${TAG}_fit "$PY" -B -u wifi_fit.py $BAS ${M}_cond.npz $REF || { echo "WIFI FIT FAILED"; exit 3; }
echo "=== $TAG diagnostics $(date)"
WIFI_FIT=output/wifi_${TAG}_fit.npz WIFI_DIAG_OUT=output/wifi_${TAG}_diag.json "$PY" -B -u wifi_diag.py $BAS ${M}_cond.npz $REF || { echo "WIFI DIAG FAILED"; exit 3; }
echo "=== $TAG coverage $(date)"
COVERAGE_OUT=output/wifi_coverage_${TAG}.json "$PY" -B -u wifi_coverage.py MIX ${M}_cond.npz $REF output/wifi_${TAG}_fit.npz \
  data_stagePURE17_v3_h400 output/thrust_targets_nlo_point.npz || { echo "COVERAGE FAILED"; exit 3; }
if [ "$TAG" = MIXGEOV3S9 ]; then
  GENSPLIT_OUT=output/wifi_gensplit_${TAG}.json "$PY" -B -u wifi_gensplit.py $BAS output/wifi_${TAG}_fit.npz || { echo "GENSPLIT FAILED"; exit 3; }
fi
step_record wifi_$TAG $OUTS || exit 3
echo "WIFICOV $TAG DONE $(date)"
