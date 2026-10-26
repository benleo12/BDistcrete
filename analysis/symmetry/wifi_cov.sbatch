#!/bin/bash
#SBATCH -p sharedq
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 12:00:00
#SBATCH -J wificov
#SBATCH -o logs/wificov_%j.out
# The learning band of a trained mixture (wifi_fit.py, if not done yet), its diagnostics against the
# mean of members out of sample (wifi_diag.py, if not done yet), and the check that it covers the
# emulator's own error at the 48 held-out points with 400k events each (wifi_coverage.py).
#   TAG=MIXGEOV3S9 sbatch --export=ALL wifi_cov.sbatch
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 TORCH_THREADS=8
M=output/models/${TAG:?}
[ -f output/wifi_${TAG}_fit.npz ] || python3 -u wifi_fit.py output/wifi_${TAG}_basis.npz ${M}_cond.npz ${M}_ref_v2_slim.npz || { echo "WIFI FIT FAILED"; exit 3; }
[ -f output/wifi_${TAG}_diag.json ] || python3 -u wifi_diag.py output/wifi_${TAG}_basis.npz ${M}_cond.npz ${M}_ref_v2_slim.npz || echo "WIFI DIAG FAILED"
[ -f output/wifi_coverage_${TAG}.json ] || python3 -u wifi_coverage.py MIX ${M}_cond.npz ${M}_ref_v2_slim.npz output/wifi_${TAG}_fit.npz data_stagePURE17_v3_h400 output/thrust_targets_nlo_point.npz || { echo "COVERAGE FAILED"; exit 3; }
echo "WIFICOV $TAG DONE $(date)"
