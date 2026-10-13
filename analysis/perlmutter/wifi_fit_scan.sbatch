#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 03:00:00
#SBATCH -J wifi_scan
#SBATCH -o logs/wifi_scan_%j.out
#SBATCH -e logs/wifi_scan_%j.err
# The WiFi fit and its out-of-sample diagnostics for the larger ensembles (wifi_fit.py, wifi_diag.py),
# which need tens of GB at thirty-two members and therefore run on a CPU node. TAGS lists the exports.
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export OMP_NUM_THREADS=64
for TAG in ${TAGS:-MIXGEO16 MIXGEO32}; do
  echo "=== $TAG fit $(date)"
  python3 -u wifi_fit.py output/wifi_${TAG}_basis.npz output/models/${TAG}_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "FIT FAILED $TAG"
  echo "=== $TAG diag $(date)"
  python3 -u wifi_diag.py output/wifi_${TAG}_basis.npz output/models/${TAG}_cond.npz output/models/${TAG}_ref_v2_slim.npz || echo "DIAG FAILED $TAG"
done
python3 wifi_scan_table.py MIXGEO MIXGEO8 MIXGEO16 MIXGEO32
echo "WIFI SCAN DONE $(date)"
