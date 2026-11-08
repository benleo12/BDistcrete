#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb_finish
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=120G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseA/observables/logs/finish_%j.out
# After nb_array.sbatch: csv links of the stage mirrors, verification against the originals,
# consistency of the merged/truncated sets with their sources, and the summary. Idempotent.
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseA/observables
P=/mnt/beegfs/bassi/reweighting/env/torch/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/beegfs/bassi/reweighting/maxent_code || exit 1
echo "=== finish on $(hostname) $(date)"
n=$(ls $W/units | wc -l); want=$(($(wc -l < $W/units.tsv) - 1)); echo "unit records: $n of $want"
[ "$n" -eq "$want" ] || { echo "INCOMPLETE: $n of $want units, nothing linked"; exit 2; }
$P -u $W/scripts/nb_links.py $W csv || { echo "CSV LINKS FAILED"; exit 1; }
$P -u $W/scripts/nb_links.py $W verify; echo "verify rc $?"
$P -u $W/scripts/nb_consist.py $W 64; echo "consist rc $?"
$P -u $W/scripts/nb_summary.py $W > $W/summary.txt; echo "summary rc $?"
ls $W/tmp | wc -l | xargs echo "files left in tmp:"
echo "=== finish end $(date)"
