#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb2_finish
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=180G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseB/observables/logs/finish_%j.out
# After nb2_array.sbatch. STEPS (default "checks refresh layout") selects the parts:
#   checks   every unit done; no temp file left in an _nb folder; Phase A's link check (nb_links.py
#            verify) and merged/truncated check (nb_consist.py) on the new files; spot checks
#            (nb2_spot.py); the independent recomputation (indep_check.py); the summary; no original
#            file changed
#   refresh  refresh_bundles.py all (step 3)
#   layout   layout_plan.py, layout_make.py, check_paths.py (step 4)
# Every part is idempotent, so a requeue repeats it.
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables
P="/mnt/beegfs/bassi/reweighting/env/torch/bin/python -B -u"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
STEPS=${STEPS:-"checks refresh layout"}
cd $W || exit 1
echo "=== finish on $(hostname) $(date) restart ${SLURM_RESTART_COUNT:-0} STEPS=$STEPS"
n=$(ls $W/units | wc -l); want=$(($(wc -l < $W/units.tsv) - 1)); echo "unit records: $n of $want"
if [ -z "${ALLOW_INCOMPLETE:-}" ]; then   # ALLOW_INCOMPLETE=1: a refresh whose sources are already done
  [ "$n" -eq "$want" ] || { echo "INCOMPLETE: $n of $want units"; exit 2; }
  [ -z "$(ls -A $W/pending 2>/dev/null)" ] || { echo "PENDING RECORDS LEFT"; ls $W/pending; exit 2; }
fi
if [[ " $STEPS " == *" checks "* ]]; then
  echo "--- temp files of nb2_units.py left in _nb folders:"
  find /mnt/beegfs/bassi/reweighting/gen/data_*_nb /mnt/beegfs/bassi/reweighting/maxent_code/data_*_nb -maxdepth 2 -name ".*.nb2tmp.*" -print -delete | wc -l
  echo "--- nb_links.py verify (Phase A's check, on the new unit records)"
  $P scripts/nb_links.py $W verify | tail -3; echo "verify rc ${PIPESTATUS[0]}"
  echo "--- nb_consist.py (merged and truncated sets)"
  $P scripts/nb_consist.py $W 64; echo "consist rc $?"
  echo "--- spot checks"
  $P scripts/nb2_spot.py $W 24 2026 | tail -2; echo "spot rc ${PIPESTATUS[0]}"
  echo "--- independent recomputation"
  $P scripts/indep_check.py $W 64 300 11 64 | tail -40
  echo "--- summary"
  $P scripts/nb2_summary.py $W > /dev/null && head -3 $W/results/nb2_summary.txt
  echo "--- files in the original folders changed since 03:50 today (must be none):"
  find /mnt/beegfs/bassi/reweighting/gen/data_release /mnt/beegfs/bassi/reweighting/gen/data_v3 /mnt/beegfs/bassi/reweighting/gen/data_abc \
       /mnt/beegfs/bassi/reweighting/gen/data_abc_blind /mnt/beegfs/bassi/reweighting/gen/data_widebox /mnt/beegfs/bassi/reweighting/gen/data_ext \
       /mnt/beegfs/bassi/reweighting/gen/data_abc_ext /mnt/beegfs/bassi/reweighting/gen/data_trainext /mnt/beegfs/bassi/reweighting/gen/data_release_orig_held \
       /mnt/beegfs/bassi/reweighting/maxent_code/data_stagePURE17* /mnt/beegfs/bassi/reweighting/maxent_code/data_abc_h400 \
       /mnt/beegfs/bassi/reweighting/maxent_code/data_abc_blind /mnt/beegfs/bassi/reweighting/maxent_code/output \
       -newermt "2026-11-06 03:50" ! -path "*_nb*" 2>/dev/null | head -20
fi
if [[ " $STEPS " == *" refresh "* ]]; then
  echo "--- step 3: refresh_bundles.py $(date)"
  /usr/bin/time -v $P scripts/refresh_bundles.py ${REFRESH_JOBS:-all} 64 2>&1 | grep -v "^\s*\(Page\|Volunt\|Invol\|Swaps\|File sys\|Socket\|Signals\|Average\|Exit\|Minor\|Major\)"
  rc=${PIPESTATUS[0]}; echo "refresh rc $rc $(date)"; [ "$rc" -eq 0 ] || exit 3
  $P scripts/layout_plan.py $W/results/layout_plan.tsv
  echo "--- read-back check of the refreshed files $(date)"
  $P scripts/verify_refresh.py $W/results/layout_plan.tsv ${VERIFY_NRUN:-40} 5 16 | tail -40; echo "verify_refresh rc ${PIPESTATUS[0]}"
fi
if [[ " $STEPS " == *" layout "* ]]; then
  echo "--- step 4: layout $(date)"
  $P scripts/layout_plan.py $W/results/layout_plan.tsv
  $P scripts/layout_make.py $W/results/layout_plan.tsv 16; echo "layout_make rc $?"
  $P scripts/check_paths.py $W/results/data_tokens.tsv --hash | tail -40; echo "check_paths rc ${PIPESTATUS[0]}"
fi
echo "=== finish end $(date)"
