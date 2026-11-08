#!/bin/bash
#SBATCH -p preq
#SBATCH -J lmx_nb3_finish
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=180G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_phaseB/observables_v3/logs/finish_%j.out
# After nb3_array.sbatch. STEPS (default "checks refresh rule verify"; '+' also separates) selects the parts:
#   checks   every unit done, no pending record; the temp files of nb3_units.py removed (only directly
#            inside the folders of NB_FOLDERS.txt, only names .shapes_run_*.csv.nb3tmp.*); the folder check
#            (nb3_folders.py); Phase A's link check (nb_links.py verify) and merged/truncated check
#            (nb_consist.py); the spot check (nb3_spot.py, every rewritten unit first); the independent
#            recomputation (indep_check3.py); the summary (nb3_summary.py); no original file changed since
#            units.tsv was copied; gate.py
#   refresh  only before the rerun (RUN/output/.lmx_rerun_start must not exist) and with RUN's
#            beam_photons.py equal to this package's (sync_run.sh copies it): stray temp files of an
#            interrupted refresh removed (*.npz.tmp<pid>.npz, MANIFEST.tsv.tmp<pid>, bundle_rule.json.tmp<pid>
#            directly in RUN/output, RUN/output/models, lmx_runs/release_ref); delta_caches.py snap (v2 values; stops
#            if a file is not B0's), refresh3_bundles.py all (B0's refresh from the originals and the v3 _nb
#            CSVs), delta_caches.py check (only events whose mask changed may change), delta_caches.py
#            manifest
#   rule     only before the rerun, like refresh: delta_caches.py rule: RUN/output/bundle_rule.json (I1), left as
#            it is when it already holds the same rule and bundles (every scoring's fingerprint holds its md5)
#   verify   verify_refresh3.py (B0's read-back, and bundle_rule.json), check_paths3.py --hash
# Every part can be repeated before the rerun starts: a requeue repeats it (the snapshot is taken once). Once
# RUN/output/.lmx_rerun_start exists, refresh and rule stop (exit 8): the rerun reads what they write.
set -u
W=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables_v3
B0=/mnt/beegfs/bassi/reweighting/lmx_phaseB/observables
RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
LR=/mnt/beegfs/bassi/reweighting/lmx_runs/release_ref
P="/mnt/beegfs/bassi/reweighting/env/torch/bin/python -B -u"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PLAN=$B0/results/layout_plan.tsv
STEPS=${STEPS:-"checks refresh rule verify"}
STEPS=" ${STEPS//+/ } "
cd $W || exit 1
echo "=== finish on $(hostname) $(date) restart ${SLURM_RESTART_COUNT:-0} STEPS=$STEPS"
n=$(ls $W/units | wc -l); want=$(($(wc -l < $W/units.tsv) - 1)); echo "unit records: $n of $want"
[ "$n" -eq "$want" ] || { echo "INCOMPLETE: $n of $want units"; exit 2; }
[ -z "$(ls -A $W/pending 2>/dev/null)" ] || { echo "PENDING RECORDS LEFT"; ls $W/pending; exit 2; }
if [[ "$STEPS" == *" checks "* ]]; then
  echo "--- temp files of nb3_units.py left in the folders of NB_FOLDERS.txt (removed):"
  grep -v '^#' scripts/NB_FOLDERS.txt | while read -r d; do
    [ -n "$d" ] && [ -d "$d" ] && find "$d" -maxdepth 1 -type f -name '.shapes_run_*.csv.nb3tmp.*' -print -delete
  done > $W/results/nb3tmp_removed.txt; wc -l < $W/results/nb3tmp_removed.txt
  echo "--- folder check"
  rm -f $W/results/nb_folders.json                  # the test job's; gate.py needs this job's
  $P scripts/nb3_folders.py $W; echo "folders rc $?"
  echo "--- nb_links.py verify (Phase A's check, on the v3 unit records)"
  $P scripts/nb_links.py $W verify | tail -3; echo "verify rc ${PIPESTATUS[0]}"
  echo "--- nb_consist.py (merged and truncated sets)"
  $P scripts/nb_consist.py $W 64; echo "consist rc $?"
  echo "--- spot checks"
  $P scripts/nb3_spot.py $W 24 2026 | tail -2; echo "spot rc ${PIPESTATUS[0]}"
  echo "--- independent recomputation"
  $P scripts/indep_check3.py $W 64 300 11 64 | tail -40
  echo "--- summary"
  $P scripts/nb3_summary.py $W > /dev/null && grep -A6 "^== ALL" $W/results/nb3_summary.txt && \
    sed -n '/^== rule (ii) in pair/,$p' $W/results/nb3_summary.txt
  echo "--- files in the original folders changed since units.tsv was copied (must be none):"
  find /mnt/beegfs/bassi/reweighting/gen/data_release /mnt/beegfs/bassi/reweighting/gen/data_v3 /mnt/beegfs/bassi/reweighting/gen/data_abc \
       /mnt/beegfs/bassi/reweighting/gen/data_abc_blind /mnt/beegfs/bassi/reweighting/gen/data_widebox /mnt/beegfs/bassi/reweighting/gen/data_ext \
       /mnt/beegfs/bassi/reweighting/gen/data_abc_ext /mnt/beegfs/bassi/reweighting/gen/data_trainext /mnt/beegfs/bassi/reweighting/gen/data_release_orig_held \
       /mnt/beegfs/bassi/reweighting/maxent_code/data_stagePURE17* /mnt/beegfs/bassi/reweighting/maxent_code/data_abc_h400 \
       /mnt/beegfs/bassi/reweighting/maxent_code/data_abc_blind /mnt/beegfs/bassi/reweighting/maxent_code/output \
       -newer $W/units.tsv ! -path "*_nb*" 2>/dev/null | head -20
  $P scripts/gate.py $W || { echo "CHECKS FAILED: no refresh"; exit 3; }
fi
if [[ "$STEPS" == *" refresh "* ]]; then
  if [ -e $RUN/output/.lmx_rerun_start ]; then
    echo "RERUN STARTED ($RUN/output/.lmx_rerun_start exists): the caches are not refreshed under it. Ask first."
    exit 8
  fi
  a=$(md5sum < scripts/beam_photons.py | cut -d' ' -f1); b=$(md5sum < $RUN/beam_photons.py 2>/dev/null | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "RUN's beam_photons.py (md5 ${b:-none}) is not this package's ($a): copy the code into RUN first (sync_run.sh)"; exit 9; }
  echo "--- stray temp files of an interrupted refresh (removed) $(date)"
  { find $RUN/output $RUN/output/models -maxdepth 1 -type f \( -name '*.npz.tmp[0-9]*.npz' -o -name 'MANIFEST.tsv.tmp[0-9]*' \
         -o -name 'bundle_rule.json.tmp[0-9]*' \) -print -delete
    find $LR -maxdepth 1 -type f -name '*.npz.tmp[0-9]*.npz' -print -delete; } > $W/results/stray_tmp_removed.txt
  wc -l < $W/results/stray_tmp_removed.txt
  echo "--- snapshot of the v2 caches $(date)"
  $P scripts/delta_caches.py snap $PLAN || exit 4
  echo "--- refresh3_bundles.py $(date)"
  /usr/bin/time -v $P scripts/refresh3_bundles.py ${REFRESH_JOBS:-all} 64 2>&1 | grep -v "^\s*\(Page\|Volunt\|Invol\|Swaps\|File sys\|Socket\|Signals\|Average\|Exit\|Minor\|Major\)"
  rc=${PIPESTATUS[0]}; echo "refresh rc $rc $(date)"; [ "$rc" -eq 0 ] || exit 5
  echo "--- delta check $(date)"
  $P scripts/delta_caches.py check $PLAN | tail -30; rc=${PIPESTATUS[0]}; echo "delta rc $rc"; [ "$rc" -eq 0 ] || exit 6
  $P scripts/delta_caches.py manifest $PLAN || exit 7
fi
if [[ "$STEPS" == *" rule "* ]]; then
  if [ -e $RUN/output/.lmx_rerun_start ]; then
    echo "RERUN STARTED ($RUN/output/.lmx_rerun_start exists): bundle_rule.json is part of every scoring's fingerprint, so it is not written under it. Ask first."
    exit 8
  fi
  echo "--- bundle_rule.json $(date)"
  $P scripts/delta_caches.py rule $PLAN || exit 10
fi
if [[ "$STEPS" == *" verify "* ]]; then
  echo "--- read-back check of the refreshed files and bundle_rule.json $(date)"
  $P scripts/verify_refresh3.py $PLAN ${VERIFY_NRUN:-60} 5 16 | tail -31; echo "verify_refresh rc ${PIPESTATUS[0]}"
  echo "--- check_paths3.py --hash $(date)"
  $P scripts/check_paths3.py $B0/results/data_tokens.tsv --hash | tail -31; echo "check_paths rc ${PIPESTATUS[0]}"
fi
echo "=== finish end $(date)"
