#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J refreshv2
#SBATCH -o logs/refreshv2_%j.out
#SBATCH -e logs/refreshv2_%j.err
# The release closure tables on the corrected (exact thrust axis) shapes. The v2 reference bundles
# replace the old ones (kept as *_ref.npz.v1) and the mixture scripts read the verified v2 mixture.
set -u
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
for t in E Fauglong MIX17aug; do
  [ -f output/models/${t}_ref_v2.npz ] || { echo "MISSING output/models/${t}_ref_v2.npz"; exit 2; }
  [ -f output/models/${t}_ref.npz.v1 ] || mv output/models/${t}_ref.npz output/models/${t}_ref.npz.v1
  cp output/models/${t}_ref_v2.npz output/models/${t}_ref.npz
done
ls -la output/models/*_ref.npz output/models/*_ref.npz.v1
export STAGEF_CSV=stageF_design_aug.csv DM_DATA=data_stageDM17aug_v2
echo "=== package $(date)";  python package_release.py E Fauglong=F MIX17aug=MIX17 || { echo PACKAGE FAILED; exit 2; }
# make_stage_table.py exits 2 when the manifest lists stages beyond these three (C_1M in the shipped
# one), since it would truncate STAGES.md. Set STAGE_TABLE_SUBSET=1 or point RELEASE_DIR at a scratch copy.
echo "=== stage table $(date)"; python make_stage_table.py E Fauglong=F MIX17aug=MIX17 || { echo TABLE FAILED; exit 3; }
echo "=== closure vs distance $(date)"; CLOSURE_DIST_OUT=output/closure_vs_distance.csv python closure_vs_distance.py E Fauglong=F MIX17aug=MIX17 || { echo CVD FAILED; exit 4; }
echo "=== accuracy law $(date)"; CLOSURE_DIST_IN=output/closure_vs_distance.csv python fit_accuracy_law.py || { echo LAW FAILED; exit 5; }
cp output/closure_vs_distance.csv release/accuracy_law_points.csv
echo "=== direction response $(date)"; python head_direction_response.py release/models E F MIX17 | tail -3
echo "ALL DONE $(date)"
