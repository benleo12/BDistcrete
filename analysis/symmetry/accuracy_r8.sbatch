#!/bin/bash
#SBATCH -p preq
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 1:00:00
#SBATCH -J acc_r8
#SBATCH -o logs/accuracy_r8_%j.out
# The accuracy law of the r8 release networks, from the closure tables table_r8.sbatch writes:
# per-point error against displacement (closure_vs_distance.py), then the robust floor-plus-slope
# fit (fit_accuracy_law.py), both into the scratch release directory.
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1; [ -f $R/env/torch/bin/activate ] && source $R/env/torch/bin/activate
export OMP_NUM_THREADS=4 STAGEE_CSV=stageE_design_v3all.csv STAGEF_CSV=stageF_design_v3all.csv DM_DATA=data_stagePURE17_v3_h400 STAGEE_DATA=data_stagePURE17_v3_h400 STAGEF_DATA=data_stagePURE17_v3_h400 RELEASE_DIR=release_r8_table
CLOSURE_DIST_OUT=release_r8_table/accuracy_law_points.csv python3 -u closure_vs_distance.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || { echo "ACC FAILED"; exit 3; }
CLOSURE_DIST_IN=release_r8_table/accuracy_law_points.csv python3 -u fit_accuracy_law.py || { echo "ACC FAILED"; exit 3; }
echo "ACC DONE $(date)"
