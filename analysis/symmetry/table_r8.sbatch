#!/bin/bash
#SBATCH -p preq
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 2:00:00
#SBATCH -J table_r8
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/table_r8_%j.out
# Closure of the r8 release networks on the held-out runs of their designs, for release/STAGES.md,
# closure_<stage>.csv and closure_summary.json. C_1M has no bundle here and is read back from its CSV.
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 STAGEE_CSV=stageE_design_v3all.csv STAGEF_CSV=stageF_design_v3all.csv DM_DATA=data_stagePURE17_v3_h400 RELEASE_DIR=release_r8_table STAGEE_DATA=data_stagePURE17_v3_h400 STAGEF_DATA=data_stagePURE17_v3_h400
python3 -u make_stage_table.py C_1M E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || { echo "TABLE FAILED"; exit 3; }
RELEASE_DIR=release_r8 python3 -u precision_gap.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || echo "GAP FAILED"
echo "TABLE DONE $(date)"
