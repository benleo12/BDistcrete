#!/bin/bash
#SBATCH -p preq
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH -t 2:00:00
#SBATCH -J pkg_r8
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/package_r8_%j.out
# Package the networks of paper r8 for the public release: Stage D (Sherpa, export E_P2v3fullfac0,
# public name E), Stage E (Herwig, F_P2v3fullfac0, public F) and the seventeen-parameter mixture
# (MIXGEOV3S9, public MIX17), on their v3 designs.
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
export OMP_NUM_THREADS=8 STAGEE_CSV=stageE_design_v3all.csv STAGEF_CSV=stageF_design_v3all.csv DM_DATA=data_stagePURE17_v3 RELEASE_DIR=release_r8
python3 -u package_release.py E_P2v3fullfac0=E F_P2v3fullfac0=F MIXGEOV3S9=MIX17 || { echo "PACKAGING FAILED"; exit 3; }
echo "PACKAGING DONE $(date)"
