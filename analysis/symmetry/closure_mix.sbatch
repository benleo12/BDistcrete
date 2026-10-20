#!/bin/bash
#SBATCH -p sharedq
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J closmix
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/closure_mix_%j.out
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
export DM_DATA=data_stagePURE17_v3 LADDER_HEAD_KIND=geometric LADDER_MIX_SPLIT=8,8 OMP_NUM_THREADS=4
python3 -u fig_closure_mix.py 7214 8211 || echo "CLOSURE MIX FAILED"
