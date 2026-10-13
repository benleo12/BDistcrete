#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -c 64
#SBATCH -t 03:00:00
#SBATCH -J trunc_geo
#SBATCH -o logs/trunc_geo_%j.out
#SBATCH -e logs/trunc_geo_%j.err
# The truncation test of App. C on the concatenation network with the multiplicative head
# (dctr_truncation.py), with the data prepared exactly as in its training (concat_geo.sbatch).
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export DM_DATA=data_stagePURE17 CONCAT_HEAD=geometric LADDER_MIX_SPLIT=8,8
export CONCAT_NREF_PER=5528 CONCAT_EVENTS_PER_RUN=25000
python3 -u dctr_truncation.py MIX output/models/concat_MIX_geohead.pt
echo "TRUNC GEO DONE rc=$? $(date)"
