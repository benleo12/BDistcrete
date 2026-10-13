#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 05:00:00
#SBATCH -J concat_geo
#SBATCH -o logs/concat_geo_%j.out
#SBATCH -e logs/concat_geo_%j.err
# The concatenation (DCTR) network with the MULTIPLICATIVE mixture head on the pure-run data set,
# the control for the seventeen-parameter row of the network comparison, trained with the
# production recipe and tested at the same sixteen held-out pure points as the factorized model.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
export DM_DATA=data_stagePURE17 CONCAT_HEAD=geometric LADDER_MIX_SPLIT=8,8
export CONCAT_NREF_PER=5528 CONCAT_EVENTS_PER_RUN=25000
export CONCAT_OUT=output/concat_baseline_MIX_geohead.json
echo "concat geometric on $DM_DATA, nref_per=$CONCAT_NREF_PER, events per run $CONCAT_EVENTS_PER_RUN $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u concat_baseline.py MIX
echo "CONCAT GEO DONE rc=$? $(date)"
