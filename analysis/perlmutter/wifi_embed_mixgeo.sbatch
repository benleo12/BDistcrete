#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -J embed_geo
#SBATCH -o logs/embed_mixgeo_%j.out
#SBATCH -e logs/embed_mixgeo_%j.err
# The wifi basis of the trained multiplicative mixture (the step that failed in job 59619982 on a
# torch.load default, fixed in wifi_embed.py).
set -uo pipefail
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export OMP_NUM_THREADS=32 DM_DATA=data_stagePURE17
python3 -u wifi_embed.py MIX output/models/MIXGEO_cond.npz output/models/MIXGEO_trunk.pt
echo "EMBED MIXGEO DONE $(date)"
