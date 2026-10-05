#!/bin/bash
#SBATCH -A <your-gpu-account> -C gpu -q regular -N 1 -G 1 -c 32 -t 02:30:00 -J concat_Csvd
#SBATCH -o logs/concat_Csvd_%j.log
module load pytorch
cd $SCRATCH/BDistcrete/analysis
export CONCAT_OUT=output/concat_baseline_C.json
python -u concat_baseline.py C
