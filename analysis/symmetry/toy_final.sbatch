#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -J toyfinal
#SBATCH --array=0-8
#SBATCH -o /mnt/beegfs/bassi/reweighting/maxent_code/logs/toy_final_%A_%a.out
# The Gaussian toy with the paper's network and training (toy_final.py): three trainings at K=48 and
# one at each K of the scan.
set -uo pipefail
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code; source $R/env/torch/bin/activate
SPECS=(48:0 48:1 48:2 1:0 2:0 3:0 4:0 8:0 16:0)
IFS=: read -r TOY_K TOY_SEED <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
export TOY_K TOY_SEED LADDER_ACT=silu LADDER_EMB_SCALE=auto OMP_NUM_THREADS=4
echo "=== toy K=$TOY_K seed=$TOY_SEED on $(hostname) $(date)"
python3 -u toy_final.py || { echo "TOY FAILED"; exit 3; }
echo "TOY DONE $(date)"
