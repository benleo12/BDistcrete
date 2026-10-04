#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=60G
#SBATCH -t 02:00:00
#SBATCH -J recalT
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/recalT_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/recalT_%j.err
set -u
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export STAGEF_CSV=stageF_design_aug.csv
for t in E Fauglong; do
  echo "=== $t $(date)"; python recalibrate_T.py $t > logs/recal_T_$t.log 2>&1; echo "exit $?"
done
echo "=== MIX17aug $(date)"; DM_DATA=data_stageDM17aug python recalibrate_T.py MIX17aug > logs/recal_T_MIX17aug.log 2>&1; echo "exit $?"
echo "ALL DONE $(date)"
