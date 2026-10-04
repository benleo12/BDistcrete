#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/concat_%x_%j.log
# The unfactorized (concatenation) baseline at Stages A, B and C under the production recipe of
# the factorized ladder (SiLU, automatic embedding scale, lr 1e-3), without the held-out runs that
# duplicate training runs (Stage B 6704, Stage C 6902).
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 CONCAT_NREF_PER=12000
STAGE=${STAGE:?set STAGE=A, B or C}
case $STAGE in
  A) export CONCAT_DROP= ;;
  B) export CONCAT_DROP=6704 ;;
  C) export CONCAT_DROP=6902 ;;
esac
export CONCAT_OUT=output/concat_baseline_${STAGE}_prod.json
echo "stage $STAGE act=$LADDER_ACT emb=$LADDER_EMB_SCALE lr=$LADDER_LR drop=$CONCAT_DROP $(date)"
python3 -u concat_baseline.py $STAGE
echo "CONCAT $STAGE DONE rc=$? $(date)"
