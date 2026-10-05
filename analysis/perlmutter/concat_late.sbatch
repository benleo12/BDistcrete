#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -o logs/concat_late_%x_%j.log
# The late-fusion baseline (theta joined after the sum over particles), under the production recipe
# and with the same data and reference sizes as the early-fusion runs of concat_small (Stage C) and
# concat_big (MIX), so the three networks of Sec. 5.8 are compared on identical footing.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 CONCAT_MODE=late
STAGE=${STAGE:?set STAGE=C or MIX}
case $STAGE in
  C) export CONCAT_NREF_PER=12000 CONCAT_DROP=6902 ;;
  MIX) export CONCAT_NREF_PER=5989 DM_DATA=data_stageDM17aug_v2 ;;
esac
export CONCAT_OUT=output/concat_baseline_${STAGE}_late.json
echo "stage $STAGE mode=$CONCAT_MODE act=$LADDER_ACT emb=$LADDER_EMB_SCALE lr=$LADDER_LR $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u concat_baseline.py $STAGE
echo "LATE $STAGE DONE rc=$? $(date)"
