#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 02:30:00
#SBATCH -o logs/concat_%x_%j.log
# The unfactorized (concatenation) baseline under the production recipe, for the comparison with
# the factorized network at three, eight and seventeen parameters. The recipe settings are exported
# explicitly: without them r2_ladder falls back to ReLU and no embedding scale.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
STAGE=${STAGE:?set STAGE=C, E or MIX}
case $STAGE in
  C|E) export CONCAT_NREF_PER=12000 ;;
  MIX) export CONCAT_NREF_PER=5989 DM_DATA=data_stageDM17aug_v2 ;;
esac
export CONCAT_OUT=output/concat_baseline_${STAGE}_silu.json
echo "stage $STAGE act=$LADDER_ACT emb=$LADDER_EMB_SCALE lr=$LADDER_LR nref_per=$CONCAT_NREF_PER $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u concat_baseline.py $STAGE
echo "CONCAT $STAGE DONE rc=$? $(date)"
