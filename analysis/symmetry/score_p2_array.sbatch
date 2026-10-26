#!/bin/bash
#SBATCH -p h200q
#SBATCH --gres=gpu:h200:1
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J p2score
#SBATCH -o logs/p2score_%A_%a.out
# The strict test (400k events per held-out point) of every model of a train_p2.sbatch array, one
# element per "STAGE:NET:SEED" in SPECS, the same SPECS and DESIGN_TAG as the training array.
# Submit with --dependency=aftercorr:<training array> so element i scores model i once it exists.
# Strict data: data_abc_h400 for A-C, data_stagePURE17_v3_h400 for E and F (it holds their training
# runs and the 16 single-generator held-out points at 400k), data_stagePURE17_orig_h400 for MIX
# (data_stagePURE17_v3_h400 for the mixture of the 720 runs, DESIGN_TAG v3full: its 48 points).
set -uo pipefail
MAXENT_DIR=${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}; R=$(dirname "$MAXENT_DIR")   # analysis/ in the repository, R its parent
cd "$MAXENT_DIR" || exit 1
read -ra ALL <<< "${SPECS:?}"; IFS=: read -r STAGE NET SEED <<< "${ALL[$SLURM_ARRAY_TASK_ID]}"
DT=${DESIGN_TAG:-}; TAG=P2${DT}${NET}${SEED}
if [ "$STAGE" = MIX ]; then
  MODEL=output/models/MIX${TAG}_$([ $NET = cat ] && echo concat.pt || echo trunk.pt)
  case $DT in v3*) DATA=data_stagePURE17_v3_h400 ;; *) DATA=data_stagePURE17_orig_h400 ;; esac
else
  MODEL=output/models/${STAGE}_${TAG}_$([ $NET = cat ] && echo concat.pt || echo trunk.pt)
  case $STAGE in A|B|C) DATA=data_abc_h400 ;; *) DATA=data_stagePURE17_v3_h400 ;; esac
fi
[ -f "$MODEL" ] || { echo "NO MODEL $MODEL (training failed or not finished)"; exit 4; }
export MAXENT_DIR STAGE NET SEED MODEL DATA OUT=output/p2${DT}strict_${STAGE}_${NET}_s${SEED}.json
export MMAP_BASE=${MMAP_BASE:-$R/mmap} E_CSV=${E_CSV:-stageE_design.csv} F_CSV=${F_CSV:-stageF_design_aug.csv}
case $DT in aug*) export ABC_DESIGN=$R/gen/designs ;; esac
# score_p2.sbatch is in the analysis folder on Symmetry and in analysis/symmetry/ in the repository
S=score_p2.sbatch; [ -f $S ] || S=symmetry/score_p2.sbatch
bash $S
