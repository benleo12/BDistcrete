#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
# Measured, not guessed: stage F at 36000 steps takes 23:55, stage E 25:54, the 17-parameter
# mixture 28:29. The 12:00:00 this used to request cost 12 node hours against a balance of
# 11 and simply bounced. Override with `sbatch -t` for a longer-budget run.
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J train_stage
#SBATCH -o logs/train_%x_%j.out
#SBATCH -e logs/train_%x_%j.err
set -e
STAGE=${STAGE:?set STAGE=E or F}
cd $SCRATCH/BDistcrete/analysis; mkdir -p logs output
module load pytorch/2.6.0
# MANDATORY on Perlmutter: every earlier training here silently ran with the r2_ladder defaults,
# i.e. ReLU and no embedding autoscale, which is not the published configuration.
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
# d = 8, so the second-order rank count is 1 + 8 + 36 = 45; train above it.
export LADDER_K=${LADDER_K:-48}
export LADDER_LOWMEM=1 LADDER_MMAP=1
# A FRESH mmap dir per submission: reusing one has hung Lustre.
export LADDER_MMAP_DIR=$SCRATCH/mmap_${SLURM_JOB_ID}
export LADDER_STAGES=$STAGE
# Which design CSV the stage is built from. The Herwig augmentation trains on the 112-point
# stageF_design_aug.csv while the original 64-point file stays in place, so the two heads
# remain separately reproducible. Echoed below because a silently defaulted design is the
# one mistake here that still produces a plausible model.
export STAGEE_CSV=${STAGEE_CSV:-stageE_design.csv}
export STAGEF_CSV=${STAGEF_CSV:-stageF_design.csv}
export LADDER_STEPS=${LADDER_STEPS:-36000}
# Overridable, so a longer-budget rerun does not overwrite the run it is being compared with.
export LADDER_OUT=${LADDER_OUT:-output/ladder_stage${STAGE}.json}
# Export the suffix ONLY if it has a value. Exporting it empty leaves the name present and
# empty, and the 19 stage drivers that set their own default do so with os.environ.setdefault,
# which does not replace a key that exists. The driver's suffix would stay empty, its export
# would be written under the bare stage key, and the check at the end of the driver would look
# for exactly that and find it, so a run under the wrong name would look successful.
if [ -n "${LADDER_SUFFIX:-}" ]; then export LADDER_SUFFIX; else unset LADDER_SUFFIX; fi
echo "stage=$STAGE K=$LADDER_K steps=$LADDER_STEPS act=$LADDER_ACT emb=$LADDER_EMB_SCALE design=$([ "$STAGE" = F ] && echo $STAGEF_CSV || ([ "$STAGE" = E ] && echo $STAGEE_CSV || echo n/a))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Refuse to train on an incomplete design. A design listing more runs than exist on disk does
# not crash the trainer, it just trains on what loaded and produces a head that looks normal.
python3 check_stage_data.py $STAGE || { echo "STAGE $STAGE DATA INCOMPLETE, refusing to train"; exit 3; }
# LADDER_DRIVER lets a variant run (a training-run subset, say) reuse this whole recipe,
# since the mandatory settings above are what make a run comparable with the published ones.
if [ -n "${LADDER_DRIVER:-}" ]; then
  python3 -u "$LADDER_DRIVER"
else
  python3 -u r2_ladder.py $STAGE
fi
rc=$?
rm -rf $LADDER_MMAP_DIR
echo "TRAIN STAGE $STAGE DONE rc=$rc $(date)"
exit $rc
