#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 04:00:00
#SBATCH -J seedstudy
#SBATCH -o logs/seedstudy_%j.out
# Training-seed spread of Table 4, and the mixture-head control. Up to four trainings run side by
# side, one per GPU, each with the published recipe of its column and followed by the closure
# evaluation the paper used for that column:
#   fact   factorized network, r2_ladder.py (A, B, C, E) or stage_mixture.py (MIX), then
#          ladder_widths_dedup.py at T = 1 (A, B, C) or ladder_widths_big.py (E, MIX)
#   concat concatenation network (theta on every particle), concat_baseline.py
#   late   late fusion (theta after the sum), concat_baseline.py with CONCAT_MODE=late
#   mixe   concatenation network with the exact mixture head (CONCAT_HEAD=mixture), MIX only
#   mixl   late fusion with the exact mixture head, MIX only
# TASKS="design:stage:seed ..." with at most four entries. Every output carries the seed in its
# name, so no published export or result file is touched. The published Stage B factorized seeds
# are 100-103, so seed 0 of fact:B trains with offset 100. SMOKE=1 runs tiny budgets for timing.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
mkdir -p logs output/seedstudy
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
SMOKE=${SMOKE:-0}
TASKS=${TASKS:?set TASKS="design:stage:seed ..."}

run_task() {
  local design stage seed gpu=$2
  IFS=: read design stage seed <<< "$1"
  export CUDA_VISIBLE_DEVICES=$gpu
  local tagsfx=""; [ "$SMOKE" = 1 ] && tagsfx=smoke
  case $design in
    concat|late|mixe|mixl)
      { [ $design = late ] || [ $design = mixl ]; } && export CONCAT_MODE=late
      { [ $design = mixe ] || [ $design = mixl ]; } && export CONCAT_HEAD=mixture
      export CONCAT_SEED_OFFSET=$seed CONCAT_OUT=output/seedstudy/${design}_${stage}_s${seed}${tagsfx}.json
      case $stage in
        A) export CONCAT_NREF_PER=12000 CONCAT_DROP= ;;
        B) export CONCAT_NREF_PER=12000 CONCAT_DROP=6704 ;;
        C) export CONCAT_NREF_PER=12000 CONCAT_DROP=6902 ;;
        E) export CONCAT_NREF_PER=12000 ;;
        MIX) export CONCAT_NREF_PER=5989 DM_DATA=data_stageDM17aug_v2 ;;
      esac
      [ "$SMOKE" = 1 ] && export CONCAT_STEPS=120 CONCAT_NREF_PER=300
      python3 -u concat_baseline.py $stage ;;
    fact)
      local off=$seed; [ $stage = B ] && [ $seed = 0 ] && off=100
      export LADDER_SEED_OFFSET=$off
      [ "$SMOKE" = 1 ] && export LADDER_STEPS=200
      case $stage in
        A|B|C)
          local suf=_seed${seed}${tagsfx} drop=
          [ $stage = B ] && drop=6704; [ $stage = C ] && drop=6902
          export LADDER_SUFFIX=$suf LADDER_OUT=output/seedstudy/fact_${stage}_s${seed}${tagsfx}.json
          LADDER_STAGES=$stage python3 -u r2_ladder.py \
            && python3 apply_T.py ${stage}${suf} 1.0 \
            && DROP=$drop EXPORT=output/models/${stage}${suf}_cond.npz \
               python3 -u ladder_widths_dedup.py $stage output/models/${stage}${suf}_ref.npz ;;
        E)
          local suf=_seed${seed}${tagsfx}
          export LADDER_SUFFIX=$suf LADDER_OUT=output/seedstudy/fact_E_s${seed}${tagsfx}.json \
                 LADDER_K=48 LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_MMAP_DIR=$SCRATCH/mmap_${SLURM_JOB_ID}_$gpu \
                 LADDER_STAGES=E STAGEE_CSV=stageE_design.csv
          python3 check_stage_data.py E && python3 -u r2_ladder.py E \
            && python3 -u ladder_widths_big.py E output/models/E${suf}_cond.npz output/models/E${suf}_ref.npz
          local st=$?; rm -rf $LADDER_MMAP_DIR; return $st ;;
        MIX)
          local suf=17aug_seed${seed}${tagsfx}
          export LADDER_SUFFIX=$suf DM_TAG=MIX$suf DM_DATA=data_stageDM17aug_v2 \
                 STAGED_OUT=output/seedstudy/fact_MIX_s${seed}${tagsfx}.json \
                 LADDER_K=48 LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_MMAP_DIR=$SCRATCH/mmap_${SLURM_JOB_ID}_$gpu \
                 LADDER_STAGES=MIX
          python3 check_stage_data.py MIX && python3 -u stage_mixture.py \
            && python3 -u ladder_widths_big.py MIX output/models/MIX${suf}_cond.npz output/models/MIX${suf}_ref.npz
          local st=$?; rm -rf $LADDER_MMAP_DIR; return $st ;;
      esac ;;
    *) echo "unknown design $design"; return 2 ;;
  esac
}

echo "seedstudy job $SLURM_JOB_ID smoke=$SMOKE tasks: $TASKS $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
i=0; pids=()
for t in $TASKS; do
  [ $i -ge 4 ] && { echo "more than four tasks, ignoring $t"; continue; }
  ( run_task $t $i ) > logs/seedstudy_${t//:/_}_${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!); echo "started $t on GPU $i"; i=$((i+1))
done
rc=0; j=0
for t in $TASKS; do
  [ $j -ge ${#pids[@]} ] && break
  wait ${pids[$j]}; r=$?; echo "TASK $t rc=$r $(date)"; [ $r -ne 0 ] && rc=$r; j=$((j+1))
done
echo "SEEDSTUDY DONE rc=$rc $(date)"
exit $rc
