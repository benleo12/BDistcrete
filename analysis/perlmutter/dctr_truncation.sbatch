#!/bin/bash
#SBATCH -A <your-gpu-account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 01:30:00
#SBATCH -J dctrtrunc
#SBATCH -o logs/dctrtrunc_%j.out
# The truncation test of the concatenation networks' learned log ratio (dctr_truncation.py) on the
# published weights of Table 4: Stages A, B, C and the eight-parameter model side by side, then the
# seventeen-parameter mixture. Each first reproduces its published closure width.
set -u
module load pytorch/2.6.0
cd $SCRATCH/BDistcrete/analysis
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
run() {  # stage gpu nref drop published [dm_data]
  CUDA_VISIBLE_DEVICES=$2 CONCAT_NREF_PER=$3 CONCAT_DROP=$4 TRUNC_PUBLISHED=$5 DM_DATA=${6:-} \
    python3 -u dctr_truncation.py $1 output/models/concat_$1.pt > logs/dctrtrunc_$1_${SLURM_JOB_ID}.log 2>&1
  echo "TRUNC $1 rc=$? $(date)"
}
run A 0 12000 "" output/concat_baseline_A_prod.json &
run B 1 12000 6704 output/concat_baseline_B_prod.json &
run C 2 12000 6902 output/concat_baseline_C_prod.json &
run E 3 12000 "" output/concat_baseline_E_silu.json &
wait
run MIX 0 5989 "" output/concat_baseline_MIX_silu.json data_stageDM17aug_v2
for t in A B C E MIX; do tail -4 logs/dctrtrunc_${t}_${SLURM_JOB_ID}.log; done
echo "DCTRTRUNC DONE $(date)"
