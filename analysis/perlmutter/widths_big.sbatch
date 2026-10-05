#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J widthsv2
#SBATCH -o logs/widthsv2_%j.out
#SBATCH -e logs/widthsv2_%j.err
# Binned closure widths of the E, F (augmented) and MIX17 stages on the corrected shapes.
set -u
cd $SCRATCH/BDistcrete/analysis
module load pytorch/2.6.0
export OMP_NUM_THREADS=16
python3 -u ladder_widths_big.py E output/models/E_cond.npz output/models/E_ref.npz > logs/widthsv2_E.log 2>&1 &
STAGEF_CSV=stageF_design_aug.csv python3 -u ladder_widths_big.py F output/models/Fauglong_cond.npz output/models/Fauglong_ref.npz > logs/widthsv2_F.log 2>&1 &
DM_DATA=data_stageDM17aug_v2 python3 -u ladder_widths_big.py MIX output/models/MIX17aug_cond.npz output/models/MIX17aug_ref.npz > logs/widthsv2_MIX.log 2>&1 &
wait
for t in E F MIX; do echo "--- $t"; cat logs/widthsv2_$t.log | grep -v "^\[stage"; done
echo "WIDTHS DONE $(date)"
