#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J mix17v2b
#SBATCH -o $SCRATCH/BDistcrete/analysis/logs/mix17v2b_%j.out
#SBATCH -e $SCRATCH/BDistcrete/analysis/logs/mix17v2b_%j.err
# Part B: the MIX17aug reference bundle replayed on the verified v2 mixture data.
set -uo pipefail
cd $SCRATCH/maxent_code
module load pytorch/2.6.0
DM_DATA=data_stageDM17aug_v2 python3 -u rebuild_ref_lowmem.py MIX17aug > logs/refv2_MIX17aug.log 2>&1
rc=$?
tail -9 logs/refv2_MIX17aug.log
echo "PART B DONE rc=$rc $(date)"
exit $rc
