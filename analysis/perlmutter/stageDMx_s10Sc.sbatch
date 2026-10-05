#!/bin/bash
#SBATCH -A <your-gpu-account> -C gpu -q regular -N 1 -G 1 -c 32 -t 03:00:00 -J sDMxSc_s10
#SBATCH -o logs/stageDMx_s10_%j.log
module load pytorch
cd $SCRATCH/BDistcrete/analysis
# A FRESH mmap dir per submission: reusing one has hung Lustre.
export DM_DATA=data_stageDM2 LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3 LADDER_LOWMEM=1 LADDER_MMAP=1 LADDER_DEAD=report LADDER_REVIVE=1 LADDER_STEPS=72000 LADDER_K=24 LADDER_HEAD_KIND=mixture LADDER_SEED_OFFSET=10 LADDER_SUFFIX=DMXSc10 LADDER_MMAP_DIR=$SCRATCH/mmap_${SLURM_JOB_ID} STAGED_OUT=output/stageDMxSc_s10.json
python -u stageDM.py
rc=$?
rm -rf $LADDER_MMAP_DIR
exit $rc
