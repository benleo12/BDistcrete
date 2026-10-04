#!/bin/bash
sbatch $SCRATCH/anchor1M/ref/ref_array.sbatch
sbatch $SCRATCH/anchor1M/meps/meps_array.sbatch
squeue -u $USER
