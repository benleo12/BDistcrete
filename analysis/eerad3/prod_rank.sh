#!/bin/bash
# one production rank: distinct seed index, shared read-only grid
exec $SCRATCH/eerad3_thrust/eerad3-1.0/eerad3 -i eerad3.input -n ${SLURM_PROCID:-0} \
     > prod_${SLURM_PROCID:-0}.log 2>&1
