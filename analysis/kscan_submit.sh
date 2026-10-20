#!/bin/bash
# The K scan of App. C at the final standard (Benoit, Oct 18: redo overnight). One factorized
# training per K, seed 0, through train_p2.sbatch exactly as the paper's cells (324k steps, no
# patience, four bootstrap members, fit fraction 0.2, best stop-split checkpoint, temperature on the
# stop split), each followed element by element by its strict test (score_p2_array.sbatch).
#   Stages A, B, C on the densified designs (gen/designs, gen/data_abc), K = 1 2 3 4 6 8 16
#   the Sherpa stage on the densified v3 design (stageE_design_v3all.csv), K = 1 2 4 6 8 9 10 12 16 24 32 45
# K = 48 is the standard and is taken from the Phase 2 cells (DESIGN_TAG augfull/aug and v3full,
# seed 0). Outputs output/p2augK<K>[strict]_<S>_fac_s0.json and output/p2v3K<K>[strict]_E_fac_s0.json.
# Memory maps in /dev/shm (RAM, charged to the job), so the scan writes nothing large to beegfs.
#   bash kscan_submit.sh
set -u; export PATH=/cm/shared/apps/slurm/current/bin:$PATH
R=/mnt/beegfs/bassi/reweighting; cd $R/maxent_code
export STEPS=324000 PATIENCE=0 GPU_FETCH=1 MMAP_BASE=/dev/shm
# the Sherpa stage first: its trainings are the longest
for K in 1 2 4 6 8 9 10 12 16 24 32 45; do
  T=$(K=$K DESIGN_TAG=v3K$K E_DATA=data_stagePURE17_v3 E_CSV=stageE_design_v3all.csv SPECS="E:fac:0" \
      sbatch --parsable --array=0 -c 6 --mem=200G -J kscanE --export=ALL train_p2.sbatch)
  S=$(K=$K DESIGN_TAG=v3K$K E_CSV=stageE_design_v3all.csv SPECS="E:fac:0" \
      sbatch --parsable --array=0 -c 6 --mem=200G --dependency=aftercorr:$T --export=ALL score_p2_array.sbatch)
  echo "E K=$K train $T strict $S"
done
for K in 1 2 3 4 6 8 16; do
  T=$(K=$K DESIGN_TAG=augK$K ABC_DESIGN=$R/gen/designs SPECS="A:fac:0 B:fac:0 C:fac:0" \
      sbatch --parsable --array=0-2 -c 6 --mem=160G -J kscanABC --export=ALL train_p2.sbatch)
  S=$(K=$K DESIGN_TAG=augK$K SPECS="A:fac:0 B:fac:0 C:fac:0" \
      sbatch --parsable --array=0-2 -c 6 --mem=160G --dependency=aftercorr:$T --export=ALL score_p2_array.sbatch)
  echo "ABC K=$K train $T strict $S"
done
