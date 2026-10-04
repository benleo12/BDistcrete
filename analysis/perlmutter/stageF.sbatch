#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -J stageF
#SBATCH -o $SCRATCH/stageF/logs/stageF_%j.out
#SBATCH -e $SCRATCH/stageF/logs/stageF_%j.err
BASE=$SCRATCH/stageF
mkdir -p $BASE/logs $BASE/runs $BASE/data $BASE/shim
ln -sf /lib64/libreadline.so.7 $BASE/shim/libreadline.so.8 2>/dev/null
cd $BASE
echo "stageF: Herwig 7.3.0p1, 80000 events per run, started $(date)"
tail -n +2 $BASE/code/stageF_design.csv | tr ',' ' ' | \
  xargs -P 77 -L 1 bash -c '
    rid=$0; role=$1; shift 1   # shift never touches $0, so ONE shift drops only role
    '"$BASE"'/code/gen_stageF.sh "$rid" "$@" 80000 >> '"$BASE"'/logs/run_$rid.log 2>&1 \
      || echo "FAILED $rid" >> '"$BASE"'/logs/failures_stageF.txt'
echo "stageF DONE $(date)"
echo "npz: $(ls $BASE/data/particles_full_8*.npz 2>/dev/null | wc -l)  shapes: $(ls $BASE/data/shapes_run_8*.csv 2>/dev/null | wc -l)"
[ -f $BASE/logs/failures_stageF.txt ] && cat $BASE/logs/failures_stageF.txt
