#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -J stageE_gen
#SBATCH -o logs/stageE_%j.out
#SBATCH -e logs/stageE_%j.err
BASE=$SCRATCH/stageE
mkdir -p $BASE/logs $BASE/runs $BASE/data
cd $BASE
EV=${EV:-80000}
echo "Stage E generation, $EV events per run, started $(date)"
# whole node is ours: run the 82 cards concurrently with xargs, one core each.
# (a while-read pipeline would background into a subshell and `wait` would not see them)
tail -n +2 $BASE/code/stageE_design.csv | tr ',' ' ' | \
  xargs -P 82 -L 1 bash -c '
    rid=$0; role=$1; shift 1   # shift never touches $0, so ONE shift drops only role
    '"$BASE"'/code/gen_stageE.sh "$rid" "$@" '"$EV"' >> '"$BASE"'/logs/run_$rid.log 2>&1 \
      || echo "FAILED $rid" >> '"$BASE"'/logs/failures.txt'
echo "STAGE E GENERATION DONE $(date)"
n=$(ls $BASE/data/particles_full_7*.npz 2>/dev/null | wc -l)
s=$(ls $BASE/data/shapes_run_7*.csv 2>/dev/null | wc -l)
echo "npz produced: $n / 82    shapes produced: $s / 82"
[ -f $BASE/logs/failures.txt ] && echo "FAILURES:" && cat $BASE/logs/failures.txt
