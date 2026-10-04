#!/bin/bash
# The 48 off-locus Herwig points from design_stageF_augment.py.
#
# This follows stageF.sbatch exactly rather than using a job array. The original campaign ran 77
# runs of 80000 events in 23 minutes of wall time on one node with xargs, so 48 runs fit in the
# same shape with room to spare, and the array version I wrote first would have been both slower
# (a quarter node per task in the shared queue) and a fresh path to get wrong.
#SBATCH -A <your-account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J stageFaug
#SBATCH -o $SCRATCH/stageF/logs/stageFaug_%j.out
#SBATCH -e $SCRATCH/stageF/logs/stageFaug_%j.err
BASE=$SCRATCH/stageF
CSV=$BASE/code/stageF_augment_design.csv
EVENTS=${EVENTS:-80000}
mkdir -p $BASE/logs $BASE/runs $BASE/data $BASE/shim
ln -sf /lib64/libreadline.so.7 $BASE/shim/libreadline.so.8 2>/dev/null
cd $BASE

# Pre-flight. Every check here is one that has already cost a run somewhere in this project:
# a truncated parameter list, a row with the wrong field count, and a run id that silently
# overwrote existing data.
[ -f "$CSV" ] || { echo "no design at $CSV"; exit 2; }
n=$(tail -n +2 "$CSV" | wc -l)
bad=$(awk -F, 'NR>1 && NF!=10 {print NR": "NF" fields"}' "$CSV")
[ -z "$bad" ] || { echo "rows with the wrong field count:"; echo "$bad"; exit 2; }
clash=$(awk -F, 'NR>1{print $1}' "$CSV" | while read -r r; do
          [ -f "$BASE/data/particles_full_${r}.npz" ] && echo "$r"; done)
[ -z "$clash" ] || { echo "these run ids already have data, refusing to overwrite:"; echo "$clash"; exit 2; }
echo "stageFaug: $n runs, $EVENTS events each, Herwig 7.3.0p1, started $(date)"

tail -n +2 "$CSV" | tr ',' ' ' | \
  xargs -P 48 -L 1 bash -c '
    rid=$0; role=$1; shift 1   # shift never touches $0, so ONE shift drops only role
    '"$BASE"'/code/gen_stageF.sh "$rid" "$@" '"$EVENTS"' >> '"$BASE"'/logs/run_$rid.log 2>&1 \
      || echo "FAILED $rid" >> '"$BASE"'/logs/failures_stageFaug.txt'

echo "stageFaug DONE $(date)"
got=$(awk -F, 'NR>1{print $1}' "$CSV" | while read -r r; do
        [ -f "$BASE/data/particles_full_${r}.npz" ] && echo x; done | wc -l)
echo "produced $got of $n particle files, $(awk -F, 'NR>1{print $1}' "$CSV" | while read -r r; do [ -f "$BASE/data/shapes_run_${r}.csv" ] && echo x; done | wc -l) shape files"
# Report failures if there were any, then exit 0 EXPLICITLY. Without the last line the script's
# status is the status of the test above it, so a clean run with no failures file exits 1 and
# Slurm marks the job FAILED. That is what happened to job 58720756: 48 of 48 runs produced,
# no failures, exit code 1, and the dependent training job went DependencyNeverSatisfied.
if [ -f $BASE/logs/failures_stageFaug.txt ]; then
  echo "FAILURES:"; cat $BASE/logs/failures_stageFaug.txt
  exit 1
fi
echo "no failures"
exit 0
