#!/bin/bash
# launch_farm.sh -- runs on a login node under nohup. Chains successive ONE-node
# interactive-QOS sallocs until every job in all_jobs.txt has a summary line.
# Each salloc runs the farm driver; the driver requeues interrupted jobs itself.
# Allocation-not-granted -> wait 180 s and retry. Farm complete -> exit.
BASE=$SCRATCH/energyflower
LOG=$BASE/logs/salloc.log
LOGS=$BASE/logs
mkdir -p "$LOGS"

all_done() {
    local n_all n_done
    n_all=$(grep -cv '^$' $BASE/jobs/all_jobs.txt)
    n_done=$(wc -l < "$LOGS/job_summary.txt" 2>/dev/null || echo 0)
    [ "$n_done" -ge "$n_all" ]
}

for attempt in $(seq 1 40); do
    if all_done; then
        echo "=== ALL JOBS ACCOUNTED FOR $(date)" >> "$LOG"
        echo "FARM CHAIN COMPLETE" >> "$LOG"
        exit 0
    fi
    s0=$(cat $BASE/logs/farm_start_epoch.txt 2>/dev/null || echo 0)
    echo "=== salloc attempt $attempt $(date)" >> "$LOG"
    salloc -N 1 -C gpu -q interactive -t 04:00:00 -A ${NERSC_ACCOUNT:?set to your NERSC project}_g --gpus 4 \
        srun -N 1 -n 1 --gpus 4 --cpu-bind=none bash $BASE/code/farm_driver.sh >> "$LOG" 2>&1
    rc=$?
    s1=$(cat $BASE/logs/farm_start_epoch.txt 2>/dev/null || echo 0)
    if [ "$s1" == "$s0" ]; then
        echo "=== allocation not granted (rc=$rc), retry in 180s" >> "$LOG"
        sleep 180
        continue
    fi
    echo "=== salloc round ended rc=$rc $(date)" >> "$LOG"
    sleep 10
done
if all_done; then
    echo "FARM CHAIN COMPLETE" >> "$LOG"
    exit 0
fi
echo "=== gave up after 40 salloc attempts $(date)" >> "$LOG"
exit 3
