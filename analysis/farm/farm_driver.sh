#!/bin/bash
# farm_driver.sh -- self-scheduling 4-GPU job farm, runs on the allocated compute node.
# v3: NUMA-pins each job to its GPU-affine CPU quadrant (4-way host contention was
# costing 3x per-step), queue lines carry a walltime estimate (name|est_s|cmd) and a
# worker only pops a job that fits in the remaining allocation, so nothing gets
# killed at the walltime wall and requeued. Queue is rebuilt each round from
# all_jobs.txt minus jobs already recorded in job_summary.txt.
BASE=$SCRATCH/energyflower
ALLJOBS=$BASE/jobs/all_jobs.txt
QUEUE=$BASE/jobs/queue.txt
LOGS=$BASE/logs
mkdir -p "$LOGS"
FARMLOG=$LOGS/farm.log

echo "=== FARM DRIVER v3 start $(date) on $(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none}" >> "$FARMLOG"
echo "${SLURM_JOB_ID:-none} $(date +%s)" >> $LOGS/round_starts.txt
date +%s > $BASE/logs/farm_start_epoch.txt

# rebuild queue: every job in all_jobs.txt with no summary line yet
: > "$QUEUE"
while IFS= read -r line; do
    [ -z "$line" ] && continue
    name=${line%%|*}
    if ! grep -q "^$name rc=" "$LOGS/job_summary.txt" 2>/dev/null; then
        echo "$line" >> "$QUEUE"
    fi
done < "$ALLJOBS"
rm -f "$QUEUE.lock"
echo "queue rebuilt: $(wc -l < $QUEUE) jobs pending" >> "$FARMLOG"
if [ ! -s "$QUEUE" ]; then
    echo "FARM ALL DONE (queue already empty) $(date)" >> "$FARMLOG"
    exit 0
fi

# module environment
if ! type module >/dev/null 2>&1; then
    for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/zz-lmod.sh /etc/profile.d/lmod.sh; do
        [ -r "$f" ] && source "$f" && break
    done
fi
module load pytorch/2.6.0 >> "$FARMLOG" 2>&1

export LADDER_ACT=silu LADDER_EMB_SCALE=auto PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14

nvidia-smi -L >> "$FARMLOG" 2>&1
nvidia-smi topo -m 2>/dev/null | head -8 >> "$FARMLOG"
python -c "import torch, numpy, pandas; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" >> "$FARMLOG" 2>&1
if ! python -c "import torch; assert torch.cuda.is_available()" 2>>"$FARMLOG"; then
    echo "FATAL: cuda not available, aborting farm" >> "$FARMLOG"
    exit 1
fi

# remaining walltime of this allocation, in seconds
tl=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -1)
rem_s=0
if [ -n "$tl" ]; then
    IFS=- read -r d hms <<< "$tl"
    if [ -z "$hms" ]; then hms=$d; d=0; fi
    IFS=: read -r a b c <<< "$hms"
    if [ -z "$c" ]; then c=$b; b=$a; a=0; fi
    if [ -z "$c" ]; then c=$a; a=0; b=0; fi
    rem_s=$(( d*86400 + 10#$a*3600 + 10#$b*60 + 10#$c ))
fi
[ "$rem_s" -le 0 ] && rem_s=14400
ROUND_T0=$(date +%s)
MARGIN=420
echo "remaining walltime this round: ${rem_s}s" >> "$FARMLOG"

# Perlmutter GPU-node CPU affinity: GPU0<-cores 48-63, GPU1<-32-47, GPU2<-16-31, GPU3<-0-15
declare -A CPUS=( [0]="48-63,112-127" [1]="32-47,96-111" [2]="16-31,80-95" [3]="0-15,64-79" )
declare -A NUMA=( [0]=3 [1]=2 [2]=1 [3]=0 )
PIN=1
command -v numactl >/dev/null 2>&1 || PIN=0
echo "numa pinning: $PIN" >> "$FARMLOG"

# pop the first queued job whose est fits the remaining time; prints the line or nothing
next_job() {
    local remain=$1
    (
        flock 9
        local line est lineno=0 chosen="" chosenno=0
        while IFS= read -r line; do
            lineno=$((lineno+1))
            est=$(echo "$line" | cut -d'|' -f2)
            case $est in (*[!0-9]*|'') est=99999 ;; esac
            if [ "$est" -le "$remain" ]; then chosen="$line"; chosenno=$lineno; break; fi
        done < "$QUEUE"
        if [ -n "$chosen" ]; then
            sed -i "${chosenno}d" "$QUEUE"
            echo "$chosen"
        fi
    ) 9>"$QUEUE.lock"
}

worker() {
    local gpu=$1
    while :; do
        local now remain line
        now=$(date +%s)
        remain=$(( rem_s - (now - ROUND_T0) - MARGIN ))
        [ "$remain" -le 0 ] && { echo "[$(date '+%F %T')] GPU$gpu out of walltime, worker exits" >> "$FARMLOG"; break; }
        line=$(next_job "$remain")
        if [ -z "$line" ]; then
            if [ -s "$QUEUE" ]; then
                echo "[$(date '+%F %T')] GPU$gpu nothing fits in ${remain}s, worker exits" >> "$FARMLOG"
            fi
            break
        fi
        local name=${line%%|*}
        local rest=${line#*|}
        local cmd=${rest#*|}
        echo "[$(date '+%F %T')] GPU$gpu START $name (remain ${remain}s)" >> "$FARMLOG"
        local t0=$SECONDS
        if [ "$PIN" = 1 ]; then
            ( cd "$BASE/jobs/$name" && CUDA_VISIBLE_DEVICES=$gpu numactl --physcpubind=${CPUS[$gpu]} --membind=${NUMA[$gpu]} bash -c "$cmd" ) >> "$LOGS/$name.log" 2>&1
        else
            ( cd "$BASE/jobs/$name" && CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" ) >> "$LOGS/$name.log" 2>&1
        fi
        local rc=$?
        echo "[$(date '+%F %T')] GPU$gpu END $name rc=$rc elapsed=$((SECONDS-t0))s" >> "$FARMLOG"
        echo "$name rc=$rc elapsed=$((SECONDS-t0))s gpu=$gpu jobid=${SLURM_JOB_ID:-none}" >> "$LOGS/job_summary.txt"
    done
}

for g in 0 1 2 3; do
    worker "$g" &
done
wait
echo "${SLURM_JOB_ID:-none} $(date +%s)" >> $LOGS/round_ends.txt
n_done=$(wc -l < "$LOGS/job_summary.txt" 2>/dev/null || echo 0)
n_all=$(grep -cv '^$' "$ALLJOBS")
if [ "$n_done" -ge "$n_all" ]; then
    echo "FARM ALL DONE $(date)" >> "$FARMLOG"
else
    echo "FARM ROUND DONE, $((n_all - n_done)) jobs still pending $(date)" >> "$FARMLOG"
fi
