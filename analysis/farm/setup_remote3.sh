#!/bin/bash
# setup_remote3.sh -- restructure to single-K rank jobs + split concat; queue lines
# now carry a walltime estimate: name|est_s|cmd
set -e
BASE=$SCRATCH/energyflower
cd "$BASE"

echo "--- remove superseded job dirs"
rm -rf jobs/_bench jobs/rank_C_k12 jobs/rank_C_k34 jobs/rank_C_k68 jobs/rank_C_k16 \
       jobs/rank_A_k12 jobs/rank_A_k34 jobs/rank_A_k68 jobs/rank_A_k16 \
       jobs/rank_B_k12 jobs/rank_B_k34 jobs/rank_B_k68 jobs/rank_B_k16 jobs/concat

echo "--- building job dirs"
RANKJOBS=""
for S in C A B; do
    for KV in 1 2 3 4 6 8 16; do
        RANKJOBS="$RANKJOBS rank_${S}_K${KV}"
    done
done
JOBS="$RANKJOBS widebox widebox_ctrl concat_A concat_B b_r200 mass_B mass_C k10_C k6_B k3_A showers disjoint"
for j in $JOBS; do
    mkdir -p jobs/$j/output
    cp code/*.py jobs/$j/
    for d in data_stageA_full data_stageB_full data_stageC data_widebox data_stageD3; do
        ln -sfn $BASE/data/$d jobs/$j/$d
    done
done
mkdir -p jobs/disjoint/output/models
ln -sfn $BASE/data/prod_models/A_ref.npz  jobs/disjoint/output/models/A_ref.npz
ln -sfn $BASE/data/prod_models/A_cond.npz jobs/disjoint/output/models/A_cond.npz

echo "--- writing all_jobs.txt (name|est_s|cmd)"
{
for S in C A B; do
    case $S in C) EST=6000;; *) EST=5400;; esac
    for KV in 1 2 3 4 6 8 16; do
        echo "rank_${S}_K${KV}|$EST|RANK_STAGES=$S RANK_KS=$KV RANK_ENS=4 python r3_rank.py"
    done
done
cat <<'EOQ'
widebox|9600|python r2b_widebox.py
widebox_ctrl|9600|python r2b_widebox_ctrl.py
concat_A|6000|python concat_baseline.py A
concat_B|6600|python concat_baseline.py B
b_r200|7200|LADDER_STAGES=B LADDER_SEED_OFFSET=200 LADDER_SUFFIX=_r200 LADDER_OUT=output/ladder_B_r200.json python r2_ladder.py
mass_B|7200|LADDER_STAGES=B LADDER_FEATS=mass LADDER_SEED_OFFSET=100 LADDER_SUFFIX=_mass LADDER_OUT=output/ladder_B_mass.json python r2_ladder.py
mass_C|7800|LADDER_STAGES=C LADDER_FEATS=mass LADDER_SUFFIX=_mass LADDER_OUT=output/ladder_C_mass.json python r2_ladder.py
k10_C|7800|LADDER_STAGES=C LADDER_K=10 LADDER_SUFFIX=_k10 LADDER_OUT=output/rank_v2_C.json python r2_ladder.py
k6_B|7200|LADDER_STAGES=B LADDER_K=6 LADDER_SEED_OFFSET=100 LADDER_SUFFIX=_k6 LADDER_OUT=output/rank_v2_B.json python r2_ladder.py
k3_A|6600|LADDER_STAGES=A LADDER_K=3 LADDER_SUFFIX=_k3 LADDER_OUT=output/rank_v2_A.json python r2_ladder.py
disjoint|5400|python disjoint_score.py
showers|3600|python r2d_showers.py
EOQ
} > jobs/all_jobs.txt
rm -f jobs/queue.txt jobs/queue.txt.lock
grep -cv '^$' jobs/all_jobs.txt
echo "SETUP3 DONE"
