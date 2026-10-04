#!/bin/bash
# FINAL-RECIPE RECOMPUTE CAMPAIGN, GPU phase. Recipe: published architecture + pooled-sum
# rescale (LADDER_EMB_SCALE=auto), pure bilinear head, SiLU, ENS=4, STEPS=36000, K=24.
# Old models backed up in output/models/prepub_backup/. Standard output names so the whole
# downstream evaluation chain runs unchanged. Per-stage seeds: A,C offset 0; B offset 100
# (seed 3 of the 0-3 draw is reproducibly dead on Stage B under the OLD conditioning; with
# the fix it should be fine, but B keeps the validated clean set for continuity).
set -u
cd .
PY=python
LOG=output/campaign_gpu.log
export PYTHONUNBUFFERED=1 LADDER_ACT=silu LADDER_EMB_SCALE=auto
step () { echo "=== $(date '+%F %T') $1 ===" >> $LOG; }

step "1a: production ladder A (seeds 0-3)"
LADDER_STAGES=A $PY r2_ladder.py >> $LOG 2>&1; step "exit=$?"
step "1b: production ladder B (seeds 100-103)"
LADDER_STAGES=B LADDER_SEED_OFFSET=100 $PY r2_ladder.py >> $LOG 2>&1; step "exit=$?"
step "1c: production ladder C (seeds 0-3)"
LADDER_STAGES=C $PY r2_ladder.py >> $LOG 2>&1; step "exit=$?"

step "2: rank scans A+B (ENS_SCAN=2, fixed recipe)"
RANK_STAGES=AB $PY r3_rank.py >> $LOG 2>&1; step "exit=$?"

step "3a: physical rank A@K=3"
LADDER_STAGES=A LADDER_K=3  LADDER_SUFFIX=_k3  LADDER_OUT=output/rank_production_v2.json $PY r2_ladder.py >> $LOG 2>&1; step "exit=$?"
step "3b: physical rank B@K=6 (seeds 100-103)"
LADDER_STAGES=B LADDER_K=6  LADDER_SEED_OFFSET=100 LADDER_SUFFIX=_k6  LADDER_OUT=output/rank_production_v2.json $PY r2_ladder.py >> $LOG 2>&1; step "exit=$?"
step "3c: physical rank C@K=10"
LADDER_STAGES=C LADDER_K=10 LADDER_SUFFIX=_k10 LADDER_OUT=output/rank_production_v2.json $PY r2_ladder.py >> $LOG 2>&1; step "exit=$?"

step "4: concatenation baseline A+B (same conditioning fix)"
$PY concat_baseline.py >> $LOG 2>&1; step "exit=$?"

step "5: wide-box three schemes"
$PY r2b_widebox.py >> $LOG 2>&1; step "exit=$?"
$PY r2b_widebox_ctrl.py >> $LOG 2>&1; step "exit=$?"

step "6: shower pairs"
$PY r2d_showers.py >> $LOG 2>&1; step "exit=$?"

step "7: disjoint-data score reproducibility"
$PY disjoint_score.py >> $LOG 2>&1; step "exit=$?"

echo "GPU CAMPAIGN FINISHED $(date '+%F %T')" >> $LOG
