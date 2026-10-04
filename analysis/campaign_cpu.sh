#!/bin/bash
# CPU evaluation chain on the FINAL models, in the provenance map's dependency order.
# Fires once the local Stage C production training exits. Every step logs; a nonzero exit
# does not stop the chain (later steps that depend on a failed one will fail loudly too,
# and the log preserves the causality).
set -u
cd .
PY=python
LOG=output/campaign_cpu.log
export PYTHONUNBUFFERED=1 LADDER_ACT=silu
step () { echo "=== $(date '+%F %T') $1 ===" >> $LOG; }

while pgrep -f "r2_ladder.py" > /dev/null; do sleep 60; done
step "Stage C training done; starting CPU chain"

step "1 reeval_bins (support binning re-eval, canonical Stage C widths)"
$PY reeval_bins.py >> $LOG 2>&1; step "exit=$?"
step "2 eval_valid (12-run any-point map, 8 judges)"
$PY eval_valid.py >> $LOG 2>&1; step "exit=$?"
step "3 tail_stress (3 tails x 12 points)"
$PY tail_stress.py >> $LOG 2>&1; step "exit=$?"
step "4 floor_eval_gen (80k vs 800k floor)"
$PY floor_eval_gen.py --out output/floor_eval_final.json >> $LOG 2>&1; step "exit=$?"
step "5 accuracy_law_gen (closed-loop law)"
$PY accuracy_law_gen.py --out output/accuracy_law_final.json >> $LOG 2>&1; step "exit=$?"
step "6 ensemble_ablation (1 vs 2 vs 4 members, refit T)"
$PY ensemble_ablation.py >> $LOG 2>&1; step "exit=$?"
step "7 xp_closure (fragmentation spectrum)"
$PY xp_closure.py >> $LOG 2>&1; step "exit=$?"
step "8 maxent tilt on the new Stage B cache"
$PY maxent_tilt_fixed.py >> $LOG 2>&1; step "exit=$?"
step "9 joint covariance on final models"
$PY joint_covariance.py --models output/models --stages BC \
    --out output/joint_covariance_final.json >> $LOG 2>&1; step "exit=$?"
step "10 alpha_s extraction on final models"
$PY alphas_extraction.py --models output/models \
    --out output/alphas_extraction_final.json >> $LOG 2>&1; step "exit=$?"
echo "CPU CHAIN FINISHED $(date '+%F %T')" >> $LOG
