#!/usr/bin/env bash
set -uo pipefail
cd .
echo "=== energyflow PFN (TensorFlow) $(date) ==="
TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 \
  ${EF_PYTHON:-python} -u crosscheck_ef.py 2>&1 | grep -vE "oneDNN|cpu_feature|WARNING:absl|UserWarning|warnings.warn"
echo "ef exit=$?"
echo "=== our PyTorch PFN (CPU) $(date) ==="
CROSSCHECK_DEVICE=cpu python -u crosscheck_torch.py 2>&1 | grep -vE "Deprecat|Warning"
echo "torch exit=$?"
echo "=== COMPARISON ==="
python -c "
import json
a=json.load(open('output/crosscheck_ef.json')); b=json.load(open('output/crosscheck_torch.json'))
print(f\"energyflow PFN (TF) : AUC {a['auc']:.4f}\")
print(f\"ours (PyTorch)      : AUC {b['auc']:.4f}\")
d=abs(a['auc']-b['auc']); print(f\"difference          : {d:.4f}\")
print('AGREE' if d<0.01 else 'DISAGREE - investigate')
"
