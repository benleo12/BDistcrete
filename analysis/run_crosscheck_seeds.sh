#!/usr/bin/env bash
# Run-to-run spread of the PFN cross-check (App. B): five seeds of the energyflow PFN, of our
# conditional PFN collapsed to K=1, and of a plain PyTorch PFN with the energyflow architecture,
# one training at a time. Results in output/crosscheck_seeds/, summary at the end.
set -uo pipefail
cd "$(dirname "$0")"
EF_PY=${EF_PYTHON:-python}
for s in 1 2 3 4 5; do
  echo "=== seed $s energyflow $(date)"
  TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 CROSSCHECK_SEED=$s CROSSCHECK_OUT=output/crosscheck_seeds/ef_s$s.json \
    $EF_PY -u crosscheck_ef.py 2>&1 | grep -E "AUC|Error|error" ; echo "ef exit=${PIPESTATUS[0]}"
  echo "=== seed $s torch K=1 $(date)"
  CROSSCHECK_DEVICE=${CROSSCHECK_DEVICE:-mps} CROSSCHECK_SEED=$s CROSSCHECK_OUT=output/crosscheck_seeds/torchK1_s$s.json \
    python -u crosscheck_torch.py 2>&1 | grep -E "AUC|Error|error" ; echo "torch exit=${PIPESTATUS[0]}"
  echo "=== seed $s torch plain $(date)"
  CROSSCHECK_DEVICE=${CROSSCHECK_DEVICE:-mps} CROSSCHECK_SEED=$s CROSSCHECK_PLAIN=1 CROSSCHECK_OUT=output/crosscheck_seeds/torchplain_s$s.json \
    python -u crosscheck_torch.py 2>&1 | grep -E "AUC|Error|error" ; echo "plain exit=${PIPESTATUS[0]}"
done
python - <<'PY'
import json, glob, numpy as np
for k in ('ef', 'torchK1', 'torchplain'):
    a = [json.load(open(f))['auc'] for f in sorted(glob.glob(f'output/crosscheck_seeds/{k}_s*.json'))]
    if a: print(f'{k:11s} n={len(a)} mean AUC {np.mean(a):.4f} sd {np.std(a, ddof=1) if len(a) > 1 else 0:.4f}  {np.round(a, 4).tolist()}')
PY
echo "CROSSCHECK SEEDS DONE $(date)"
