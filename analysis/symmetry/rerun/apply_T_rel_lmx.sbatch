#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J lmx_applyTrel
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# D11 for the release: the refitted temperatures of Stages D and E (E_P2v3fullfac0 and F_P2v3fullfac0,
# public names E and F) go into their exports in output/models, as apply_T_lmx.sbatch does for the paper's
# models (apply_T.py --lmx, the same guard). Only the release reads these exports (package_lmx.sbatch and
# table_lmx.sbatch, through package_release.py, make_stage_table.py and precision_gap.py); no step of the
# paper's chain does, so none waits for this job. Submitted with the scoring (afterany std_fac and recal);
# its record output/lmx_records/applyTrel.json tells the release jobs that it ended well in this rerun.
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/apply_T_rel_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
TAGS="E_P2v3fullfac0 F_P2v3fullfac0"
rm -f output/lmx_records/applyTrel.json
for T in $TAGS; do need_file output/temperature/$T.json output/models/${T}_cond.npz || exit 2; done
"$PY" -B -u apply_T.py --lmx $TAGS || { echo "APPLY FAILED"; exit 3; }
"$PY" -B - $TAGS <<'PY' || { echo "APPLY FAILED (an export does not carry its refit)"; exit 3; }
import sys, json, numpy as np
for t in sys.argv[1:]:
    T = json.load(open(f'output/temperature/{t}.json'))['T']; e = float(np.load(f'output/models/{t}_cond.npz')['temperature'])
    assert abs(T - e) < 1e-12, (t, T, e)
    print(f'{t}: refit {T}, export {e}')
PY
step_record applyTrel output/temperature/E_P2v3fullfac0.json output/temperature/F_P2v3fullfac0.json || exit 3
echo "APPLY (RELEASE) DONE $(date)"
