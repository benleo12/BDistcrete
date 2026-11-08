#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J lmx_applyT
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# D11: the refitted temperature of every model a later step of the paper reads through its export goes
# into the export in output/models (apply_T.py --lmx): a model whose temperature changed gets a file of
# its own there in place of the link to the published export, so the closure, learning-curve and
# learning-band steps of this rerun and the propagation steps (directlib, EXPORT=) all read the new
# value; a model whose temperature did not change keeps its link. Needs output/temperature/<tag>.json of
# every tag (the standard tests of specs/std_fac.txt and recal_lc_lmx). The exports of Stages D and E
# (E_P2v3fullfac0, F_P2v3fullfac0), which only the release reads, are applied by apply_T_rel_lmx.sbatch.
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/apply_T_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
TAGS=${TAGS:-"MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2 MIXGEOV3M8S9 MIXGEOV3S27F MIXGEOV3E20S9 MIXGEOV3E160S9 MIXGEOV3E320S9 MIXGEOV4S9 MIXGEOV4L C_P2augfullfac0"}
for T in $TAGS; do need_file output/temperature/$T.json output/models/${T}_cond.npz || exit 2; done
"$PY" -B -u apply_T.py --lmx $TAGS || { echo "APPLY FAILED"; exit 3; }
"$PY" -B - $TAGS <<'PY' || { echo "APPLY FAILED (an export does not carry its refit)"; exit 3; }
import sys, json, numpy as np
for t in sys.argv[1:]:
    T = json.load(open(f'output/temperature/{t}.json'))['T']; e = float(np.load(f'output/models/{t}_cond.npz')['temperature'])
    assert abs(T - e) < 1e-12, (t, T, e)
    print(f'{t}: refit {T}, export {e}')
PY
echo "APPLY DONE $(date)"
