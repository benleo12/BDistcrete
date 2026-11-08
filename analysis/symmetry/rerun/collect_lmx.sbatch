#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -J lmx_collect
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# The end of the scoring side (after every other job of submit_scoring_lmx.sh):
#   the K scan of Fig. 6 (kscan_collect.py, every cell required, K = 48 the Table 1 cells) -> output/rank_scan_final.json,
#   the inputs of the wide scan (widebox_collect.py, from the wide-scan scorings and their weight dumps)
#       -> output/widebox_wb18.json and output/wb18_weights_{single,union}_0.083.npy;
#   both refuse a result without the fingerprint of the current mark and code (LMX_CHECK=1);
#   then the check of every scoring result (lmx_fingerprint.py check --all): the 196 scorings of specs/*.txt, each
#   with the fingerprint of its own line, what they wrote beside them, and the record of every step. Only when
#   all of it passes it writes output/scoring_check.json (I4), which make_numbers.py (SUFFIX=_lin) requires.
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/collect_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
OUTS="output/rank_scan_final.json output/widebox_wb18.json output/wb18_weights_single_0.083.npy output/wb18_weights_union_0.083.npy"
rm -f $OUTS output/lmx_records/collect.json output/scoring_check.json
LMX_CHECK=1 KSCAN_REQUIRE=1 "$PY" -B -u kscan_collect.py || { echo "KSCAN COLLECT FAILED"; exit 3; }
LMX_CHECK=1 "$PY" -B -u widebox_collect.py || { echo "WIDEBOX COLLECT FAILED"; exit 3; }
step_record collect $OUTS || exit 3
"$PY" -B symmetry/rerun/lmx_fingerprint.py check --all --write || { echo "SCORING CHECK FAILED: no output/scoring_check.json"; exit 3; }
echo "COLLECT DONE $(date)"
