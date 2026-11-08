#!/bin/bash
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -J lmx_trunc
#SBATCH -o /mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code/logs/%x_%j.out
# Two counts of App. A.2, made again (no decision changes them, and nothing is copied from the published run):
#   \TruncMaxPercent, \TruncMinPercent, \TruncMaxMult: trunc_frac.py, the events with more than 100 particles over
#       the training runs of the final models, on the ORIGINAL shapes files (read only), whose mult_total counts
#       every particle, the beam photons included, as the network input does (D4 leaves the input alone; the _nb
#       files hold the multiplicity without them). BEAM_PHOTONS is unset for it.  -> output/truncation_fraction.json
#   the 82457 coefficients (main.tex, App. A.2): param_count.py, the event side of r2_ladder.py's network with one
#       output and the four kinematic inputs, and the same count from the layers of the trained Stage A trunks in
#       output/models (it stops if they differ)                                   -> output/param_count.json
# Each is compared with the published file (output/published/, read only, linked by setup_lmx.py). The outputs
# are removed when the job starts and recorded when it ends well (output/lmx_records/counts.json).
#   sbatch --export=ALL,MAXENT_DIR=$RUN symmetry/rerun/trunc_lmx.sbatch
set -uo pipefail
L=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code    # RUN, the only folder these scripts run in
[ "${MAXENT_DIR:-}" = "$L" ] || { echo "MAXENT_DIR must be $L (it is '${MAXENT_DIR:-unset}')"; exit 2; }
source "$MAXENT_DIR/symmetry/rerun/common_lmx.sh" || exit 2
P=/mnt/beegfs/bassi/reweighting       # the original folders of the published run, read only
MIXO=$P/maxent_code/data_stagePURE17_v3; ABCO=$P/gen/data_abc; DESO=$P/gen/designs
OUTS="output/truncation_fraction.json output/param_count.json"
rm -f $OUTS output/lmx_records/counts.json
need_file $MIXO/meta.json $ABCO $DESO/stageA_design_aug.csv output/published/truncation_fraction.json || exit 2
for d in $MIXO $ABCO; do
  case "$(realpath $d)" in *_nb|*_nb/*) echo "$d resolves to a photon-free (_nb) folder: the original is needed"; exit 2 ;; esac
done
TRUNKS="output/models/A_P2augfullfac0_trunk.pt output/models/A_P2augfullfac1_trunk.pt output/models/A_P2augfullfac2_trunk.pt"
need_file $TRUNKS || exit 2
echo "=== truncation fraction on $MIXO, $ABCO, $DESO $(date)"
env -u BEAM_PHOTONS OMP_NUM_THREADS=8 "$PY" -B -u trunc_frac.py $MIXO $ABCO $DESO output/truncation_fraction.json \
  || { echo "TRUNC FAILED"; exit 3; }
"$PY" -B - output/truncation_fraction.json output/published/truncation_fraction.json <<'PY'
import json, sys
new, pub = (json.load(open(f)) for f in sys.argv[1:3])
for k in sorted(set(new) | set(pub)):
    a, b = new.get(k, {}), pub.get(k, {})
    print(f'{k:22s} rerun: {a.get("events")} events, {a.get("over_100")} over 100, max {a.get("max")}, '
          f'{100*a.get("fraction", float("nan")):.4f}%   published: {b.get("events")}, {b.get("over_100")}, '
          f'max {b.get("max")}, {100*b.get("fraction", float("nan")):.4f}%')
print('truncation fraction: ' + ('IDENTICAL to the published file' if new == pub else 'DIFFERS from the published file'))
PY
echo "=== parameter count from r2_ladder.py and the trained Stage A trunks $(date)"
OMP_NUM_THREADS=4 PARAM_COUNT_OUT=output/param_count.json "$PY" -B -u param_count.py $TRUNKS || { echo "PARAM COUNT FAILED"; exit 3; }
if [ -e output/published/param_count.json ]; then
  "$PY" -B - output/param_count.json output/published/param_count.json <<'PY'
import json, sys
new, pub = (json.load(open(f)) for f in sys.argv[1:3])
same = all(new.get(k) == pub.get(k) for k in pub)
print(f'parameter count: K = 1 event side {new["K1"]["event_side_total"]}; '
      + ('every key of the published file IDENTICAL' if same else 'DIFFERS from the published file'))
PY
fi
step_record counts $OUTS || exit 3
echo "COUNTS DONE $(date)"
