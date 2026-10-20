#!/bin/bash
# The wide coupling scan of App. B at the final standard (Benoit, Oct 18: redo overnight).
# Data: the coupling alone over [0.08, 0.20], 21 training runs spaced by 0.006 (9000-9020) and six
# held-out runs (9100-9105 at 0.083, 0.107, 0.131, 0.155, 0.179, 0.197), 80k events each, the Stage A
# card of gen_stageABC.sh (Sherpa 3.0.4, CSS, Ahadic, every other parameter at its default), into
# gen/data_widebox. Same run ids and points as the published scan.
# Networks: three references of 63,000 events each, the paper's training standard (train_p2.sbatch:
# K 48, 324k steps, no patience, four bootstrap members, fit fraction 0.2, best stop-split
# checkpoint, temperature on the stop split), the stage built from a design file (STAGEW_CSV):
#   single  the run at 0.14 only (LADDER_REF_RUNS=9010), trained on all 21 runs
#   pool    the union of all 21 runs, 3000 events each
#   local   per held-out point, the union of its 7 nearest runs, 9000 events each, trained on those 7
# The reference weights at every held-out point are saved for the weight panel (LADDER_DUMP_WEIGHTS).
#   bash widebox_submit.sh
set -u; export PATH=/cm/shared/apps/slurm/current/bin:$PATH
R=/mnt/beegfs/bassi/reweighting; cd $R
source env/torch/bin/activate
python3 - <<'EOF'
import csv
R = '/mnt/beegfs/bassi/reweighting'
ASV = [round(0.080 + 0.006*i, 4) for i in range(21)]
TRAIN = {9000 + i: a for i, a in enumerate(ASV)}
HELD = {9100: 0.083, 9101: 0.107, 9102: 0.131, 9103: 0.155, 9104: 0.179, 9105: 0.197}
with open(f'{R}/gen/tasks/widebox.txt', 'w') as f:
    f.write('# wide coupling scan: 21 training runs and six held-out runs, the Stage A card, 80k events\n')
    for rid, a in list(TRAIN.items()) + list(HELD.items()):
        f.write(f'gen_stageABC.sh {rid} {a:.4f} - - - 80000\n')
def design(path, train, held):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['run_id', 'role', 'alphas'])
        for rid, a in train.items(): w.writerow([rid, 'train', f'{a:.4f}'])
        for rid, a in held.items(): w.writerow([rid, 'held', f'{a:.4f}'])
design(f'{R}/gen/designs/stageW_design.csv', TRAIN, HELD)
for rid, a in HELD.items():
    near = sorted(TRAIN, key=lambda r: abs(TRAIN[r] - a))[:7]
    design(f'{R}/gen/designs/stageW_local_{rid}.csv', {r: TRAIN[r] for r in sorted(near)}, {rid: a})
print('tasks and designs written')
EOF
G=$(sbatch --parsable --array=0 --export=ALL,TASKS=gen/tasks/widebox.txt,GEN_DATA=gen/data_widebox gen/gen_batch.sbatch)
echo "generation $G"
cd $R/maxent_code
export K=48 STEPS=324000 PATIENCE=0 GPU_FETCH=1 MMAP_BASE=/dev/shm STAGEW_DATA=$R/gen/data_widebox LADDER_DUMP_WEIGHTS=1 LADDER_NREF_TOTAL=63000
sub() {  # DESIGN_TAG, design csv, reference runs
  DESIGN_TAG=$1 STAGEW_CSV=$R/gen/designs/$2 LADDER_REF_RUNS=$3 SPECS="W:fac:0" \
    sbatch --parsable --array=0 -c 6 --mem=64G -J widebox --dependency=afterok:$G --export=ALL train_p2.sbatch
}
echo "single $(sub wbsingle stageW_design.csv 9010)"
echo "pool   $(sub wbpool stageW_design.csv '')"
for rid in 9100 9101 9102 9103 9104 9105; do echo "local $rid $(sub wblocal$rid stageW_local_$rid.csv '')"; done
