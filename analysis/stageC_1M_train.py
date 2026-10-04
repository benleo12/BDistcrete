#!/usr/bin/env python3
"""Stage C retrain on the UNION of the original and the 1M-campaign reference runs.

Same frozen recipe as the published ladder (r2_ladder.run_stage, untouched): same
architecture, rank, step budget, stopping rule, held-out points 6900-6904. Only the
train-run list grows: each 3x3x3 grid point now has TWO runs, the original 80k-event
run (6800+i) and the fresh 40k-event run (7800+i, new seeds, same Sherpa v3.0.4 card),
so 120k training events per point. The exported reference pool subsamples 20k events
per train run: 54 runs -> 1.08M pooled reference events (vs 324k before).

The 7900-7904 runs are NOT used here at all: they stay untouched as fully independent
validation runs for after training.

Outputs: output/models/C_1M_cond.npz / C_1M_ref.npz / C_1M_trunk.pt
         output/stageC_1M_result.json
"""
import os, json
os.environ['LADDER_SUFFIX'] = '_1M'
import r2_ladder as R

R.NREF_PER = 20000
# earlier checkpoints added: with 54 runs the first stock checkpoint (6000) was already
# past the union optimum, and a selection pinned to the grid edge measures the grid
R.CKPTS = [2000, 4000] + R.CKPTS
grid = [(a, s, k) for a in [0.112, 0.120, 0.128]
        for s in [0.30, 0.46, 0.65] for k in [0.80, 1.21, 1.80]]
R.STAGES['C']['train'].update({7800+i: t for i, t in enumerate(grid)})
print(f"train runs: {len(R.STAGES['C']['train'])}, NREF_PER {R.NREF_PER}, "
      f"device {R.DEV}", flush=True)
result = R.run_stage('C')
json.dump(result, open('output/stageC_1M_result.json', 'w'), indent=1)
print('STAGE C 1M RETRAIN DONE')
