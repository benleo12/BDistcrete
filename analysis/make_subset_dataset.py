#!/usr/bin/env python3
"""A cut-down copy of a pure-run data set for the learning curve: every STRIDE-th training run of
each generator, and/or only the first NEV events of every training run. Held-out runs are linked
unchanged, so every variant is tested on the same events.
    SRC=data_stagePURE17 DST=data_stagePURE17_ev40 NEV=40000 python make_subset_dataset.py
    SRC=data_stagePURE17 DST=data_stagePURE17_q STRIDE=4 python make_subset_dataset.py
"""
import os, json
import numpy as np

SRC, DST = os.environ['SRC'], os.environ['DST']
NEV = int(os.environ.get('NEV', '0')); STRIDE = int(os.environ.get('STRIDE', '1'))
m = json.load(open(f'{SRC}/meta.json'))
S = [r for r in m['train'] if r['n_herwig'] == 0]; H = [r for r in m['train'] if r['n_herwig'] > 0]
m['train'] = S[::STRIDE] + H[::STRIDE]
os.makedirs(DST, exist_ok=True)
for r in m['train'] + m['held']:
    rid = r['rid']; npz, csv = f'particles_full_{rid:04d}.npz', f'shapes_run_{rid:04d}.csv'
    for f in (npz, csv):
        if os.path.lexists(f'{DST}/{f}'):
            os.remove(f'{DST}/{f}')
    if NEV and r in m['train']:
        z = np.load(f'{SRC}/{npz}'); arr = {k: z[k][:NEV] for k in z.files}
        rows = open(f'{SRC}/{csv}').read().splitlines()[:NEV + 1]
        assert all(len(v) == len(rows) - 1 == NEV for v in arr.values()), rid
        np.savez_compressed(f'{DST}/{npz}', **arr)
        open(f'{DST}/{csv}', 'w').write('\n'.join(rows) + '\n')
        key = 'n_sherpa' if r['n_sherpa'] else 'n_herwig'; r[key] = NEV
    else:
        for f in (npz, csv):
            os.symlink(os.path.realpath(f'{SRC}/{f}'), f'{DST}/{f}')
m['n_train_events'] = int(min(r['n_sherpa'] + r['n_herwig'] for r in m['train']))
m['note'] += f' (learning curve: stride {STRIDE}, first {NEV or "all"} events of each training run)'
json.dump(m, open(f'{DST}/meta.json', 'w'), indent=1)
print(f'{DST}: {len(m["train"])} training runs x {m["n_train_events"]} events, {len(m["held"])} held')
