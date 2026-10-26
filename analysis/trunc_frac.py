"""Fraction of events with more than 100 kept particles (the input keeps the 100 most energetic),
over the training runs of the final models, from the full multiplicity in the shape files: the
mixture of the 720 runs (data_stagePURE17_v3) and the densified designs of Stages A to C
(gen/data_abc, training runs of gen/designs/stage<S>_design_aug.csv). Writes
output/truncation_fraction.json, which make_numbers.py reads. Run from analysis/.

    python trunc_frac.py
"""
import json, numpy as np, pandas as pd
GEN = '../gen'
sets = {'mixture (720 runs)': ['data_stagePURE17_v3/shapes_run_%04d.csv' % t['rid'] for t in json.load(open('data_stagePURE17_v3/meta.json'))['train']]}
for S in 'ABC':
    d = pd.read_csv(f'{GEN}/designs/stage{S}_design_aug.csv')
    sets[f'Stage {S}'] = [f'{GEN}/data_abc/shapes_run_{r:04d}.csv' for r in d[d.role == 'train'].run_id]
out = {}
for name, files in sets.items():
    n = over = 0; mx = 0
    for f in files:
        m = pd.read_csv(f, usecols=['mult_total'])['mult_total'].values
        n += len(m); over += int((m > 100).sum()); mx = max(mx, int(m.max()))
    out[name] = dict(events=n, over_100=over, fraction=over/n, max=mx)
    print(name, n, over, '%.4f%%' % (100*over/n), 'max', mx, flush=True)
json.dump(out, open('output/truncation_fraction.json', 'w'), indent=1)
