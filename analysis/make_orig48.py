"""data_stagePURE17_orig48 and data_stagePURE17_orig48_h400: the 208-run release set
(data_stagePURE17_orig, its training runs, box and normalization unchanged) with the 48 held-out
points of the 720-run set in place of its own 16, at 80k events (from data_stagePURE17_v3) and at
400k events (from data_stagePURE17_v3_h400). Every row of the learning curve can then be scored at
the same 48 points. Run from analysis/.

    python make_orig48.py
"""
import json, os
import numpy as np

SRC = 'data_stagePURE17_orig'
a = json.load(open(f'{SRC}/meta.json'))
box = np.array(a['s_box'] + a['h_box'])
for HELD, DST, nev in (('data_stagePURE17_v3', 'data_stagePURE17_orig48', 80000),
                       ('data_stagePURE17_v3_h400', 'data_stagePURE17_orig48_h400', 400000)):
    b = json.load(open(f'{HELD}/meta.json'))
    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):
        os.remove(f'{DST}/{f}')
    for t in a['train']:
        rid = t['rid']
        for f in (f'particles_full_{rid:04d}.npz', f'shapes_run_{rid:04d}.csv'):
            os.symlink(os.path.realpath(f'{SRC}/{f}'), f'{DST}/{f}')
    for h in b['held']:
        rid = h['rid']; th = np.array(h['theta'][:16])
        inside = np.all((th >= box[:, 0] - 1e-9) & (th <= box[:, 1] + 1e-9))
        assert inside, f'held point {rid} lies outside the box of the 208-run set'
        for f in (f'particles_full_{rid:04d}.npz', f'shapes_run_{rid:04d}.csv'):
            os.symlink(os.path.realpath(f'{HELD}/{f}'), f'{DST}/{f}')
    m = dict(a); m['held'] = b['held']; m['n_held_events'] = nev
    m['note'] = a.get('note', '') + f' | the 48 held-out points of {HELD} in place of the 16'
    json.dump(m, open(f'{DST}/meta.json', 'w'))
    print(DST, 'train', len(m['train']), 'held', len(m['held']), 'files', len(os.listdir(DST)))
