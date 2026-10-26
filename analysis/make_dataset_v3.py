#!/usr/bin/env python3
"""The v3 pure-run training set on Symmetry: the 224 released runs (regenerated, gen/data_release)
plus the v3 runs (gen/data_v3), linked into maxent_code/data_stagePURE17_v3/ with a meta.json in
the format of make_pure_dataset.py, so stage_mixture.py trains on it unchanged.

Boxes, axis orders and the placeholder centres are those of data_stagePURE17 (designs/PURE17_meta.json),
and the theta of every released run is checked to equal the one in that meta. Held-out records carry
held_set = 'release' (the 16 points of the paper) or 'v3' (the 32 new points).

Options (environment):
  V3_SETS=release[,v3][,v4]   which runs to include (default release,v3)
  V3_TRAIN_EVENTS=N      keep only the first N events of every TRAINING run (written as real
                         files), for the events-per-run learning curve; held-out runs untouched
  V3_HELD_EXT=K          append the extensions gen/data_ext/k1..kK to every HELD-OUT run (real
                         files), for a sharper closure test
  V3_TRAIN_EXT=K         append the extensions gen/data_trainext/k1..kK (seed rid + k*10^6, the
                         held-out convention) to every TRAINING run (real files), for the
                         events-per-run learning curve above 80k
    python make_dataset_v3.py        (run in gen/ on Symmetry)
"""
import os, csv, json
import numpy as np

MAXENT_DIR = os.environ.get('MAXENT_DIR', '/mnt/beegfs/bassi/reweighting/maxent_code')
R = os.path.dirname(MAXENT_DIR)
DST = os.environ.get('V3_DST', f'{MAXENT_DIR}/data_stagePURE17_v3')
ref = json.load(open(f'{R}/gen/designs/PURE17_meta.json'))
s_axes, h_axes, s_box, h_box = ref['s_axes'], ref['h_axes'], ref['s_box'], ref['h_box']
s_cen = [0.5*(lo + hi) for lo, hi in s_box]; h_cen = [0.5*(lo + hi) for lo, hi in h_box]
ref_theta = {x['rid']: x['theta'] for x in ref['train'] + ref['held']}
SOURCES = [  # (data dir, design, side, set)
    (f'{R}/gen/data_release', f'{R}/gen/designs/stageE_design.csv', 'S', 'release'),
    (f'{R}/gen/data_release', f'{R}/gen/designs/stageF_design_aug.csv', 'H', 'release'),
    (f'{R}/gen/data_v3', f'{R}/gen/designs/stageE_design_v3.csv', 'S', 'v3'),
    (f'{R}/gen/data_v3', f'{R}/gen/designs/stageF_design_v3.csv', 'H', 'v3'),
    (f'{R}/gen/data_v4', f'{R}/gen/designs/stageE_design_v4.csv', 'S', 'v4'),
    (f'{R}/gen/data_v4', f'{R}/gen/designs/stageF_design_v4.csv', 'H', 'v4'),
]
SETS = os.environ.get('V3_SETS', 'release,v3').split(',')   # 'release' alone rebuilds data_stagePURE17
NTRAIN_EV = int(os.environ.get('V3_TRAIN_EVENTS', '0'))
HELD_EXT = int(os.environ.get('V3_HELD_EXT', '0'))
TRAIN_EXT = int(os.environ.get('V3_TRAIN_EXT', '0'))
SOURCES = [x for x in SOURCES if x[3] in SETS]
os.makedirs(DST, exist_ok=True)
meta = dict(train=[], held=[], seed=0, s_axes=s_axes, h_axes=h_axes, s_box=s_box, h_box=h_box,
            ntheta=len(s_axes) + len(h_axes) + 1, s_src='gen/data_release + gen/data_v3',
            h_src='gen/data_release + gen/data_v3',
            note=f'v3: released pure runs (regenerated on Symmetry) plus the v3 design; sets {SETS}, '
                 f'training events {NTRAIN_EV or "all"}, training extensions {TRAIN_EXT}, '
                 f'held-out extensions {HELD_EXT}')
for data, design, side, sset in SOURCES:
    axes = s_axes if side == 'S' else h_axes; box = np.array(s_box if side == 'S' else h_box)
    for r in csv.DictReader(open(design)):
        if r['role'] not in ('train', 'held'):
            continue
        rid = int(r['run_id']); th = [float(r[a]) for a in axes]
        assert ((np.array(th) >= box[:, 0] - 1e-9) & (np.array(th) <= box[:, 1] + 1e-9)).all(), (rid, th)
        theta = (th + h_cen + [0.0]) if side == 'S' else (s_cen + th + [1.0])
        if sset == 'release':
            assert np.allclose(theta, ref_theta[rid], rtol=0, atol=1e-12), f'theta of {rid} differs from PURE17'
        held = r['role'] == 'held'
        srcs = [data] + ([f'{R}/gen/data_ext/k{k}' for k in range(1, HELD_EXT + 1)] if held else
                         [f'{R}/gen/data_trainext/k{k}' for k in range(1, TRAIN_EXT + 1)])
        cut = NTRAIN_EV if (NTRAIN_EV and not held) else 0
        for b in (f'{DST}/particles_full_{rid}.npz', f'{DST}/shapes_run_{rid}.csv'):
            if os.path.lexists(b):
                os.remove(b)
        if len(srcs) == 1 and not cut:
            for f in (f'particles_full_{rid}.npz', f'shapes_run_{rid}.csv'):
                assert os.path.exists(f'{data}/{f}'), f'missing {data}/{f}'
                os.symlink(f'{data}/{f}', f'{DST}/{f}')
        else:   # truncated or merged: real files, arrays and csv rows in the same event order
            zs = [np.load(f'{d}/particles_full_{rid}.npz') for d in srcs]
            arr = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
            csvs = [open(f'{d}/shapes_run_{rid}.csv').read().splitlines() for d in srcs]
            rows = [csvs[0][0]] + [x for c in csvs for x in c[1:]]
            if cut:
                arr = {k: v[:cut] for k, v in arr.items()}; rows = rows[:cut + 1]
            assert all(len(v) == len(rows) - 1 for v in arr.values()), f'{rid}: npz and csv lengths differ'
            np.savez_compressed(f'{DST}/particles_full_{rid}.npz', **arr)
            open(f'{DST}/shapes_run_{rid}.csv', 'w').write('\n'.join(rows) + '\n')
        n = len(np.load(f'{DST}/particles_full_{rid}.npz')['mask'])
        rec = dict(rid=rid, theta=theta, n_sherpa=n if side == 'S' else 0, n_herwig=0 if side == 'S' else n)
        if r['role'] == 'held':
            rec['held_set'] = sset
        meta['train' if r['role'] == 'train' else 'held'].append(rec)
meta['n_train_events'] = int(min(x['n_sherpa'] + x['n_herwig'] for x in meta['train']))
meta['n_held_events'] = int(min(x['n_sherpa'] + x['n_herwig'] for x in meta['held']))
json.dump(meta, open(f'{DST}/meta.json', 'w'), indent=1)
nS = sum(1 for x in meta['train'] if x['n_herwig'] == 0)
print(f'{DST}: {len(meta["train"])} training runs ({nS} Sherpa, {len(meta["train"]) - nS} Herwig), '
      f'{len(meta["held"])} held ({sum(x["held_set"] == "release" for x in meta["held"])} release), '
      f'{meta["n_train_events"]} events per run, d={meta["ntheta"]}')
