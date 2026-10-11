#!/usr/bin/env python3
"""The pure-run data set of the multiplicative mixture: every training and held-out run of the
Sherpa design (Stage D of the paper, data_stageE) and of the Herwig design (Stage E, data_stageF),
linked into one directory with a meta.json in the mixture format, so stage_mixture.py trains the
seventeen-parameter head on them with LADDER_HEAD_KIND=geometric and LADDER_TARGET_PURE=1.

A Sherpa run carries theta = (theta_S, centre of the Herwig box, 0) and a Herwig run
theta = (centre of the Sherpa box, theta_H, 1). The absent generator's block never enters the
head at the pure endpoints, so the centre is a placeholder. The boxes and axis orders are those of
the additive mixture data set, which was built from the same two designs.

    python make_pure_dataset.py            # writes data_stagePURE17/
"""
import os, csv, json, sys
import numpy as np

DST = os.environ.get('PURE_DST', 'data_stagePURE17')
S_DATA, S_CSV = 'data_stageE', 'data_stageE/stageE_design.csv'
H_DATA, H_CSV = 'data_stageF', 'data_stageF/stageF_design_aug.csv'
REF_META = os.environ.get('MIX_META', 'data_stageDM17aug_v2/meta.json')

m = json.load(open(REF_META))
s_axes, h_axes, s_box, h_box = m['s_axes'], m['h_axes'], m['s_box'], m['h_box']
s_cen = [0.5*(lo + hi) for lo, hi in s_box]; h_cen = [0.5*(lo + hi) for lo, hi in h_box]
os.makedirs(DST, exist_ok=True)
meta = dict(train=[], held=[], seed=0, s_axes=s_axes, h_axes=h_axes, s_box=s_box, h_box=h_box,
            ntheta=len(s_axes) + len(h_axes) + 1, n_train_events=80000, n_held_events=80000,
            s_src=S_DATA, h_src=H_DATA, note='pure runs of both generators, for the multiplicative mixture')


def link(src, rid):
    for f in (f'particles_full_{rid:04d}.npz', f'shapes_run_{rid:04d}.csv'):
        a, b = os.path.abspath(f'{src}/{f}'), f'{DST}/{f}'
        assert os.path.exists(a), a
        if os.path.islink(b) or os.path.exists(b):
            os.remove(b)
        os.symlink(a, b)
    n = len(np.load(f'{src}/particles_full_{rid:04d}.npz')['mask'])
    return n


for data, csvp, axes, side in ((S_DATA, S_CSV, s_axes, 'S'), (H_DATA, H_CSV, h_axes, 'H')):
    rows = [r for r in csv.DictReader(open(csvp)) if r['role'] in ('train', 'held')]
    box = np.array(s_box if side == 'S' else h_box)
    for r in rows:
        rid = int(r['run_id']); th = [float(r[a]) for a in axes]
        assert ((np.array(th) >= box[:, 0] - 1e-9) & (np.array(th) <= box[:, 1] + 1e-9)).all(), (rid, th)
        theta = (th + h_cen + [0.0]) if side == 'S' else (s_cen + th + [1.0])
        n = link(data, rid)
        rec = dict(rid=rid, theta=theta, n_sherpa=n if side == 'S' else 0, n_herwig=0 if side == 'S' else n)
        meta['train' if r['role'] == 'train' else 'held'].append(rec)
ntr = [x['n_sherpa'] + x['n_herwig'] for x in meta['train']]
meta['n_train_events'] = int(min(ntr)); meta['n_held_events'] = int(min(x['n_sherpa'] + x['n_herwig'] for x in meta['held']))
json.dump(meta, open(f'{DST}/meta.json', 'w'), indent=1)
nS = sum(1 for x in meta['train'] if x['n_herwig'] == 0); nH = len(meta['train']) - nS
print(f'{DST}: {len(meta["train"])} training runs ({nS} Sherpa, {nH} Herwig), {len(meta["held"])} held, '
      f'{meta["n_train_events"]} events per run, d={meta["ntheta"]}')
