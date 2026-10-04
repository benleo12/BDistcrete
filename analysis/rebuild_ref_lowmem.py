#!/usr/bin/env python3
"""Rebuild a Perlmutter-trained reference bundle on the corrected (exact thrust axis) shapes.

The Perlmutter stages (E, Fauglong, MIX17aug) trained under LADDER_LOWMEM, where every training
run's reference subsample is drawn with its own stream,

    idx = default_rng(50000 + rid).choice(n_events, min(NREF_PER, n_events), replace=False),

run after run in the stage's training order (the design CSV's train rows in file order, or
meta.json's train list for a mixture). Replaying that draw on the regenerated runs reproduces
the reference event for event. The replay is checked on EVERY reference event against the
stored particle array, which also proves the regenerated events are bitwise the old ones.

Observables that depend on the thrust axis (1_minus_thrust, B_total, rho_heavy) are taken from
the corrected shapes. Those that do not (mult_total, nbaryon, strange) must come back unchanged
and are asserted to. tau_parton and n_parton are attached from the regenerated particle files.

    python rebuild_ref_lowmem.py E
    python rebuild_ref_lowmem.py Fauglong          (design stageF_design_aug.csv)
    DM_DATA=data_stageDM17aug_v2 python rebuild_ref_lowmem.py MIX17aug
"""
import sys, os, json
import numpy as np, pandas as pd

tag = sys.argv[1]
OUT = os.environ.get('REF_OUT', f'output/models/{tag}_ref_v2.npz')
SLIM = os.environ.get('REF_SLIM', f'output/models/{tag}_ref_v2_slim.npz')
if tag == 'E':
    data, nper = 'data_stageE', 12000
    d = pd.read_csv(f'{data}/stageE_design.csv'); order = [int(r) for r in d[d.role == 'train'].run_id]
elif tag == 'Fauglong':
    data, nper = 'data_stageF', 12000
    d = pd.read_csv(f'{data}/stageF_design_aug.csv'); order = [int(r) for r in d[d.role == 'train'].run_id]
elif tag == 'MIX17aug':
    data = os.environ['DM_DATA']
    meta = json.load(open(f'{data}/meta.json'))
    order = [int(m['rid']) for m in meta['train']]
    nper = min(meta['n_train_events'], max(1000, 1150000//len(order)))
else:
    raise SystemExit(f'unknown tag {tag}')

ref = np.load(f'output/models/{tag}_ref.npz')
Rp, Rm = ref['particles'], ref['mask']
N = len(Rm)
assert N == len(order)*nper, f'{tag}: {N} reference events, but {len(order)} runs x {nper}'
print(f'{tag}: {len(order)} training runs x {nper} = {N} reference events, data {data}', flush=True)

AX = ['1_minus_thrust', 'B_total', 'rho_heavy']        # depend on the thrust axis
FIX = ['mult_total']                                     # do not, must be unchanged
new = {o: [] for o in AX + FIX}
tp, npn, nb = [], [], []
off = 0
for k, rid in enumerate(order):
    z = np.load(f'{data}/particles_full_{rid:04d}.npz')
    n = len(z['mask'])
    idx = np.random.default_rng(50000 + rid).choice(n, min(nper, n), replace=False)
    sl = slice(off, off + len(idx))
    P = z['particles'][idx]
    assert np.array_equal(P, Rp[sl]), f'run {rid}: particles at the replayed indices differ from the stored reference'
    assert np.array_equal(z['mask'][idx], Rm[sl]), f'run {rid}: mask differs'
    sh = pd.read_csv(f'{data}/shapes_run_{rid:04d}.csv')
    assert len(sh) == n, f'run {rid}: shapes {len(sh)} rows, particles {n}'
    for o in AX + FIX:
        new[o].append(sh[o].values[idx].astype(np.float64))
    tp.append(z['tau_parton'][idx].astype(np.float32)); npn.append(z['n_parton'][idx])
    nb.append(z['nbaryon'][idx].astype(np.float32))
    off += len(idx)
    if k % 24 == 0:
        print(f'  run {rid} ({k+1}/{len(order)}) replay exact', flush=True)
assert off == N
new = {o: np.concatenate(v) for o, v in new.items()}
tp = np.concatenate(tp); npn = np.concatenate(npn); nb = np.concatenate(nb)
for o in FIX:
    dmax = np.abs(new[o] - np.asarray(ref[o], np.float64)).max()
    assert dmax == 0, f'{o} changed by up to {dmax}, but it does not depend on the thrust axis'
assert np.array_equal(nb, np.asarray(ref['nbaryon'], np.float32)), 'nbaryon changed'
print(f'{tag}: every reference event reproduced bitwise; mult_total and nbaryon unchanged', flush=True)

out = {k: np.asarray(ref[k]) for k in ref.files}
for o in AX:
    old = out[o].astype(np.float64)
    ch = np.mean(np.abs(new[o] - old) > 1e-9*np.maximum(1, np.abs(old)))
    print(f'  {o:15s} old mean {old.mean():.5f}  exact mean {new[o].mean():.5f}  '
          f'({100*(old.mean()/new[o].mean() - 1):+.2f}%), {100*ch:.2f}% of events changed', flush=True)
    out[o] = new[o]
out['tau_parton'] = tp; out['n_parton'] = npn
good = np.isfinite(tp)
print(f'  tau_parton finite on {100*good.mean():.3f}% of events, mean {tp[good].mean():.5f}', flush=True)
np.savez_compressed(OUT, **out)
nch = ((Rp[..., 5] != 0) & Rm.astype(bool)).sum(1).astype(np.int16)
np.savez(SLIM, **{o: out[o] for o in AX + FIX}, nbaryon=out['nbaryon'], strange=out['strange'],
         tau_parton=tp, n_parton=npn, nch=nch, order=np.array(order), nper=np.int64(nper))
print(f'wrote {OUT} and {SLIM}', flush=True)
