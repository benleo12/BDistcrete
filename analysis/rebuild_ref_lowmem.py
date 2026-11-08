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

The rerun (D4, score_obs.py). With BEAM_PHOTONS=remove the shapes come from photon-free (_nb) shapes
files, and the multiplicity of all kept particles and the kaon fraction leave out the beam photons
(from the replayed particles, which are checked against the stored reference as before); the
stored multiplicity must still equal the shapes file's before the beam photons are subtracted. The
bundles carry n_beam and beam_photons = 'remove'. REF_OUT and REF_SLIM must then be given (REF_OUT=none
writes no full bundle, which would copy the particles), and REF_IN names the bundle with the stored
reference (default output/models/<tag>_ref.npz). REF_STAGE=<A, B, C, E, F> replays a reference drawn
by r2_ladder.py for that stage (its STAGES entry: STAGE<S>_DATA, STAGEE_CSV, STAGEF_CSV,
LADDER_ABC_DESIGN, 12000 events of every training run).
"""
import sys, os, json
import numpy as np, pandas as pd
import score_obs as SO

tag = sys.argv[1]
if SO.REMOVE:
    assert os.environ.get('REF_OUT') and os.environ.get('REF_SLIM'), 'BEAM_PHOTONS=remove: give REF_OUT (or none) and REF_SLIM'
OUT = os.environ.get('REF_OUT', f'output/models/{tag}_ref_v2.npz')
SLIM = os.environ.get('REF_SLIM', f'output/models/{tag}_ref_v2_slim.npz')
REF_IN = SO.need(os.environ.get('REF_IN', f'output/models/{tag}_ref.npz'))
for _p in (OUT, SLIM):
    assert _p == 'none' or os.path.realpath(_p) != os.path.realpath(REF_IN), f'{_p} would overwrite the stored reference'
if os.environ.get('REF_STAGE'):
    # a reference drawn by r2_ladder.py itself (LADDER_LOWMEM): the stage's training runs in their
    # registered order, NREF_PER events of each
    import r2_ladder as R
    S = os.environ['REF_STAGE']; cfgS = R.STAGES[S]
    assert os.environ.get(f'STAGE{S}_DATA'), f'set STAGE{S}_DATA: the stage has a default data folder'
    data, nper, order = cfgS['data'], R.NREF_PER, [int(r) for r in cfgS['train']]
elif tag == 'E':
    data, nper = 'data_stageE', 12000
    d = pd.read_csv(f'{data}/stageE_design.csv'); order = [int(r) for r in d[d.role == 'train'].run_id]
elif tag == 'Fauglong':
    data, nper = 'data_stageF', 12000
    d = pd.read_csv(f'{data}/stageF_design_aug.csv'); order = [int(r) for r in d[d.role == 'train'].run_id]
else:
    # any mixture-format data set (MIX17aug, MIXGEO on the pure runs): its reference draws
    # follow stage_mixture.py, DM_REF_TOTAL events in all, spread equally over the training runs
    data = os.environ['DM_DATA']
    meta = json.load(open(SO.need(f'{data}/meta.json')))
    order = [int(m['rid']) for m in meta['train']]
    nper = min(meta['n_train_events'], max(1000, int(os.environ.get('DM_REF_TOTAL', '1150000'))//len(order)))

ref = np.load(REF_IN)
Rp, Rm = ref['particles'], ref['mask']
N = len(Rm)
assert N == len(order)*nper, f'{tag}: {N} reference events, but {len(order)} runs x {nper}'
print(f'{tag}: {len(order)} training runs x {nper} = {N} reference events, data {data}, beam photons {SO.MODE}', flush=True)

AX = ['1_minus_thrust', 'B_total', 'rho_heavy']        # depend on the thrust axis
FIX = ['mult_total']                                     # do not, must be unchanged
new = {o: [] for o in AX + FIX}
tp, npn, nb, nbeam, strg = [], [], [], [], []
has_parton = has_nb = None
off = 0
for k, rid in enumerate(order):
    z = np.load(SO.need(f'{data}/particles_full_{rid:04d}.npz'))
    n = len(z['mask'])
    idx = np.random.default_rng(50000 + rid).choice(n, min(nper, n), replace=False)
    sl = slice(off, off + len(idx))
    P = z['particles'][idx]; M = z['mask'][idx]
    assert np.array_equal(P, Rp[sl]), f'run {rid}: particles at the replayed indices differ from the stored reference'
    assert np.array_equal(M, Rm[sl]), f'run {rid}: mask differs'
    sh = SO.read_shapes(f'{data}/shapes_run_{rid:04d}.csv')
    assert len(sh) == n, f'run {rid}: shapes {len(sh)} rows, particles {n}'
    for o in AX + FIX:
        new[o].append(sh[o].values[idx].astype(np.float64))
    if has_parton is None:
        has_parton, has_nb = 'tau_parton' in z.files, 'nbaryon' in z.files
    if has_parton:
        tp.append(z['tau_parton'][idx].astype(np.float32)); npn.append(z['n_parton'][idx])
    if has_nb:
        nb.append(z['nbaryon'][idx].astype(np.float32))
    if SO.REMOVE:
        b = SO.n_beam(P, M); nbeam.append(b.astype(np.int16)); strg.append(SO.strange_frac(P, M, nb=b))
    off += len(idx)
    if k % 24 == 0:
        print(f'  run {rid} ({k+1}/{len(order)}) replay exact', flush=True)
assert off == N
new = {o: np.concatenate(v) for o, v in new.items()}
if SO.REMOVE:
    # the stored multiplicity counts the beam photons; the photon-free shapes files may keep it so or
    # already leave them out (the observables task rewrote mult_total that way): either is accepted, event
    # by event, and nothing else
    nbeam = np.concatenate(nbeam)
    _old = np.asarray(ref['mult_total'], np.float64)
    _nb = SO.mult_total(_old, Rp, Rm, where=f'{tag} reference', nb=nbeam)
    _ok = (new['mult_total'] == _old) | (new['mult_total'] == _nb)
    assert _ok.all(), f'mult_total of the shapes files matches neither the stored value nor it less the beam photons in {int((~_ok).sum())} events'
    print(f'  mult_total of the shapes files: {int((new["mult_total"] == _nb).sum())} events without the beam photons, '
          f'{int(((new["mult_total"] == _old) & (_old != _nb)).sum())} with them', flush=True)
else:
    for o in FIX:
        dmax = np.abs(new[o] - np.asarray(ref[o], np.float64)).max()
        assert dmax == 0, f'{o} changed by up to {dmax}, but it does not depend on the thrust axis'
if has_nb and 'nbaryon' in ref.files:
    nb = np.concatenate(nb)
    assert np.array_equal(nb, np.asarray(ref['nbaryon'], np.float32)), 'nbaryon changed'
print(f'{tag}: every reference event reproduced bitwise; ' + ('mult_total checked against the beam photons' if SO.REMOVE else 'mult_total unchanged')
      + (', nbaryon unchanged' if has_nb else ''), flush=True)

out = {k: np.asarray(ref[k]) for k in ref.files}
for o in AX:
    old = out[o].astype(np.float64)
    ch = np.mean(np.abs(new[o] - old) > 1e-9*np.maximum(1, np.abs(old)))
    print(f'  {o:15s} old mean {old.mean():.5f}  exact mean {new[o].mean():.5f}  '
          f'({100*(old.mean()/new[o].mean() - 1):+.2f}%), {100*ch:.2f}% of events changed', flush=True)
    out[o] = new[o]
if has_parton:
    tp = np.concatenate(tp); npn = np.concatenate(npn)
    out['tau_parton'] = tp; out['n_parton'] = npn
    good = np.isfinite(tp)
    print(f'  tau_parton finite on {100*good.mean():.3f}% of events, mean {tp[good].mean():.5f}', flush=True)
if SO.REMOVE:
    out['mult_total'] = _nb
    if 'strange' in ref.files:
        out['strange'] = np.concatenate(strg).astype(np.asarray(ref['strange']).dtype)
    out['n_beam'] = nbeam; out['beam_photons'] = np.array('remove')
    print(f'  beam photons: {100*np.mean(nbeam > 0):.2f}% of the reference events, {nbeam.mean():.3f} per event; '
          f'mult_total mean {np.asarray(ref["mult_total"], np.float64).mean():.4f} -> {out["mult_total"].mean():.4f}', flush=True)
if OUT != 'none':
    np.savez_compressed(OUT, **out)
nch = ((Rp[..., 5] != 0) & Rm.astype(bool)).sum(1).astype(np.int16)
slim = {o: out[o] for o in AX + FIX}
for k in ('nbaryon', 'strange', 'tau_parton', 'n_parton', 'n_beam', 'beam_photons'):
    if k in out:
        slim[k] = out[k]
np.savez(SLIM, **slim, nch=nch, order=np.array(order), nper=np.int64(nper))
print(f'wrote {OUT if OUT != "none" else "(no full bundle)"} and {SLIM}', flush=True)
