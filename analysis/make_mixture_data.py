#!/usr/bin/env python3
"""Mixture-stage data for any pair of single-generator designs.

The family is
    q(Phi; theta_S, theta_H, f) = (1 - f) q_S(Phi; theta_S) + f q_H(Phi; theta_H),
so a mixture run is just a resampled blend of one run from each side. Nothing new is
generated. Events are drawn without replacement from each source run, so no event appears
twice anywhere in the data set.

This replaces the hard-coded make_stageDM_data.py, which knew the three-parameter Sherpa grid
and the three-parameter Herwig box by heart. Here both sides come from their design CSVs, so
the eight-parameter boxes give a seventeen-parameter mixture with nothing to edit but the
environment. The structural choices that matter are kept:

  * a fraction of the runs sit at f = 0 and f = 1 exactly, and on those runs the SILENT side's
    label is drawn uniformly in its box. The data do not depend on it, and scattering the
    labels is what lets the network learn that the derivative with respect to the silent side
    vanishes there, rather than being told so.
  * the interior runs cycle f over a fixed ladder, and the two sides are paired through
    independent permutations, so theta_S and theta_H are not correlated with each other or
    with f by construction.
  * every source run holds a finite number of events, so the design is checked for
    feasibility BEFORE anything is written. Exhausting a pool halfway through a long build is
    an expensive way to find out.

Usage:
    S_DESIGN=data_stageE/stageE_design.csv H_DESIGN=data_stageF/stageF_design.csv \
    S_SRC=data_stageE H_SRC=data_stageF DM_DST=data_stageDM17 DM_NTRAIN=192 \
    python make_mixture_data.py
"""
import itertools, json, os
import numpy as np
import pandas as pd

SEED = int(os.environ.get('DM_SEED', '20260912'))
rng = np.random.default_rng(SEED)
DST = os.environ.get('DM_DST', 'data_stageDM17')
S_SRC = os.environ.get('S_SRC', 'data_stageE')
H_SRC = os.environ.get('H_SRC', 'data_stageF')
S_DESIGN = os.environ.get('S_DESIGN', f'{S_SRC}/stageE_design.csv')
H_DESIGN = os.environ.get('H_DESIGN', f'{H_SRC}/stageF_design.csv')
NTR = int(os.environ.get('DM_NTR', '25000'))          # events per training run
NHE = int(os.environ.get('DM_NHE', '80000'))          # events per held run
N_TRAIN = int(os.environ.get('DM_NTRAIN', '192'))
# DM_DRYRUN=1 builds and checks the design and then stops, so the feasibility of a given run
# count can be settled from the CSVs alone, on a machine that holds none of the events.
DRYRUN = os.environ.get('DM_DRYRUN', '') == '1'
N_HELD = int(os.environ.get('DM_NHELD', '10'))
RID0 = int(os.environ.get('DM_RID0', '9700'))
RID0_HELD = int(os.environ.get('DM_RID0_HELD', '9930'))
PURE_FRAC = float(os.environ.get('DM_PURE_FRAC', '0.125'))   # per endpoint
F_LADDER = [float(x) for x in os.environ.get('DM_F_LADDER', '0.125,0.25,0.375,0.5,0.625,0.75,0.875').split(',')]


def load_design(csv, src):
    """(train rids, train thetas, held rids, held thetas, box) from a design CSV.

    The axis columns are every column that is not the run id or the role, in file order, which
    is the same order r2_ladder builds its theta vectors in. Taking them positionally from one
    place is deliberate: a reordering between the data and the model would be invisible."""
    d = pd.read_csv(csv)
    axes = [c for c in d.columns if c not in ('run_id', 'role')]
    tr = d[d.role == 'train']
    hd = d[d.role == 'held']
    assert len(tr) and len(hd), f'{csv}: {len(tr)} train and {len(hd)} held rows'
    box = [(float(d[a].min()), float(d[a].max())) for a in axes]
    if not DRYRUN:
        for rid in list(tr.run_id) + list(hd.run_id):
            f = f'{src}/particles_full_{int(rid):04d}.npz'
            assert os.path.exists(f), f'design names run {rid} but {f} is missing'
    return (list(tr.run_id.astype(int)), tr[axes].values.astype(float),
            list(hd.run_id.astype(int)), hd[axes].values.astype(float), box, axes)


class Pool:
    """Draws without replacement from a set of source runs, one cursor per run."""

    def __init__(s, src, rids, capacity):
        s.src, s.rids, s.cap = src, rids, capacity
        s.perm, s.off = {}, {}

    def take(s, idx, n):
        rid = s.rids[idx]
        if rid not in s.perm:
            sh = pd.read_csv(f'{s.src}/shapes_run_{rid:04d}.csv')
            s.perm[rid] = rng.permutation(len(sh))
            s.off[rid] = 0
        k = s.perm[rid][s.off[rid]:s.off[rid] + n]
        assert len(k) == n, (f'pool exhausted at run {rid}: need {n}, '
                             f'{len(s.perm[rid]) - s.off[rid]} left')
        s.off[rid] += n
        k = np.sort(k)                      # sequential reads off the compressed npz
        d = np.load(f'{s.src}/particles_full_{rid:04d}.npz')
        sh = pd.read_csv(f'{s.src}/shapes_run_{rid:04d}.csv')
        return {kk: d[kk][k] for kk in d.files}, sh.iloc[k]


def rand_in(box):
    return [float(rng.uniform(lo, hi)) for lo, hi in box]


def _refuse_to_clobber():
    """Refuse to write over a finished build.

    The run ids are fixed constants and DM_DST has a fixed default, so a second build silently
    replaced the data set an already-trained head was fitted on, meta.json included. Once
    meta.json is gone nothing downstream can notice, because mixture_cfg.py derives the entire
    design from it. Set DM_FORCE=1 to overwrite deliberately."""
    mp = f'{DST}/meta.json'
    if os.path.exists(mp) and os.environ.get('DM_FORCE', '') != '1':
        raise SystemExit(
            f'{mp} already exists, so {DST} holds a finished build and some head was trained on '
            f'it. Writing here would replace the design that head reports. Choose another DM_DST, '
            f'or set DM_FORCE=1 if replacing it is what you mean to do.')


def build_design(nS, nH, s_box, h_box):
    n_pure = max(1, int(round(PURE_FRAC*N_TRAIN)))
    n_mix = N_TRAIN - 2*n_pure
    assert n_mix > 0, f'{N_TRAIN} runs cannot hold {n_pure} of each pure endpoint'
    runs = []
    # The pure endpoints spread evenly over their own design, so a small number of them still
    # covers the whole box instead of clustering.
    #
    # This used to use a stride of 7 and a comment claiming it was coprime to the design size.
    # That held for 64 and 96 sources but not for 112, where gcd(7, 112) = 7 and the 24 pure
    # Herwig runs collapsed onto 16 distinct sources. Three pure runs on one source draw 3 x NTR
    # events from it, which is what overdrew an 80,000-event run. Even spacing needs no
    # coprimality argument and cannot collide while n_pure <= n.
    assert n_pure <= min(nS, nH), f'{n_pure} pure runs cannot be spread over {nS}/{nH} sources'
    for r in range(n_pure):
        runs.append(dict(iS=int(round(r*nS/n_pure)) % nS, iH=None, f=0.0, thH=rand_in(h_box)))
        runs.append(dict(iS=None, iH=int(round(r*nH/n_pure)) % nH, f=1.0, thS=rand_in(s_box)))
    fs = [F_LADDER[r % len(F_LADDER)] for r in range(n_mix)]
    iS = [int(i) % nS for i in rng.permutation(n_mix)]
    iH = [int(i) % nH for i in rng.permutation(n_mix)]
    for r in range(n_mix):
        runs.append(dict(iS=iS[r], iH=iH[r], f=fs[r]))
    rng.shuffle(runs)
    useS, useH = np.zeros(nS), np.zeros(nH)
    for m in runs:
        if m['iS'] is not None:
            useS[m['iS']] += (1 - m['f'])*NTR
        if m['iH'] is not None:
            useH[m['iH']] += m['f']*NTR
    print(f'design: {N_TRAIN} runs ({n_pure} pure Sherpa, {n_pure} pure Herwig, {n_mix} mixed)')
    print(f'  peak source usage: Sherpa {useS.max():.0f}/{NHE}, Herwig {useH.max():.0f}/{NHE}')
    assert useS.max() <= NHE and useH.max() <= NHE, (
        f'the design overdraws a source run ({useS.max():.0f}, {useH.max():.0f} against {NHE}). '
        f'Lower DM_NTRAIN or DM_NTR.')
    return runs


def mix(m, n, sp, hp):
    nb = int(round(m['f']*n))
    na = n - nb
    parts = []
    if na:
        parts.append(sp.take(m['iS'], na))
    if nb:
        parts.append(hp.take(m['iH'], nb))
    arrs = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    sh = pd.concat([p[1] for p in parts])
    o = rng.permutation(n)
    return {k: v[o] for k, v in arrs.items()}, sh.iloc[o], na, nb


def main():
    _refuse_to_clobber()
    os.makedirs(DST, exist_ok=True)
    Str, Sth, Shr, Shth, s_box, s_axes = load_design(S_DESIGN, S_SRC)
    Htr, Hth, Hhr, Hhth, h_box, h_axes = load_design(H_DESIGN, H_SRC)
    d = len(s_axes) + len(h_axes) + 1
    K2 = 1 + d + d*(d + 1)//2
    print(f'Sherpa {len(Str)} train / {len(Shr)} held on {len(s_axes)} axes: {s_axes}')
    print(f'Herwig {len(Htr)} train / {len(Hhr)} held on {len(h_axes)} axes: {h_axes}')
    print(f'mixture dimension d={d}, second-order rank K2={K2}, '
          f'{N_TRAIN} training runs = {N_TRAIN/K2:.2f} per mode')
    design = build_design(len(Str), len(Htr), s_box, h_box)
    n_int = max(1, N_HELD - 2)
    held = [dict(iS=j % len(Shr), iH=(j + 2) % len(Hhr),
                 f=F_LADDER[j % len(F_LADDER)], n=NHE) for j in range(n_int)]
    # Running usage, so the two pure endpoints land on whichever held run has the most left
    # over rather than on a fixed index. Indexing them by the interior count reuses the run
    # that already supplied the most: with eight held runs and a fraction ladder starting at
    # 1/8, run 0 is already drawing 70000 of its 80000 events, and adding a 40000 event
    # endpoint would overdraw it and stop the build partway through.
    uS, uH = np.zeros(len(Shr)), np.zeros(len(Hhr))
    for m in held:
        uS[m['iS']] += (1 - m['f'])*m['n']
        uH[m['iH']] += m['f']*m['n']
    n_end = NHE//2
    iS_end = int(np.argmin(uS))
    iH_end = int(np.argmin(uH))
    held += [dict(iS=iS_end, iH=None, f=0.0, n=n_end, thH=rand_in(h_box)),
             dict(iS=None, iH=iH_end, f=1.0, n=n_end, thS=rand_in(s_box))]
    uS[iS_end] += n_end
    uH[iH_end] += n_end
    assert uS.max() <= NHE and uH.max() <= NHE, (
        f'the held design overdraws a held run ({uS.max():.0f}, {uH.max():.0f} against {NHE}). '
        f'Lower DM_NHELD or DM_NHE.')
    print(f'held: {len(held)} runs, peak usage Sherpa {uS.max():.0f}, Herwig {uH.max():.0f} '
          f'of {NHE}; pure endpoints on Sherpa held {iS_end} and Herwig held {iH_end}')
    if DRYRUN:
        # deliberately AFTER both feasibility checks: the training design and the held design
        # are separate constraints and the held one is the easier to get wrong
        print('DRY RUN: both designs are feasible, nothing written')
        return

    meta = dict(train=[], held=[], seed=SEED, s_axes=s_axes, h_axes=h_axes,
                s_box=s_box, h_box=h_box, ntheta=d, n_train_events=NTR, n_held_events=NHE,
                s_src=S_SRC, h_src=H_SRC)
    sp = Pool(S_SRC, Str, NHE)
    hp = Pool(H_SRC, Htr, NHE)
    for r, m in enumerate(design):
        rid = RID0 + r
        thS = list(Sth[m['iS']]) if m['iS'] is not None else list(m['thS'])
        thH = list(Hth[m['iH']]) if m['iH'] is not None else list(m['thH'])
        arrs, sh, na, nb = mix(m, NTR, sp, hp)
        np.savez_compressed(f'{DST}/particles_full_{rid:04d}.npz', **arrs)
        sh.to_csv(f'{DST}/shapes_run_{rid:04d}.csv', index=False)
        meta['train'].append(dict(rid=rid, theta=thS + thH + [m['f']],
                                  n_sherpa=na, n_herwig=nb))
        if r % 16 == 0 or r == len(design) - 1:
            print(f'  train rid {rid}: f={m["f"]:.3f} {na}+{nb} '
                  f'<1-T>={sh["1_minus_thrust"].mean():.5f}', flush=True)

    # Held runs come from the held pools, which the single-generator stages kept out of their
    # own training. Interior fractions, plus one run at each endpoint, because the endpoints
    # are where the mixture form makes its sharpest claim. The design itself was built and
    # checked for feasibility above, before anything was written.
    sph = Pool(S_SRC, Shr, NHE)
    hph = Pool(H_SRC, Hhr, NHE)
    for j, m in enumerate(held):
        rid = RID0_HELD + j
        arrs, sh, na, nb = mix(m, m['n'], sph, hph)
        thS = list(Shth[m['iS']]) if m['iS'] is not None else list(m['thS'])
        thH = list(Hhth[m['iH']]) if m['iH'] is not None else list(m['thH'])
        np.savez_compressed(f'{DST}/particles_full_{rid:04d}.npz', **arrs)
        sh.to_csv(f'{DST}/shapes_run_{rid:04d}.csv', index=False)
        meta['held'].append(dict(rid=rid, theta=thS + thH + [m['f']],
                                 n_sherpa=na, n_herwig=nb))
        print(f'  held rid {rid}: f={m["f"]:.3f} {na}+{nb} '
              f'<1-T>={sh["1_minus_thrust"].mean():.5f}', flush=True)
    json.dump(meta, open(f'{DST}/meta.json', 'w'), indent=1)
    print(f'MIXTURE DATA DONE: {len(meta["train"])} train, {len(meta["held"])} held in {DST}')


if __name__ == '__main__':
    main()
