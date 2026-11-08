#!/usr/bin/env python3
"""Read-back check of the refreshed files (step 3), separate from refresh_bundles.py: for every file
the plan marks refreshed,
  - the members are those of the original; every member that is not an observable has the same
    CRC-32, size and compressed size as in the original (the raw bytes were copied);
  - the observable arrays equal the _nb shapes files at the replayed rows as pandas reads them (the
    published scripts read them so), bit for bit, on NRUN runs drawn at random per file (all runs
    with NRUN=0); the largest difference from a correctly rounded float() parse of the same text is
    recorded; for the kaon fraction, kaons / (kept - n_beam) of the source particles;
  - where the source runs are gone (V4's 2048 v4 runs), nothing can be compared and this is said.
    python verify_refresh.py PLAN_TSV NRUN SEED NPROC"""
import os, sys, json, zipfile, random
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beam_photons as BP                                        # noqa: E402

R = '/mnt/beegfs/bassi/reweighting'
MC, GEN, OUTM, RUN = f'{R}/maxent_code', f'{R}/gen', f'{R}/maxent_code/output', f'{R}/lmx_runs'
OBS = ['1_minus_thrust', 'B_total', 'rho_heavy', 'mult_total', 'strange']
CSVOBS = OBS[:4]
PLAN, NRUN, SEED, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])


def nbdir(d):
    if d.startswith(GEN + '/'):
        rel = d[len(GEN) + 1:].split('/'); rel[0] += '_nb'
        return GEN + '/' + '/'.join(rel)
    return d + '_nb'


def rows_exact(path, idx):
    t = open(path).read().split('\n')
    h = t[0].split(','); body = t[1:-1]
    ci = [h.index(o) for o in CSVOBS]
    return np.array([[float(body[i].split(',')[c]) for c in ci] for i in idx]).reshape(-1, 4)


def nev(path):
    with zipfile.ZipFile(path) as zf, zf.open('mask.npy') as fh:
        v = np.lib.format.read_magic(fh)
        return int((np.lib.format.read_array_header_1_0(fh) if v == (1, 0) else np.lib.format.read_array_header_2_0(fh))[0][0])


def sources(tag):
    """(data folders of a training run, nper, order) for the bundle of `tag`"""
    if tag.startswith('MIXGEO'):
        sl = np.load(f'{OUTM}/models/{tag}_ref_v2_slim.npz')
        order, nper = [int(r) for r in sl['order']], int(sl['nper'])
        if tag == 'MIXGEOV3E20S9':
            return (lambda r: [f'{MC}/data_stagePURE17_v3ev20']), nper, order
        if tag in ('MIXGEOV3E160S9', 'MIXGEOV3E320S9'):
            k = 1 if '160' in tag else 3
            def f(r, k=k):
                b = f'{GEN}/data_release' if os.path.exists(f'{GEN}/data_release/particles_full_{r:04d}.npz') else f'{GEN}/data_v3'
                return [b] + [f'{GEN}/data_trainext/k{j}' for j in range(1, k + 1)]
            return f, nper, order
        if tag.startswith('MIXGEOV4'):
            def g(r):
                for d in (f'{GEN}/data_release', f'{GEN}/data_v3'):
                    if os.path.exists(f'{d}/particles_full_{r:04d}.npz'):
                        return [d]
                return None
            return g, nper, order
        return (lambda r: [f'{MC}/data_stagePURE17_v3']), nper, order
    S = tag[0]
    data, design = {'C': (f'{GEN}/data_abc', f'{GEN}/designs/stageC_design_aug.csv'),
                    'E': (f'{MC}/data_stagePURE17_v3', f'{MC}/stageE_design_v3all.csv'),
                    'F': (f'{MC}/data_stagePURE17_v3', f'{MC}/stageF_design_v3all.csv')}[S]
    d = pd.read_csv(design)
    return (lambda r: [data]), 12000, [int(r) for r in d[d.role == 'train'].run_id]


def check_bundle(item):
    rel, src = item
    dst = f'{RUN}/maxent_code/output/{rel}'
    ia = {i.filename: i for i in zipfile.ZipFile(src).infolist()}
    ib = {i.filename: i for i in zipfile.ZipFile(dst).infolist()}
    out = dict(path=rel, members_same=sorted(ia) == sorted(ib), raw_kept_differ=[], replaced=[])
    for n in ia:
        k = n[:-4]
        if k in OBS:
            out['replaced'].append(k)
        elif not (ia[n].CRC == ib[n].CRC and ia[n].file_size == ib[n].file_size and ia[n].compress_size == ib[n].compress_size):
            out['raw_kept_differ'].append(k)
    b = np.load(dst)
    tag = os.path.basename(rel).split('_ref')[0]
    srcf, nper, order = sources(tag)
    rng = random.Random(SEED + len(rel))
    ks = list(range(len(order))) if NRUN == 0 else sorted(rng.sample(range(len(order)), min(NRUN, len(order))))
    cmp = dict(runs=0, events=0, mismatch=0, strange_mismatch=0, runs_without_source=0)
    offs = np.cumsum([0] + [nper] * len(order))      # every run gives nper events (checked by the refresh)
    for k in ks:
        rid = order[k]
        srcs = srcf(rid)
        if srcs is None:
            cmp['runs_without_source'] += 1; continue
        n_run = sum(nev(f'{d}/particles_full_{rid:04d}.npz') for d in srcs)
        idx = np.random.default_rng(50000 + rid).choice(n_run, min(nper, n_run), replace=False)
        # rows of the concatenated source files
        exact = []
        bases = np.cumsum([0] + [nev(f'{d}/particles_full_{rid:04d}.npz') for d in srcs])
        for j, d in enumerate(srcs):
            sel = (idx >= bases[j]) & (idx < bases[j + 1])
            exact.append((np.where(sel)[0], rows_exact(f'{nbdir(d)}/shapes_run_{rid:04d}.csv', idx[sel] - bases[j])))
        V = np.empty((len(idx), 4))
        for pos, val in exact:
            V[pos] = val
        # the published scripts read the shapes files with pandas; the bundle must hold exactly that
        Vp = pd.concat([pd.read_csv(f'{nbdir(d)}/shapes_run_{rid:04d}.csv', usecols=CSVOBS) for d in srcs],
                       ignore_index=True)[CSVOBS].values[idx]
        sl = slice(int(offs[k]), int(offs[k]) + len(idx))
        got = np.column_stack([np.asarray(b[o][sl], np.float64) for o in CSVOBS])
        cmp['mismatch'] += int((got != Vp).sum())
        cmp['max_abs_vs_float_parse'] = max(cmp.get('max_abs_vs_float_parse', 0.0), float(np.abs(got - V).max()))
        if 'strange' in b.files:
            zs = [np.load(f'{d}/particles_full_{rid:04d}.npz') for d in srcs]
            P = np.concatenate([z['particles'] for z in zs])[idx]; M = np.concatenate([z['mask'] for z in zs])[idx].astype(bool)
            nb = BP.n_beam(P, M)
            lm = P[..., 3]; ink = (((lm > -0.36) & (lm < -0.20)) & M).sum(1)
            cmp['strange_mismatch'] += int(((ink / np.maximum(M.sum(1) - nb, 1)).astype(np.float32) != b['strange'][sl]).sum())
        cmp['runs'] += 1; cmp['events'] += len(idx)
    out['vs_nb_csv'] = cmp
    out['ok'] = out['members_same'] and not out['raw_kept_differ'] and cmp['mismatch'] == 0 and cmp['strange_mismatch'] == 0
    return out


def check_wifi(item):
    rel, src = item
    dst = f'{RUN}/maxent_code/output/{rel}'
    ia = {i.filename: i for i in zipfile.ZipFile(src).infolist()}
    ib = {i.filename: i for i in zipfile.ZipFile(dst).infolist()}
    out = dict(path=rel, members_same=sorted(ia) == sorted(ib), raw_kept_differ=[], replaced=0)
    rep = lambda k: k.startswith('held_') and any(k.startswith(f'held_{o}_') for o in OBS)
    for n in ia:
        if rep(n[:-4]):
            out['replaced'] += 1
        elif not (ia[n].CRC == ib[n].CRC and ia[n].file_size == ib[n].file_size):
            out['raw_kept_differ'].append(n)
    b = np.load(dst)
    data = f'{MC}/data_stagePURE17_v3'
    mm = 0; ms = 0
    for rid in [int(r) for r in b['held_rids']]:
        V = pd.read_csv(f'{nbdir(data)}/shapes_run_{rid:04d}.csv', usecols=CSVOBS)[CSVOBS].values
        for j, o in enumerate(CSVOBS):
            mm += int((V[:, j].astype(np.float32) != b[f'held_{o}_{rid}']).sum())
        z = np.load(f'{data}/particles_full_{rid:04d}.npz'); P, M = z['particles'], z['mask'].astype(bool)
        nb = BP.n_beam(P, M); lm = P[..., 3]; ink = (((lm > -0.36) & (lm < -0.20)) & M).sum(1)
        ms += int(((ink / np.maximum(M.sum(1) - nb, 1)).astype(np.float32) != b[f'held_strange_{rid}']).sum())
    out.update(vs_nb_csv_mismatch=mm, strange_mismatch=ms)
    out['ok'] = out['members_same'] and not out['raw_kept_differ'] and mm == 0 and ms == 0
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    rows = [l.rstrip('\n').split('\t') for l in open(PLAN)][1:]
    items = [(r[0], r[2]) for r in rows if r[1] == 'refreshed']
    bundles = [it for it in items if it[0].startswith('models/')]
    wifis = [it for it in items if it[0].startswith('wifi_')]
    with mp.Pool(NPROC) as pool:
        res = pool.map(check_bundle, bundles, chunksize=1) + pool.map(check_wifi, wifis, chunksize=1)
    # the release reference: the refreshed MIXGEOV3S9 arrays in the same order, other members raw-identical
    src, dst = f'{R}/release_ref/reference_MIX17.npz', f'{RUN}/release_ref/reference_MIX17.npz'
    ia = {i.filename: i for i in zipfile.ZipFile(src).infolist()}; ib = {i.filename: i for i in zipfile.ZipFile(dst).infolist()}
    a, s = np.load(dst), np.load(f'{RUN}/maxent_code/output/models/MIXGEOV3S9_ref_v2_slim.npz')
    rr = dict(path='release_ref/reference_MIX17.npz', members_same=sorted(ia) == sorted(ib),
              raw_kept_differ=[n for n in ia if n[:-4] not in OBS and not (ia[n].CRC == ib[n].CRC and ia[n].file_size == ib[n].file_size)],
              equal_refreshed_slim={o: bool(np.array_equal(np.asarray(a[o], np.float64), np.asarray(s[o], np.float64))) for o in OBS})
    rr['ok'] = rr['members_same'] and not rr['raw_kept_differ'] and all(rr['equal_refreshed_slim'].values())
    res.append(rr)
    os.makedirs(f'{R}/lmx_phaseB/observables/results', exist_ok=True)
    json.dump(res, open(f'{R}/lmx_phaseB/observables/results/verify_refresh_{SEED}.json', 'w'), indent=1)
    for r in res:
        print(('OK  ' if r['ok'] else 'BAD ') + json.dumps(r)[:400])
    print('VERIFY REFRESH:', sum(r['ok'] for r in res), 'of', len(res), 'ok')
    sys.exit(0 if all(r['ok'] for r in res) else 1)
