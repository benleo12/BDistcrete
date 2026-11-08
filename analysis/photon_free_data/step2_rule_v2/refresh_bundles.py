#!/usr/bin/env python3
"""Step 3 of B0-observables: refreshed copies of the files that cache per-event observables of a
reference sample or of held-out runs, with the photon-free values (D4).

Replaced arrays: 1_minus_thrust, B_total, rho_heavy, mult_total and the kaon fraction strange
(its denominator, all kept particles, loses the beam photons); in a wifi basis their held_* forms.
Everything else is kept bit for bit: the unchanged zip members are copied raw (zip -d on a copy),
the new ones are added with the compression of the member they replace.

Two methods:
  csv   the source runs exist: replay the reference draw (default_rng(50000 + rid).choice(n, nper)
        for every training run in training order, as r2_ladder.py under LADDER_LOWMEM and
        rebuild_ref_lowmem.py) and read the _nb shapes files at those rows (pandas, as the published
        scripts read them). The replay is proven on every event: the source particles at those rows
        equal the bundle's particles, and the original shapes files give the bundle's old values.
  rule  the data sets are gone (V4, E160, E320): the per-event rule of the _nb shapes files on the
        bundle's own particles: stored - (with - without) for the three shapes (nb_core rebuild,
        beam_photons mask), the stored value where |with - stored| > 1e-2 or the result is negative,
        mult_total - n_beam. Where the rows still exist in gen (the 720 release and v3 runs of V4; the
        base runs and data_trainext extensions of every E160 and E320 run), the rule is compared with
        the _nb shapes files at the replayed rows and those values (read with pandas) replace it, so
        that the bundle holds exactly what the shapes files hold wherever both exist.
strange = kaons / (kept - n_beam) everywhere, from the particles of the same events.

    python refresh_bundles.py JOB NPROC        JOB: mix_v3 mix_ev20 mix_e160 mix_e320 mix_v4 stage_C
                                               stage_E stage_F wifi release (or 'all')
Writes into /mnt/beegfs/bassi/reweighting/lmx_runs (maxent_code/output/<same relative path>, and
release_ref/) and a record WORK/results/refresh_<JOB>.json. Reads the originals only.
"""
import os, sys, json, time, shutil, subprocess, zipfile
import numpy as np, pandas as pd

R = '/mnt/beegfs/bassi/reweighting'
MC, GEN, OUTM = f'{R}/maxent_code', f'{R}/gen', f'{R}/maxent_code/output'
RUN = f'{R}/lmx_runs'
DST = f'{RUN}/maxent_code/output'
WORK = f'{R}/lmx_phaseB/observables'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beam_photons as BP                       # noqa: E402
import nb_core as C                             # noqa: E402

SHAPES = ['1_minus_thrust', 'B_total', 'rho_heavy']
CSVOBS = SHAPES + ['mult_total']
MISS_TOL = 1e-2
G = {}                                          # data shared with forked workers


def nbdir(d):
    """the photon-free mirror of a data folder (maxent_code/x -> x_nb, gen/x/k1 -> gen/x_nb/k1)"""
    if d.startswith(GEN + '/'):
        rel = d[len(GEN) + 1:].split('/'); rel[0] += '_nb'
        return GEN + '/' + '/'.join(rel)
    assert d.startswith(MC + '/') and '/' not in d[len(MC) + 1:], d
    return d + '_nb'


def kaons(P, M):
    lm = P[..., 3]
    return (((lm > -0.36) & (lm < -0.20)) & M).sum(1)


def strange_new(P, M, nb):
    return (kaons(P, M) / np.maximum(M.sum(1) - nb, 1)).astype(np.float32)


def strange_old(P, M):
    return (kaons(P, M) / np.maximum(M.sum(1), 1)).astype(np.float32)


# ---------------------------------------------------------------------------------- replays
def replay(order, nper, nrun):
    """[(rid, idx)] of the reference draw"""
    return [(int(r), np.random.default_rng(50000 + int(r)).choice(nrun[int(r)], min(nper, nrun[int(r)]), replace=False))
            for r in order]


def nevents_npz(path):
    with zipfile.ZipFile(path) as zf, zf.open('mask.npy') as fh:
        v = np.lib.format.read_magic(fh)
        return int((np.lib.format.read_array_header_1_0(fh) if v == (1, 0) else np.lib.format.read_array_header_2_0(fh))[0][0])


def csv_rows(task):
    """one run of the csv method: old and new values at the replayed rows, and the checks"""
    k, rid, idx, src_npz, csv_orig, csv_nb, off = task
    old = G['old']; Pb = G.get('P'); Mb = G.get('M')
    z = np.load(src_npz)
    P, M = z['particles'][idx], z['mask'][idx].astype(bool)
    so = pd.read_csv(csv_orig, usecols=CSVOBS)
    sn = pd.read_csv(csv_nb, usecols=CSVOBS)
    assert len(so) == len(sn) == len(z['mask']), (rid, len(so), len(sn))
    sl = slice(off, off + len(idx))
    chk = {}
    if Pb is not None:
        chk['particles_equal'] = bool(np.array_equal(P, Pb[sl]) and np.array_equal(M, Mb[sl]))
    for o in CSVOBS:
        # the original shapes files give the bundle's old values: bit for bit (n_exact), and in any
        # case to 1e-12 (a wrong row would differ by far more)
        d = np.abs(so[o].values[idx].astype(np.float64) - old[o][sl])
        chk[f'old_{o}_equal'] = bool(d.max() <= 1e-12)
        chk[f'old_{o}_inexact'] = int((d > 0).sum())
    chk['nbaryon_equal'] = bool(np.array_equal(z['nbaryon'][idx].astype(np.float32), old['nbaryon'][sl]))
    if 'strange' in old:
        chk['strange_old_equal'] = bool(np.array_equal(strange_old(P, M), old['strange'][sl]))
    nb = BP.n_beam(P, M)
    newv = {o: sn[o].values[idx].astype(np.float64) for o in CSVOBS}
    # the _nb mult_total is the original less the beam photons of beam_photons.py
    chk['mult_minus_nbeam'] = bool(np.array_equal(so['mult_total'].values[idx] - nb, newv['mult_total']))
    newv['strange'] = strange_new(P, M, nb)
    return k, off, newv, chk, int((nb > 0).sum())


def rule_chunk(rng):
    """the per-event rule on rows a:b of the bundle's own particles"""
    a, b = rng
    P, M, old = G['P'], G['M'], G['old']
    p, m = P[a:b], M[a:b].astype(bool)
    F = BP.beam_photon_mask(p, m)
    nb = F.sum(1)
    out = {o: old[o][a:b].astype(np.float64).copy() for o in SHAPES}
    stat = dict(beam_events=0, miss=[0, 0, 0], neg=[0, 0, 0])
    for i in np.where(nb > 0)[0]:
        f = p[i][m[i]]; bb = F[i][m[i]]
        zz, mm, UU = C.parts(f)
        s, _ = C.scale_of(zz, mm, UU)
        ev = C.four(zz, mm, UU, s)
        w = np.array(C.three(ev)); wo = np.array(C.three(ev[~bb]))
        st = np.array([old[o][a + i] for o in SHAPES], np.float64)
        cand = st - (w - wo)
        for j, o in enumerate(SHAPES):
            if not abs(w[j] - st[j]) <= MISS_TOL:
                stat['miss'][j] += 1
            elif not cand[j] >= 0:
                stat['neg'][j] += 1
            else:
                out[o][i] = cand[j]
        stat['beam_events'] += 1
    out['mult_total'] = old['mult_total'][a:b].astype(np.float64) - nb
    out['strange'] = strange_new(p, m, nb)
    return a, out, stat, nb


def pmap(fn, tasks, nproc):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(nproc) as pool:
        return list(pool.imap_unordered(fn, tasks, chunksize=1))


# ---------------------------------------------------------------------------------- writing
def write_replaced(src, dst, new):
    """dst = src with the members of `new` replaced (same dtype and shape, same compression);
    the other members are copied raw by zip -d on a copy"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    assert dst.startswith(RUN + '/'), dst
    info = {i.filename: i for i in zipfile.ZipFile(src).infolist()}
    z = np.load(src)
    for k, v in new.items():
        assert k in z.files, (src, k)
        o = z[k]
        assert o.shape == v.shape and o.dtype == v.dtype, (src, k, o.shape, v.shape, o.dtype, v.dtype)
    tmp = f'{dst}.tmp{os.getpid()}.npz'
    shutil.copyfile(src, tmp)
    subprocess.run(['zip', '-q', '-d', tmp] + [f'{k}.npy' for k in new], check=True)
    with zipfile.ZipFile(tmp, 'a', allowZip64=True) as zf:
        for k, v in new.items():
            zi = zipfile.ZipInfo(f'{k}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = info[f'{k}.npy'].compress_type
            with zf.open(zi, 'w', force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.ascontiguousarray(v), allow_pickle=False)
    os.replace(tmp, dst)


def verify_written(src, dst, new):
    """every member of src is in dst; replaced ones equal `new`, all others are identical arrays
    (and identical compressed bytes: same CRC-32 and size)"""
    a, b = np.load(src), np.load(dst)
    ia = {i.filename: i for i in zipfile.ZipFile(src).infolist()}
    ib = {i.filename: i for i in zipfile.ZipFile(dst).infolist()}
    res = dict(members_same=sorted(ia) == sorted(ib), replaced={}, kept={})
    for k in a.files:
        if k in new:
            x = b[k]
            res['replaced'][k] = bool(x.dtype == new[k].dtype and x.shape == new[k].shape and np.array_equal(x, new[k]))
        else:
            n = f'{k}.npy'
            same_raw = ia[n].CRC == ib[n].CRC and ia[n].file_size == ib[n].file_size and ia[n].compress_size == ib[n].compress_size
            x, y = a[k], b[k]
            res['kept'][k] = bool(same_raw and x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y))
    res['ok'] = res['members_same'] and all(res['replaced'].values()) and all(res['kept'].values())
    return res


def changes(old, new):
    out = {}
    for o in new:
        x, y = np.asarray(old[o], np.float64), np.asarray(new[o], np.float64)
        out[o] = dict(mean_old=float(x.mean()), mean_new=float(y.mean()), rel_change=float(y.mean() / x.mean() - 1),
                      n_changed=int((x != y).sum()), n=int(len(x)), min_new=float(y.min()), n_neg_new=int((y < 0).sum()))
    return out


# ---------------------------------------------------------------------------------- jobs
def load_old(path, keys):
    z = np.load(path)
    return {k: np.asarray(z[k]) for k in keys if k in z.files}


def mix_csv(tags, data, nproc, rec):
    """mixture bundles (slim and _ref) of runs that exist: csv method"""
    t0 = tags[0]
    slim0, ref0 = f'{OUTM}/models/{t0}_ref_v2_slim.npz', f'{OUTM}/models/{t0}_ref.npz'
    keys = CSVOBS + ['strange', 'nbaryon']
    old = load_old(slim0, keys + ['order', 'nper', 'nch'])
    z = np.load(ref0)
    G['P'] = z['particles']; G['M'] = z['mask'].astype(bool)
    oref = load_old(ref0, keys)
    rec['slim_vs_ref_old_equal'] = {k: bool(np.array_equal(np.asarray(old[k], np.float64), np.asarray(oref[k], np.float64)))
                                    for k in keys}
    rec['nch_slim_vs_ref_particles'] = bool(np.array_equal(old['nch'], ((G['P'][..., 5] != 0) & G['M']).sum(1).astype(np.int16)))
    order, nper = [int(r) for r in old['order']], int(old['nper'])
    # every tag must hold the same events and the same old values as the first
    rec['tags_identical'] = {}
    for t in tags[1:]:
        s = load_old(f'{OUTM}/models/{t}_ref_v2_slim.npz', keys + ['order', 'nper', 'nch'])
        same = all(np.array_equal(s[k], old[k]) for k in s) and (
            subprocess.run(['cmp', '-s', ref0, f'{OUTM}/models/{t}_ref.npz']).returncode == 0)
        rec['tags_identical'][t] = bool(same)
        assert same, f'{t} does not hold the events of {t0}'
    G['old'] = {k: np.asarray(old[k], np.float64) if k != 'strange' and k != 'nbaryon' else old[k] for k in keys}
    nrun = {r: nevents_npz(f'{data}/particles_full_{r:04d}.npz') for r in order}
    tasks, off = [], 0
    for k, (rid, idx) in enumerate(replay(order, nper, nrun)):
        tasks.append((k, rid, idx, f'{data}/particles_full_{rid:04d}.npz', f'{data}/shapes_run_{rid:04d}.csv',
                      f'{nbdir(data)}/shapes_run_{rid:04d}.csv', off))
        off += len(idx)
    assert off == len(old['mult_total']), (off, len(old['mult_total']))
    res = pmap(csv_rows, tasks, nproc)
    new = {o: np.empty(off, np.float64) for o in CSVOBS}; new['strange'] = np.empty(off, np.float32)
    agg, inexact = {}, {}
    nbe = 0
    for k, o_, nv, chk, nbev in res:
        n = len(nv['mult_total'])
        for o in new:
            new[o][o_:o_ + n] = nv[o]
        for c, v in chk.items():
            if c.endswith('_inexact'):
                inexact[c] = inexact.get(c, 0) + v
            else:
                agg[c] = agg.get(c, 0) + (not v)
        nbe += nbev
    rec['replay'] = dict(runs=len(tasks), events=off, nper=nper, data=data, data_nb=nbdir(data),
                         failures_per_check=agg, old_values_not_bitwise=inexact, beam_photon_events=nbe)
    assert not any(agg.values()), f'replay checks failed: {agg}'
    rec['changes'] = changes(old, new)
    G.clear()
    return new


def mix_rule(tags, nproc, rec, src_map=None):
    """mixture bundles whose runs are gone: the per-event rule on the bundle's particles"""
    t0 = tags[0]
    slim0, ref0 = f'{OUTM}/models/{t0}_ref_v2_slim.npz', f'{OUTM}/models/{t0}_ref.npz'
    keys = CSVOBS + ['strange', 'nbaryon']
    old = load_old(slim0, keys + ['order', 'nper', 'nch'])
    z = np.load(ref0)
    G['P'] = z['particles']; G['M'] = z['mask'].astype(bool)
    oref = load_old(ref0, keys)
    rec['slim_vs_ref_old_equal'] = {k: bool(np.array_equal(np.asarray(old[k], np.float64), np.asarray(oref[k], np.float64)))
                                    for k in keys}
    rec['nch_slim_vs_ref_particles'] = bool(np.array_equal(old['nch'], ((G['P'][..., 5] != 0) & G['M']).sum(1).astype(np.int16)))
    rec['strange_old_from_particles'] = bool(np.array_equal(strange_old(G['P'], G['M']), old['strange']))
    assert all(rec['slim_vs_ref_old_equal'].values()) and rec['nch_slim_vs_ref_particles']
    rec['tags_identical'] = {}
    for t in tags[1:]:
        s = load_old(f'{OUTM}/models/{t}_ref_v2_slim.npz', keys + ['order', 'nper', 'nch'])
        same = all(np.array_equal(s[k], old[k]) for k in s) and (
            subprocess.run(['cmp', '-s', ref0, f'{OUTM}/models/{t}_ref.npz']).returncode == 0)
        rec['tags_identical'][t] = bool(same)
        assert same, f'{t} does not hold the events of {t0}'
    G['old'] = {k: np.asarray(old[k], np.float64) for k in CSVOBS}
    G['nper'] = int(old['nper'])
    N = len(G['M'])
    edges = np.linspace(0, N, nproc * 8 + 1).astype(int)
    res = pmap(rule_chunk, [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a], nproc)
    new = {o: np.empty(N, np.float64) for o in CSVOBS}; new['strange'] = np.empty(N, np.float32)
    nball = np.empty(N, np.int64)
    st = dict(beam_events=0, miss=[0, 0, 0], neg=[0, 0, 0])
    for a, out, s, nb in res:
        n = len(nb)
        for o in new:
            new[o][a:a + n] = out[o]
        nball[a:a + n] = nb
        st['beam_events'] += s['beam_events']
        st['miss'] = [x + y for x, y in zip(st['miss'], s['miss'])]; st['neg'] = [x + y for x, y in zip(st['neg'], s['neg'])]
    rec['rule'] = dict(events=N, **st, fallback_names=SHAPES)
    rec['changes'] = changes(old, new)
    # where the source runs exist: compare with the _nb shapes files at the replayed rows
    if src_map is not None:
        order, nper = [int(r) for r in old['order']], int(old['nper'])
        G['new'] = new
        tasks = [(rid, k * nper, src_map(rid)) for k, rid in enumerate(order)]
        res = pmap(cmp_run, [t for t in tasks if t[2] is not None], nproc)
        cmpres = dict(runs=len(res), runs_without_source=sum(t[2] is None for t in tasks), events=0)
        for r, off_, vals in res:
            for k2, v in r.items():
                if k2.startswith('max_'):
                    cmpres[k2] = max(cmpres.get(k2, 0.0), v)
                else:
                    cmpres[k2] = cmpres.get(k2, 0) + v
            # where the rows exist, the refreshed values are those of the _nb shapes files, read as
            # every published script reads them (pandas); the rule stays only where they are gone
            for j, o in enumerate(CSVOBS):
                new[o][off_:off_ + len(vals)] = vals[:, j]
        cmpres['events_from_csv'] = cmpres['events']
        cmpres['events_from_rule'] = N - cmpres['events']
        rec['source_comparison'] = cmpres
        rec['changes'] = changes(old, new)
    G.clear()
    return new


def cmp_run(task):
    """one run with sources: particles, old values and new values against the source files"""
    rid, off, srcs = task
    nper = G['nper']
    n_run = sum(nevents_npz(f'{d}/particles_full_{rid:04d}.npz') for d in srcs)
    idx = np.random.default_rng(50000 + rid).choice(n_run, min(nper, n_run), replace=False)
    zs = [np.load(f'{d}/particles_full_{rid:04d}.npz') for d in srcs]
    P = np.concatenate([q['particles'] for q in zs])[idx]; M = np.concatenate([q['mask'] for q in zs])[idx]
    sl = slice(off, off + len(idx))
    out = dict(events=len(idx), particles_mismatch=int(not (np.array_equal(P, G['P'][sl]) and np.array_equal(M, G['M'][sl]))))
    so = pd.concat([pd.read_csv(f'{d}/shapes_run_{rid:04d}.csv', usecols=CSVOBS) for d in srcs], ignore_index=True)
    lines = []
    for d in srcs:
        t = open(f'{nbdir(d)}/shapes_run_{rid:04d}.csv').read().split('\n')
        head = t[0].split(','); lines += t[1:-1]
    ci = [head.index(o) for o in CSVOBS]
    rows = [lines[i].split(',') for i in idx]
    sn_exact = np.array([[float(r[c]) for c in ci] for r in rows])
    sn_pd = pd.concat([pd.read_csv(f'{nbdir(d)}/shapes_run_{rid:04d}.csv', usecols=CSVOBS) for d in srcs],
                      ignore_index=True)[CSVOBS].values[idx]
    ov = np.column_stack([G['old'][o][sl] for o in CSVOBS]); nv = np.column_stack([G['new'][o][sl] for o in CSVOBS])
    out['old_mismatch'] = int((so[CSVOBS].values[idx] != ov).sum())
    out['new_mismatch'] = int((sn_exact != nv).sum())
    out['new_mismatch_pandas'] = int((sn_pd != nv).sum())
    out['max_abs_new_diff'] = float(np.abs(sn_exact - nv).max())
    out['strange_new_mismatch'] = int((strange_new(P, M.astype(bool), BP.n_beam(P, M.astype(bool))) != G['new']['strange'][sl]).sum())
    out['max_abs_rule_minus_csv_pandas'] = float(np.abs(sn_pd - nv).max())
    return out, off, sn_pd


def write_mix(tags, new, rec):
    rec['files'] = {}
    for t in tags:
        for suf in ('_ref_v2_slim.npz', '_ref.npz'):
            src = f'{OUTM}/models/{t}{suf}'; dst = f'{DST}/models/{t}{suf}'
            z = np.load(src)
            nn = {k: new[k].astype(z[k].dtype) for k in CSVOBS + ['strange']}
            write_replaced(src, dst, nn)
            rec['files'][dst] = verify_written(src, dst, nn)
            assert rec['files'][dst]['ok'], (dst, rec['files'][dst])


def stage_ref(stage, nproc, rec):
    """C_P2augfullfac0 (gen/data_abc, stageC_design_aug.csv), E/F_P2v3fullfac0 (data_stagePURE17_v3,
    stage{E,F}_design_v3all.csv): the reference of r2_ladder.py under LADDER_LOWMEM, NREF_PER = 12000
    events from every training run in design order"""
    tag, data, design = {'C': ('C_P2augfullfac0', f'{GEN}/data_abc', f'{GEN}/designs/stageC_design_aug.csv'),
                         'E': ('E_P2v3fullfac0', f'{MC}/data_stagePURE17_v3', f'{MC}/stageE_design_v3all.csv'),
                         'F': ('F_P2v3fullfac0', f'{MC}/data_stagePURE17_v3', f'{MC}/stageF_design_v3all.csv')}[stage]
    src = f'{OUTM}/models/{tag}_ref.npz'
    d = pd.read_csv(design)
    order = [int(r) for r in d[d.role == 'train'].run_id]
    keys = CSVOBS + ['strange', 'nbaryon']
    z = np.load(src)
    G['P'] = z['particles']; G['M'] = z['mask'].astype(bool)
    old = {k: np.asarray(z[k]) for k in keys if k in z.files}
    G['old'] = {k: np.asarray(v, np.float64) if k in CSVOBS else v for k, v in old.items()}
    nrun = {r: nevents_npz(f'{data}/particles_full_{r:04d}.npz') for r in order}
    tasks, off = [], 0
    for k, (rid, idx) in enumerate(replay(order, 12000, nrun)):
        tasks.append((k, rid, idx, f'{data}/particles_full_{rid:04d}.npz', f'{data}/shapes_run_{rid:04d}.csv',
                      f'{nbdir(data)}/shapes_run_{rid:04d}.csv', off))
        off += len(idx)
    assert off == len(G['M']), (tag, off, len(G['M']))
    res = pmap(csv_rows, tasks, nproc)
    new = {o: np.empty(off, np.float64) for o in CSVOBS}; new['strange'] = np.empty(off, np.float32)
    agg, inexact = {}, {}; nbe = 0
    for k, o_, nv, chk, nbev in res:
        n = len(nv['mult_total'])
        for o in new:
            new[o][o_:o_ + n] = nv[o]
        for c, v in chk.items():
            if c.endswith('_inexact'):
                inexact[c] = inexact.get(c, 0) + v
            else:
                agg[c] = agg.get(c, 0) + (not v)
        nbe += nbev
    rec['replay'] = dict(tag=tag, runs=len(tasks), events=off, nper=12000, data=data, data_nb=nbdir(data), design=design,
                         failures_per_check=agg, old_values_not_bitwise=inexact, beam_photon_events=nbe)
    assert not any(agg.values()), f'{tag}: replay checks failed: {agg}'
    rec['changes'] = changes(old, new)
    G.clear()
    dst = f'{DST}/models/{tag}_ref.npz'
    nn = {k: new[k].astype(z[k].dtype) for k in CSVOBS + ['strange'] if k in z.files}
    write_replaced(src, dst, nn)
    rec['files'] = {dst: verify_written(src, dst, nn)}
    assert rec['files'][dst]['ok'], rec['files'][dst]


def wifi_job(nproc, rec):
    """held_* of the wifi bases: every event of the 48 held-out runs of data_stagePURE17_v3"""
    tags = ['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2', 'MIXGEOV3M8S9']
    data = f'{MC}/data_stagePURE17_v3'
    B0 = np.load(f'{OUTM}/wifi_{tags[0]}_basis.npz')
    rids = [int(r) for r in B0['held_rids']]
    obs = [str(o) for o in B0['obs']]
    rep = [o for o in obs if o in CSVOBS + ['strange']]
    new, chk = {}, dict(old_equal=0, strange_old_equal=0, nbaryon_equal=0, mult_minus_nbeam=0, runs=0, events=0)
    for rid in rids:
        z = np.load(f'{data}/particles_full_{rid:04d}.npz'); P, M = z['particles'], z['mask'].astype(bool)
        so = pd.read_csv(f'{data}/shapes_run_{rid:04d}.csv', usecols=CSVOBS)
        sn = pd.read_csv(f'{nbdir(data)}/shapes_run_{rid:04d}.csv', usecols=CSVOBS)
        nb = BP.n_beam(P, M)
        for o in CSVOBS:
            if o in rep:
                chk['old_equal'] += int(not np.array_equal(so[o].values.astype(np.float32), B0[f'held_{o}_{rid}']))
                new[f'held_{o}_{rid}'] = sn[o].values.astype(np.float32)
        chk['strange_old_equal'] += int(not np.array_equal(strange_old(P, M), B0[f'held_strange_{rid}']))
        chk['nbaryon_equal'] += int(not np.array_equal(z['nbaryon'].astype(np.float32), B0[f'held_nbaryon_{rid}']))
        chk['mult_minus_nbeam'] += int(not np.array_equal(so['mult_total'].values - nb, sn['mult_total'].values))
        new[f'held_strange_{rid}'] = strange_new(P, M, nb)
        chk['runs'] += 1; chk['events'] += len(M)
    rec['held'] = dict(rids=len(rids), replaced_obs=rep, failures=chk, data=data, data_nb=nbdir(data))
    assert chk['old_equal'] == chk['strange_old_equal'] == chk['nbaryon_equal'] == chk['mult_minus_nbeam'] == 0, chk
    rec['files'] = {}
    for t in tags:
        src = f'{OUTM}/wifi_{t}_basis.npz'; dst = f'{DST}/wifi_{t}_basis.npz'
        B = np.load(src)
        assert [int(r) for r in B['held_rids']] == rids and [str(o) for o in B['obs']] == obs
        same_old = all(np.array_equal(B[k], B0[k]) for k in new)
        rec.setdefault('bases_hold_same_old_values', {})[t] = bool(same_old)
        assert same_old, t
        write_replaced(src, dst, new)
        rec['files'][dst] = verify_written(src, dst, new)
        assert rec['files'][dst]['ok'], (dst, rec['files'][dst])


def release_job(nproc, rec):
    """release_ref/reference_MIX17.npz: the events of MIXGEOV3S9's reference, same order"""
    src = f'{R}/release_ref/reference_MIX17.npz'; dst = f'{RUN}/release_ref/reference_MIX17.npz'
    old = np.load(f'{OUTM}/models/MIXGEOV3S9_ref_v2_slim.npz')
    newb = np.load(f'{DST}/models/MIXGEOV3S9_ref_v2_slim.npz')
    z = np.load(src)
    keys = CSVOBS + ['strange']
    rec['old_equal_slim'] = {k: bool(np.array_equal(z[k], old[k])) for k in keys + ['nbaryon', 'nch']}
    assert all(rec['old_equal_slim'].values()), rec['old_equal_slim']
    nn = {k: np.asarray(newb[k]).astype(z[k].dtype) for k in keys}
    rec['changes'] = changes({k: z[k] for k in keys}, nn)
    write_replaced(src, dst, nn)
    rec['files'] = {dst: verify_written(src, dst, nn)}
    assert rec['files'][dst]['ok'], rec['files'][dst]


def e_src(k):
    """source folders of a training run of the 160k (k=1) or 320k (k=3) set: the base run and its
    data_trainext extensions, in the order of make_dataset_v3.py"""
    def f(rid):
        base = f'{GEN}/data_release' if os.path.exists(f'{GEN}/data_release/particles_full_{rid:04d}.npz') else f'{GEN}/data_v3'
        srcs = [base] + [f'{GEN}/data_trainext/k{j}' for j in range(1, k + 1)]
        return srcs if all(os.path.exists(f'{d}/particles_full_{rid:04d}.npz') for d in srcs) else None
    return f


def v4_src(rid):
    for d in (f'{GEN}/data_release', f'{GEN}/data_v3'):
        if os.path.exists(f'{d}/particles_full_{rid:04d}.npz'):
            return [d]
    return None


if __name__ == '__main__':
    JOB, NPROC = sys.argv[1], int(sys.argv[2])
    jobs = ['mix_v3', 'mix_ev20', 'mix_e160', 'mix_e320', 'mix_v4', 'stage_C', 'stage_E', 'stage_F', 'wifi', 'release'] \
        if JOB == 'all' else JOB.split(',')
    os.makedirs(f'{WORK}/results', exist_ok=True)
    for j in jobs:
        t0 = time.time()
        rec = dict(job=j, start=time.strftime('%c'), host=os.uname()[1],
                   md5=dict(beam_photons=subprocess.run(['md5sum', f'{HERE}/beam_photons.py'], capture_output=True, text=True).stdout.split()[0],
                            nb_core=subprocess.run(['md5sum', f'{HERE}/nb_core.py'], capture_output=True, text=True).stdout.split()[0]))
        print(f'=== {j} {rec["start"]}', flush=True)
        if j == 'mix_v3':
            tags = ['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2', 'MIXGEOV3S27F', 'MIXGEOV3M8S9']
            write_mix(tags, mix_csv(tags, f'{MC}/data_stagePURE17_v3', NPROC, rec), rec)
        elif j == 'mix_ev20':
            write_mix(['MIXGEOV3E20S9'], mix_csv(['MIXGEOV3E20S9'], f'{MC}/data_stagePURE17_v3ev20', NPROC, rec), rec)
        elif j == 'mix_e160':
            write_mix(['MIXGEOV3E160S9'], mix_rule(['MIXGEOV3E160S9'], NPROC, rec, e_src(1)), rec)
        elif j == 'mix_e320':
            write_mix(['MIXGEOV3E320S9'], mix_rule(['MIXGEOV3E320S9'], NPROC, rec, e_src(3)), rec)
        elif j == 'mix_v4':
            write_mix(['MIXGEOV4S9', 'MIXGEOV4L'], mix_rule(['MIXGEOV4S9', 'MIXGEOV4L'], NPROC, rec, v4_src), rec)
        elif j.startswith('stage_'):
            stage_ref(j[-1], NPROC, rec)
        elif j == 'wifi':
            wifi_job(NPROC, rec)
        elif j == 'release':
            release_job(NPROC, rec)
        rec['seconds'] = time.time() - t0
        json.dump(rec, open(f'{WORK}/results/refresh_{j}.json', 'w'), indent=1)
        print(f'=== {j} done in {rec["seconds"]:.0f} s', json.dumps(rec.get('changes', {}))[:2000], flush=True)
