#!/usr/bin/env python3
"""The 27 refreshed caches of RUN/output and lmx_runs/release_ref/reference_MIX17.npz under the v3
rule: only the events whose beam-photon mask changes from v2 to v3 may change, in every replaced
array (1_minus_thrust, B_total, rho_heavy, mult_total, strange, or their held_* forms).

    python delta_caches.py snap PLAN_TSV         before refresh3_bundles.py, once. Every refreshed file
        must still be B0's (size and sha256 of MANIFEST.tsv; the release reference is not listed there
        and only recorded), else nothing is saved and the script stops: someone changed it since.
        Saves the replaced arrays (the v2 values) to WORK/v2_arrays/ and the hashes to
        WORK/results/delta_snap.json; with that record present it does nothing.
    python delta_caches.py check PLAN_TSV        after refresh3_bundles.py. Per file and array: the
        elements that differ from the v2 values lie in events whose mask changes (v2 -> v3); in those
        events mult_total = original - n_beam(v3), strange = kaons / (kept - n_beam(v3)), and the shapes
        equal the v3 _nb CSV at the replayed row as pandas reads it, or, where the run is gone (V4),
        stored - (with - without) recomputed with the verifier's independent code (indep_shapes.py, to
        1e-6, same fallback; where that value moves from v2 by more than 1e-8, the refreshed one must
        move at least half as far, so a change above 1e-8 cannot be left out unseen). One exception
        outside those events: in the rows of a gone run the refresh
        recomputes the shapes on this node, and another CPU may round them differently from B0's node;
        there a shape may differ from v2 by up to 1e-14 (B0's code differs between this Mac and the
        cluster by at most 6.66e-16; the smallest real change seen is 2.4e-9). Such values are counted
        and reported; anything larger, or
        any difference in a row read from a CSV, fails. The rows of a run are found from the replay's
        sizes (min(nper, events of the run); a gone run has nper, and the sizes must add up).
        WORK/results/delta_check.json; exit 1 on any failure.
    python delta_caches.py manifest PLAN_TSV     after a passed check: size and sha256 of the refreshed
        rows of RUN/output/MANIFEST.tsv recomputed (temp file, then rename); every other row unchanged.
    python delta_caches.py rule PLAN_TSV         after the manifest: RUN/output/bundle_rule.json (I1),
        every refreshed cache and the release reference by real path, with the sha256 of MANIFEST.tsv
        (the release reference: of manifest3.json), the md5 of the beam_photons.py that refreshed it
        (WORK/results/refresh_<job>.json) and the rule version. RUN's beam_photons.py must be that same
        file (sync_run.sh copies it), else nothing is written (exit 1). A table that already holds the
        same rule and bundles is left as it is ('unchanged'): every scoring's fingerprint holds its md5.
PLAN_TSV is B0's layout_plan.tsv (the rows of kind 'refreshed'). Reads the originals only."""
import os, sys, json, time, glob, hashlib, zipfile, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool

R = '/mnt/beegfs/bassi/reweighting'
MC, GEN, OUTM = f'{R}/maxent_code', f'{R}/gen', f'{R}/maxent_code/output'
RUN = f'{R}/lmx_runs'
OUT = f'{RUN}/maxent_code/output'
REL_SRC, REL_DST = f'{R}/release_ref/reference_MIX17.npz', f'{RUN}/release_ref/reference_MIX17.npz'
WORK = f'{R}/lmx_phaseB/observables_v3'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beam_photons as BP                           # noqa: E402  v3
import beam_photons_v2 as BP2                       # noqa: E402  B0's v2
from indep_shapes import shapes, rebuild            # noqa: E402  the verifier's independent code

CSVOBS = ['1_minus_thrust', 'B_total', 'rho_heavy', 'mult_total']
OBS = CSVOBS + ['strange']
SHAPES = CSVOBS[:3]
ULP = 1e-14                 # |new - v2| allowed in a gone run's shapes outside the changed events
RULE_TOL = 1e-6             # |new - independent| allowed in a gone run's changed events (the codes differ in their
                            # rebuild and thrust: B5's indep_check3 saw up to 9.4e-10)
RULE_MOVE = 1e-8            # where the independent value moves from v2 by more than this, the new one must move at
                            # least half as far
RUN_BP = f'{RUN}/maxent_code/beam_photons.py'
RULE_JSON = f'{OUT}/bundle_rule.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def nbdir(d):
    if d.startswith(GEN + '/'):
        rel = d[len(GEN) + 1:].split('/'); rel[0] += '_nb'
        return GEN + '/' + '/'.join(rel)
    return d + '_nb'


def nev(path):
    with zipfile.ZipFile(path) as zf, zf.open('mask.npy') as fh:
        v = np.lib.format.read_magic(fh)
        return int((np.lib.format.read_array_header_1_0(fh) if v == (1, 0) else np.lib.format.read_array_header_2_0(fh))[0][0])


def sources(tag):
    """(data folders of a training run or None, nper, order) of the bundle of `tag` (verify_refresh.py)"""
    if tag.startswith('MIXGEO'):
        sl = np.load(f'{OUTM}/models/{tag}_ref_v2_slim.npz')
        order, nper = [int(r) for r in sl['order']], int(sl['nper'])
        if tag == 'MIXGEOV3E20S9':
            return (lambda r: [f'{MC}/data_stagePURE17_v3ev20']), nper, order
        if tag in ('MIXGEOV3E160S9', 'MIXGEOV3E320S9'):
            k = 1 if '160' in tag else 3
            def f(r, k=k):
                b = f'{GEN}/data_release' if os.path.exists(f'{GEN}/data_release/particles_full_{r:04d}.npz') else f'{GEN}/data_v3'
                srcs = [b] + [f'{GEN}/data_trainext/k{j}' for j in range(1, k + 1)]
                return srcs if all(os.path.exists(f'{d}/particles_full_{r:04d}.npz') for d in srcs) else None
            return f, nper, order
        if tag.startswith('MIXGEOV4'):
            def g(r):
                for d in (f'{GEN}/data_release', f'{GEN}/data_v3'):
                    if os.path.exists(f'{d}/particles_full_{r:04d}.npz'):
                        return [d]
                return None
            return g, nper, order
        return (lambda r: [f'{MC}/data_stagePURE17_v3']), nper, order
    data, design = {'C': (f'{GEN}/data_abc', f'{GEN}/designs/stageC_design_aug.csv'),
                    'E': (f'{MC}/data_stagePURE17_v3', f'{MC}/stageE_design_v3all.csv'),
                    'F': (f'{MC}/data_stagePURE17_v3', f'{MC}/stageF_design_v3all.csv')}[tag[0]]
    d = pd.read_csv(design)
    return (lambda r: [data]), 12000, [int(r) for r in d[d.role == 'train'].run_id]


def refreshed(plan):
    rows = [l.rstrip('\n').split('\t') for l in open(plan)]
    head = rows[0]
    rel = [dict(zip(head, r))['path'] for r in rows[1:] if dict(zip(head, r))['kind'] == 'refreshed']
    return rel + ['release_ref/reference_MIX17.npz']


def dst_of(rel):
    return REL_DST if rel.startswith('release_ref/') else f'{OUT}/{rel}'


def src_of(rel):
    return REL_SRC if rel.startswith('release_ref/') else f'{OUTM}/{rel}'


def replaced_keys(z):
    return [k for k in z.files if k in OBS or (k.startswith('held_') and any(k.startswith(f'held_{o}_') for o in OBS))]


def snapname(rel):
    return f"{WORK}/v2_arrays/{rel.replace('/', '__')}"


# ---------------------------------------------------------------------------------------- snap
SNAP = f'{WORK}/results/delta_snap.json'


def hash_one(rel):
    p = dst_of(rel)
    return rel, os.path.getsize(p), sha256(p)


def save_one(rel):
    z = np.load(dst_of(rel))
    arr = {k: np.asarray(z[k]) for k in replaced_keys(z)}
    tmp = snapname(rel) + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, **arr)
    os.replace(tmp, snapname(rel))
    return len(arr)


def snap(plan):
    """once, before the refresh: a finished snapshot is never taken again (a requeued job after the
    refresh would otherwise save v3 values as v2), and nothing is saved unless every file is B0's"""
    if os.path.exists(SNAP):
        print(f'snap: the snapshot of {json.load(open(SNAP))["taken"]} exists; kept')
        return 0
    man = {}
    for l in list(open(f'{OUT}/MANIFEST.tsv'))[1:]:
        q = l.rstrip('\n').split('\t')
        man[q[0]] = q
    rels = refreshed(plan)
    with Pool(8) as pool:
        res = pool.map(hash_one, rels, chunksize=1)
    rec, bad = {}, []
    for rel, size, h in res:
        rec[rel] = dict(size=size, sha256=h)
        if rel in man:
            ok = man[rel][1] == 'refreshed' and int(man[rel][3]) == size and man[rel][4] == h
            rec[rel]['manifest_ok'] = ok
            if not ok:
                bad.append(rel)
        elif not rel.startswith('release_ref/'):
            bad.append(rel + ' (not in MANIFEST.tsv)')
    if bad:
        json.dump(dict(files=rec, bad=bad), open(f'{WORK}/results/delta_snap_refused.json', 'w'), indent=1)
        print(f'snap REFUSED: {len(bad)} files are not B0\'s (MANIFEST.tsv): {bad}')
        return 1
    os.makedirs(f'{WORK}/v2_arrays', exist_ok=True)
    with Pool(8) as pool:
        n = pool.map(save_one, rels, chunksize=1)
    for rel, k in zip(rels, n):
        rec[rel]['arrays'] = k
    tmp = f'{SNAP}.tmp{os.getpid()}'
    json.dump(dict(taken=time.strftime('%c'), files=rec, bad=[]), open(tmp, 'w'), indent=1)
    os.replace(tmp, SNAP)
    print(f'snap: {len(rec)} files, all B0\'s; {sum(n)} arrays saved')
    return 0


# ---------------------------------------------------------------------------------------- check
def strange_new(P, M, nb):
    lm = P[..., 3]
    return ((((lm > -0.36) & (lm < -0.20)) & M).sum(1) / np.maximum(M.sum(1) - nb, 1)).astype(np.float32)


def masks(P, M):
    F3 = BP.beam_photon_mask(P, M)
    F2 = BP2.beam_photon_mask(P, M)
    return (F3 != F2).any(1), F3


def differ(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return ~((a == b) | (np.isnan(a) & np.isnan(b))) if a.dtype.kind == 'f' else a != b


def rule_value(f, fl, st):
    """stored - (with - without) by the verifier's code, with the fallback; f kept features, fl v3 flags"""
    if not fl.any():
        return st, np.ones(3, bool)
    e, p = rebuild(f, 'balance')
    w = shapes(e, p); wo = shapes(e[~fl], p[~fl])
    cand = st - (w - wo)
    fb = ~(np.abs(w - st) <= 1e-2) | ~(cand >= 0)
    return np.where(fb, st, cand), fb


def layout(tag, srcf, nper, order, N):
    """per training run in order: (rid, first row, rows, sources or None); the sizes of the replay,
    min(nper, events of the run), and nper for a gone run; they must add up to the bundle's events"""
    lay, off = [], 0
    for rid in order:
        srcs = srcf(rid)
        n = nper if srcs is None else min(nper, sum(nev(f'{d}/particles_full_{rid:04d}.npz') for d in srcs))
        lay.append((rid, off, n, srcs))
        off += n
    if off != N:
        raise RuntimeError(f'{tag}: the replay gives {off} rows, the bundle holds {N}')
    return lay


def check_bundle(rel):
    """models/<tag>_ref.npz, models/<tag>_ref_v2_slim.npz, the release reference (MIXGEOV3S9's events)"""
    tag = 'MIXGEOV3S9' if rel.startswith('release_ref/') else os.path.basename(rel).split('_ref')[0]
    zp = np.load(f'{OUTM}/models/{tag}_ref.npz')                 # the particles (unchanged by every refresh)
    P, M = zp['particles'], zp['mask'].astype(bool)
    X, F3 = masks(P, M)
    nb3 = F3.sum(1)
    new, v2, orig = np.load(dst_of(rel)), np.load(snapname(rel)), np.load(src_of(rel))
    out = dict(path=rel, tag=tag, events=int(len(M)), mask_changed=int(X.sum()), arrays={}, ok=True)
    srcf, nper, order = sources(tag)
    lay = layout(tag, srcf, nper, order, len(M))
    gone = np.zeros(len(M), bool)
    for rid, a, n, srcs in lay:
        gone[a:a + n] = srcs is None
    out['gone_rows'] = int(gone.sum())
    for k in replaced_keys(new):
        x, y = np.asarray(new[k]), np.asarray(v2[k])
        o = differ(x, y) & ~X
        r = dict(differ=int(differ(x, y).sum()), differ_outside_changed=int(o.sum()))
        if o.any():
            gap = np.abs(x[o].astype(np.float64) - y[o].astype(np.float64))
            ulp = (gap <= ULP) & gone[o] if (k in SHAPES and x.dtype == np.float64) else np.zeros(len(gap), bool)
            r.update(outside_rounding=int(ulp.sum()), outside_bad=int((~ulp).sum()), outside_max=float(gap.max()),
                     outside_max_rounding=float(gap[ulp].max()) if ulp.any() else 0.0)
            out['ok'] &= bool(ulp.all())
        out['arrays'][k] = r
    out['rounding_rows'] = int(sum(a.get('outside_rounding', 0) for a in out['arrays'].values()))
    xi = np.where(X)[0]
    if len(xi) == 0:
        return out
    chk = dict(mult_bad=0, strange_bad=0, csv_rows=0, csv_bad=0, rule_rows=0, rule_bad=0, rule_fallback_bad=0, rule_max_diff=0.0,
               rule_stale=0, rule_bad_rows=[])
    if 'mult_total' in new.files:
        chk['mult_bad'] = int((np.asarray(new['mult_total'], np.float64)[xi] != np.asarray(orig['mult_total'], np.float64)[xi] - nb3[xi]).sum())
    if 'strange' in new.files:
        chk['strange_bad'] = int((np.asarray(new['strange'])[xi] != strange_new(P[xi], M[xi], nb3[xi])).sum())
    for rid, a, n, srcs in lay:
        rows = xi[(xi >= a) & (xi < a + n)]
        if len(rows) == 0:
            continue
        if srcs is None:                                           # the run is gone: the rule
            for i in rows:
                st = np.array([float(orig[o][i]) for o in SHAPES])
                exp, fb = rule_value(P[i][M[i]], F3[i][M[i]], st)
                got = np.array([float(new[o][i]) for o in SHAPES])
                old2 = np.array([float(v2[o][i]) for o in SHAPES])
                chk['rule_rows'] += 1
                fbad = bool((fb & (got != st)).any())
                chk['rule_fallback_bad'] += int(fbad)
                chk['rule_max_diff'] = max(chk['rule_max_diff'], float(np.abs(got - exp).max()))
                far = ~(np.abs(got - exp) <= RULE_TOL)
                moved = np.abs(exp - old2)
                stale = (moved > RULE_MOVE) & ~(np.abs(got - old2) >= 0.5 * moved)     # the change left out
                chk['rule_bad'] += int(far.any())
                chk['rule_stale'] += int(stale.any())
                if (fbad or far.any() or stale.any()) and len(chk['rule_bad_rows']) < 20:
                    chk['rule_bad_rows'].append(dict(row=int(i), rid=rid, stored=st.tolist(), got=got.tolist(), indep=exp.tolist(),
                                                     v2=old2.tolist()))
            continue
        n_run = sum(nev(f'{d}/particles_full_{rid:04d}.npz') for d in srcs)
        idx = np.random.default_rng(50000 + rid).choice(n_run, min(nper, n_run), replace=False)
        Vp = pd.concat([pd.read_csv(f'{nbdir(d)}/shapes_run_{rid:04d}.csv', usecols=CSVOBS) for d in srcs],
                       ignore_index=True)[CSVOBS].values[idx]
        loc = rows - a
        for j, o in enumerate(CSVOBS):
            if o in new.files:
                chk['csv_bad'] += int((np.asarray(new[o], np.float64)[rows] != Vp[loc, j]).sum())
        chk['csv_rows'] += len(rows)
    out['changed_events'] = chk
    out['ok'] &= chk['mult_bad'] == chk['strange_bad'] == chk['csv_bad'] == chk['rule_bad'] == chk['rule_fallback_bad'] == \
        chk['rule_stale'] == 0
    return out


def check_wifi(rel):
    new, v2, orig = np.load(dst_of(rel)), np.load(snapname(rel)), np.load(src_of(rel))
    rids = [int(r) for r in new['held_rids']]
    obs = [o for o in OBS if f'held_{o}_{rids[0]}' in new.files]
    data = f'{MC}/data_stagePURE17_v3'
    out = dict(path=rel, runs=len(rids), events=0, mask_changed=0, differ=0, differ_outside_changed=0,
               csv_bad=0, mult_bad=0, strange_bad=0, ok=True)
    for rid in rids:
        z = np.load(f'{data}/particles_full_{rid:04d}.npz'); P, M = z['particles'], z['mask'].astype(bool)
        X, F3 = masks(P, M); nb3 = F3.sum(1)
        out['events'] += len(M); out['mask_changed'] += int(X.sum())
        for o in obs:
            k = f'held_{o}_{rid}'
            d = differ(new[k], v2[k])
            out['differ'] += int(d.sum()); out['differ_outside_changed'] += int((d & ~X).sum())
        if X.any():
            xi = np.where(X)[0]
            sn = pd.read_csv(f'{nbdir(data)}/shapes_run_{rid:04d}.csv', usecols=CSVOBS)
            for o in obs:
                if o in CSVOBS:
                    out['csv_bad'] += int((new[f'held_{o}_{rid}'][xi] != sn[o].values.astype(np.float32)[xi]).sum())
            if 'mult_total' in obs:
                out['mult_bad'] += int((np.asarray(new[f'held_mult_total_{rid}'], np.float64)[xi] !=
                                        np.asarray(orig[f'held_mult_total_{rid}'], np.float64)[xi] - nb3[xi]).sum())
            out['strange_bad'] += int((new[f'held_strange_{rid}'][xi] != strange_new(P[xi], M[xi], nb3[xi])).sum())
    out['ok'] = out['differ_outside_changed'] == out['csv_bad'] == out['mult_bad'] == out['strange_bad'] == 0
    return out


def check(plan):
    if not os.path.exists(SNAP):
        print('check: no finished snapshot (delta_caches.py snap before the refresh)')
        return 1
    res = []
    for rel in refreshed(plan):
        try:
            r = check_wifi(rel) if os.path.basename(rel).startswith('wifi_') else check_bundle(rel)
        except Exception as e:                                      # e.g. runs that do not add up: BAD, said why
            r = dict(path=rel, ok=False, error=f'{type(e).__name__}: {e}')
        res.append(r)
        print(('OK  ' if r['ok'] else 'BAD ') + json.dumps(r)[:600], flush=True)
    json.dump(res, open(f'{WORK}/results/delta_check.json', 'w'), indent=1)
    print('DELTA CHECK:', sum(r['ok'] for r in res), 'of', len(res), 'ok;', sum(r.get('rounding_rows', 0) for r in res),
          f'values of gone runs outside the changed events differ from v2 by rounding (<= {ULP:g}) and are accepted')
    return 0 if all(r['ok'] for r in res) else 1


# ---------------------------------------------------------------------------------------- manifest
def manifest(plan):
    rels = set(refreshed(plan))
    lines = list(open(f'{OUT}/MANIFEST.tsv'))
    head = lines[0].rstrip('\n').split('\t')
    assert head == ['path', 'kind', 'source', 'size', 'sha256'], head
    rows = [l.rstrip('\n').split('\t') for l in lines[1:]]
    todo = [r[0] for r in rows if r[1] == 'refreshed']
    assert set(todo) <= rels, set(todo) - rels
    with Pool(8) as pool:
        hs = dict(zip(todo, pool.map(sha256, [f'{OUT}/{p}' for p in todo])))
    rec = {}
    for r in rows:
        if r[1] == 'refreshed':
            p = f'{OUT}/{r[0]}'
            rec[r[0]] = dict(old_size=int(r[3]), old_sha256=r[4], size=os.path.getsize(p), sha256=hs[r[0]])
            r[3], r[4] = str(os.path.getsize(p)), hs[r[0]]
    tmp = f'{OUT}/MANIFEST.tsv.tmp{os.getpid()}'
    with open(tmp, 'w') as fh:
        fh.write(lines[0])
        for r in rows:
            fh.write('\t'.join(r) + '\n')
    os.replace(tmp, f'{OUT}/MANIFEST.tsv')
    rec['_release_ref'] = dict(sha256=sha256(REL_DST), size=os.path.getsize(REL_DST))
    json.dump(rec, open(f'{WORK}/results/manifest3.json', 'w'), indent=1)
    print(f'MANIFEST.tsv: {len(todo)} refreshed rows updated, {sum(v["old_sha256"] != v["sha256"] for k, v in rec.items() if not k.startswith("_"))} with a new sha256')
    return 0


# ---------------------------------------------------------------------------------------- rule
def md5file(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def kind_of(rel):
    b = os.path.basename(rel)
    if rel.startswith('release_ref/'):
        return 'release_ref'
    if b.startswith('wifi_'):
        return 'wifi'
    if b.endswith('_ref_v2_slim.npz'):
        return 'slim'
    if b[:2] in ('C_', 'E_', 'F_'):
        return f'stage{b[0]}'
    return 'ref'


def rule(plan):
    """RUN/output/bundle_rule.json (I1), once the manifest holds the new sha256 values"""
    rels = refreshed(plan)
    md5_pkg = md5file(f'{HERE}/beam_photons.py')
    md5_run = md5file(RUN_BP) if os.path.isfile(RUN_BP) else None
    if md5_run != md5_pkg:
        print(f'rule REFUSED: RUN\'s beam_photons.py ({RUN_BP}) has md5 {md5_run}, this package\'s {md5_pkg}: '
              f'copy the code into RUN first (sync_run.sh), then run this step again (STEPS=rule)')
        return 1
    made = {}                                       # real path of a written file -> md5 of the rule it used
    for f in sorted(glob.glob(f'{WORK}/results/refresh_*.json')):
        r = json.load(open(f))
        for dst in r.get('files', {}):
            made[os.path.realpath(dst)] = r['md5']['beam_photons']
    man = {q[0]: q for q in (l.rstrip('\n').split('\t') for l in list(open(f'{OUT}/MANIFEST.tsv'))[1:])}
    m3 = json.load(open(f'{WORK}/results/manifest3.json'))
    bundles, bad = {}, []
    for rel in rels:
        real = os.path.realpath(dst_of(rel))
        if rel.startswith('release_ref/'):
            sha = m3['_release_ref']['sha256']
        elif rel in man and man[rel][1] == 'refreshed' and m3.get(rel, {}).get('sha256') == man[rel][4]:
            sha = man[rel][4]
        else:
            bad.append(f'{rel}: MANIFEST.tsv and manifest3.json disagree or the row is missing'); continue
        if made.get(real) != md5_pkg:
            bad.append(f'{rel}: refreshed with beam_photons md5 {made.get(real)}, not {md5_pkg}'); continue
        bundles[real] = dict(sha256=sha, beam_photons_md5=made[real], rule_version=BP.RULE_VERSION, kind=kind_of(rel))
    if bad:
        print('rule REFUSED:', bad)
        return 1
    try:
        have = json.load(open(RULE_JSON)) if os.path.exists(RULE_JSON) else {}
    except ValueError:
        have = {}
    if (have.get('rule_version'), have.get('beam_photons_md5'), have.get('bundles')) == (BP.RULE_VERSION, md5_run, bundles):
        print(f'bundle_rule.json: unchanged ({len(bundles)} bundles, rule {BP.RULE_VERSION}, beam_photons md5 {md5_run}, '
              f'written {have.get("written")}); not rewritten')
        return 0
    rec = dict(rule_version=BP.RULE_VERSION, beam_photons_md5=md5_run,
               written=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), bundles=bundles)
    tmp = f'{RULE_JSON}.tmp{os.getpid()}'
    json.dump(rec, open(tmp, 'w'), indent=1)
    os.replace(tmp, RULE_JSON)
    print(f'bundle_rule.json: {len(bundles)} bundles, rule {BP.RULE_VERSION}, beam_photons md5 {md5_run}')
    return 0


if __name__ == '__main__':
    mode, plan = sys.argv[1], sys.argv[2]
    os.makedirs(f'{WORK}/results', exist_ok=True)
    sys.exit({'snap': snap, 'check': check, 'manifest': manifest, 'rule': rule}[mode](plan))
