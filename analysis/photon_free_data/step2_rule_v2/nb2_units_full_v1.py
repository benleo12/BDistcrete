#!/usr/bin/env python3
"""Phase B: the photon-free columns of every _nb CSV again, with the mask of beam_photons.py
(Phase A's rule plus (i) the axis on the beam and (ii) consistent photons).

Per unit (one real particle file and its CSV, the units of Phase A):
  1_minus_thrust, B_total, rho_heavy = stored - (with - without), stored = the ORIGINAL CSV value,
      with/without = rebuilt event with and without the beam photons (nb_core: parts, scale_of,
      four, three = gen/compute_efps.compute_hemisphere_shapes), exactly as in Phase A;
      fallback: keep the stored value of an observable in an event where |with - stored| > 1e-2
      or the new value is negative (or either is not finite);
  mult_total = stored - n_beam;
  every other column and every row without a beam photon: the original text, byte for byte.
The new CSV is written to a temp file in the folder of the _nb CSV and renamed onto it.

    python nb2_units.py WORK NPROC MODE [UIDS]
MODE 'run': dynamic workers over WORK/units.tsv (claims in WORK/claims), replace the _nb CSVs.
MODE 'test': the units named in UIDS (comma list), write WORK/test/<uid>.{csv,json}, replace nothing.
A unit is done when WORK/units/<uid>.json exists. The record is first written to
WORK/pending/<uid>.json with the md5 of the new text, so a unit interrupted after the rename is
recognised on restart (the _nb CSV then has that md5) and is not computed from its own output.
"""
import os, sys, time, json, hashlib, traceback
import numpy as np

WORK, NPROC, MODE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
UIDS = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nb_core as C                     # noqa: E402  Phase A's rebuild (md5 1ed8468b...)
import beam_photons as BP               # noqa: E402  the new mask (md5 77d8092b...)

OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
MISS_TOL = 1e-2
THRUST_EDGES = np.array([0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .12, .14, .16, .18, .20,
                         .25, .30, .35, .40, .50])
RHO_EDGES = np.r_[np.arange(0, 0.1, 0.01), 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.4]
B_EDGES = np.round(np.arange(0, 0.4201, 0.02), 4)
EDGES = {'1_minus_thrust': THRUST_EDGES, 'B_total': B_EDGES, 'rho_heavy': RHO_EDGES}
TLIMIT = float(os.environ.get('NB_TLIMIT', '1e9'))
T0 = time.time()
JOB = os.environ.get('SLURM_JOB_ID', 'local')
RST = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)


def hist(v, e):
    return [int((v < e[0]).sum())] + np.histogram(v, e)[0].tolist() + [int((v >= e[-1]).sum())]


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def split_csv(text):
    lines = text.split('\n')
    trail = lines[-1] == ''
    return lines[0], (lines[1:-1] if trail else lines[1:]), trail


def parse(body, cols):
    out = np.empty((len(body), len(cols)))
    for i, l in enumerate(body):
        q = l.split(',')
        out[i] = [float(q[c]) for c in cols]
    return out


def probe(P, M, F, BA, chunk=20000):
    """per-event beam energy (new and Phase A masks), the energy the mask changes, and the (i)
    signature (|cos| >= 1-1e-7, phi 0 or pi) where the base rule flags something: there the
    flagged photons tell whether the axis lies on the beam, so a signature photon off the beam
    measures how often rule (i) could flag a photon that is not a beam photon"""
    N = len(M)
    zb = np.zeros(N); zbA = np.zeros(N); zg = zl = 0.0
    sig = dict(events=0, axis_on_beam=0, axis_off_beam=0, E_off_GeV=[])
    for a in range(0, N, chunk):
        p = P[a:a + chunk]; m = M[a:a + chunk]; f = F[a:a + chunk]; ba = BA[a:a + chunk]
        z = p[..., 0].astype(np.float64)
        zb[a:a + chunk] = np.where(f, z, 0).sum(1); zbA[a:a + chunk] = np.where(ba, z, 0).sum(1)
        zg += float(np.where(f & ~ba, z, 0).sum()); zl += float(np.where(ba & ~f, z, 0).sum())
        ct = p[..., 1].astype(np.float64); ph = p[..., 2].astype(np.float64)
        neu = m & (p[..., 4] == 0) & (p[..., 5] == 0) & (np.abs(p[..., 3].astype(np.float64) + 3.0) < BP.LM_TOL)
        base = neu & (np.abs(np.abs(ph) - np.pi / 2) < BP.PHI_TOL)
        sg = neu & (np.abs(ct) >= BP.COS_AXIS) & ((p[..., 2] == 0) | (np.abs(ph) == BP.PI32)) & ~base
        for i in np.where(sg.any(1) & base.any(1))[0]:
            e = z[i] * BP.ECM
            c = BP._implied_cos(ct[i], ph[i], e)
            hb = base[i] & (e > BP.E_MIN)
            ci = c[hb] if hb.any() else c[base[i]]
            sig['events'] += 1
            if np.all(np.abs(np.abs(ci) - 1) < 1e-5):
                sig['axis_on_beam'] += 1
            else:
                sig['axis_off_beam'] += 1
                sig['E_off_GeV'].append(float(e[sg[i]].max()))
    return dict(zb=zb, zbA=zbA, z_gained=zg, z_lost=zl, sig=sig)


def compute(u):
    """new text and statistics of one unit (nothing written)"""
    t0 = time.time()
    z = np.load(u['npz'])
    P, M = z['particles'], z['mask'].astype(bool)
    N, PMAX = M.shape
    text_in = open(u['csv_in']).read()
    text_a = open(u['csv_out']).read()
    head, body, trail = split_csv(text_in)
    head_a, body_a, trail_a = split_csv(text_a)
    if len(body) != N or len(body_a) != N or head_a != head or trail_a != trail:
        raise RuntimeError(f"{u['uid']}: rows {len(body)} / Phase A {len(body_a)} / npz {N}, header same {head == head_a}")
    hl = head.split(',')
    col = [hl.index(o) for o in OBS]
    cm = hl.index('mult_total')
    S = parse(body, col + [cm])
    SA = parse(body_a, col + [cm])
    mult = S[:, 3]; S = S[:, :3]; mult_a = SA[:, 3]; SA = SA[:, :3]
    t_load = time.time() - t0
    F, info = BP.beam_photon_mask(P, M, detail=True)
    BA = C.beam_mask(P, M)                                   # Phase A's mask, for the comparison
    nb = F.sum(1)
    idx = np.where(nb > 0)[0]
    W = np.empty((len(idx), 3)); WO = np.empty((len(idx), 3)); sc = np.empty(len(idx)); clamp = 0
    for k, i in enumerate(idx):
        f = P[i][M[i]]; b = F[i][M[i]]
        zz, mm, UU = C.parts(f)
        s, cl = C.scale_of(zz, mm, UU)
        clamp += cl
        ev = C.four(zz, mm, UU, s)
        W[k] = C.three(ev); WO[k] = C.three(ev[~b]); sc[k] = s
    t_main = time.time() - t0 - t_load
    st = S[idx]
    cand = st - (W - WO)
    miss = ~(np.abs(W - st) <= MISS_TOL)                     # also catches a NaN
    neg = ~miss & ~(cand >= 0)
    use = ~(miss | neg)
    new = S.copy()
    nv = st.copy(); nv[use] = cand[use]; new[idx] = nv
    mult_new = mult - nb
    # the new text
    out_body = list(body)
    for k, i in enumerate(idx):
        q = body[i].split(',')
        for j, c in enumerate(col):
            if use[k, j]:
                q[c] = repr(float(new[i, j]))
        q[cm] = repr(float(mult_new[i]))
        out_body[i] = ','.join(q)
    text_new = '\n'.join([head] + out_body) + ('\n' if trail else '')
    # comparisons with Phase A, per event
    same_mask = (F == BA).all(1)
    gained = (F & ~BA).any(1); lost = (BA & ~F).any(1)
    tok_new = np.array([[l.split(',')[c] for c in col] for l in (out_body[i] for i in idx)]).reshape(-1, 3)
    tok_a = np.array([[l.split(',')[c] for c in col] for l in (body_a[i] for i in idx)]).reshape(-1, 3)
    reg = same_mask[idx][:, None] & use                       # must equal Phase A's text exactly
    reg_mismatch = int((reg & (tok_new != tok_a)).sum())
    # rows outside idx: identical to the original by construction; Phase A changed rows only where BA
    nbA = BA.sum(1)
    a_changed_rows = int(sum(body_a[i] != body[i] for i in range(N)))
    pr = probe(P, M, F, BA)
    trunc = M.sum(1) == PMAX
    stats = dict(uid=u['uid'], group=u['group'], rid=int(u['rid']), gen=u['gen'], npz=u['npz'], csv_in=u['csv_in'],
                 csv_out=u['csv_out'], n_events=int(N), pmax=int(PMAX),
                 n_beam_events=int(len(idx)), n_beam_events_A=int((nbA > 0).sum()),
                 n_hist=np.bincount(np.minimum(nb, 4), minlength=5).tolist(),
                 n_hist_A=np.bincount(np.minimum(nbA, 4), minlength=5).tolist(),
                 n_beam_sum=int(nb.sum()), n_beam_sum_A=int(nbA.sum()),
                 mask=dict(events_differ=int((~same_mask).sum()), gained=int(gained.sum()), lost=int(lost.sum()),
                           rule_i_events=int(info['rule_i'].sum()), rule_i_photons=int((F & ~BA).sum()),
                           consistency_events=int((info['n_dropped'] > 0).sum()),
                           consistency_dropped=int(info['n_dropped'].sum()),
                           z_gained=pr['z_gained'], z_lost=pr['z_lost'],
                           other_differ=int((~same_mask & ~info['rule_i'] & (info['n_dropped'] == 0)).sum())),
                 sum_zbeam=float(pr['zb'].sum()), sum_zbeam_A=float(pr['zbA'].sum()),
                 n_ebeam_gt5=int((pr['zb'] * C.ECM > 5).sum()), n_ebeam_gt20=int((pr['zb'] * C.ECM > 20).sum()),
                 truncated=dict(events=int(trunc.sum()), mult_gt_pmax=int((mult > PMAX).sum()),
                                n_beam_sum=int(nb[trunc].sum())),
                 signature_in_base_events=pr['sig'],
                 scale=dict(clamped=int(clamp), lt88=int((sc < 88).sum())),
                 regression_mismatch=reg_mismatch, regression_checked=int(reg.sum()),
                 phaseA_changed_rows=a_changed_rows, mult_a_equals_orig=bool(np.array_equal(mult_a, mult)),
                 mult=dict(sum_orig=float(mult.sum()), sum_new=float(mult_new.sum()), min_new=float(mult_new.min())),
                 obs={}, time=dict(load=t_load, main=t_main))
    for j, o in enumerate(OBS):
        ed = EDGES[o]
        r = np.abs(W[:, j] - st[:, j])
        stats['obs'][o] = dict(
            sum_orig=float(S[:, j].sum()), sum_A=float(SA[:, j].sum()), sum_new=float(new[:, j].sum()),
            sumsq_orig=float((S[:, j] ** 2).sum()), sumsq_A=float((SA[:, j] ** 2).sum()),
            sumsq_new=float((new[:, j] ** 2).sum()),
            fallback_miss=int(miss[:, j].sum()), fallback_neg=int(neg[:, j].sum()),
            nonfinite=int((~np.isfinite(W[:, j]) | ~np.isfinite(WO[:, j])).sum()),
            min_new=float(new[:, j].min()), n_neg_new=int((new[:, j] < 0).sum()),
            n_differ_A=int((new[:, j] != SA[:, j]).sum()), max_abs_new_minus_A=float(np.abs(new[:, j] - SA[:, j]).max()),
            sum_new_minus_A=float((new[:, j] - SA[:, j]).sum()),
            hist_orig=hist(S[:, j], ed), hist_A=hist(SA[:, j], ed), hist_new=hist(new[:, j], ed),
            residual_with=dict(gt1e3=int((r > 1e-3).sum()), gt1e2=int((r > 1e-2).sum()),
                               max=float(r.max()) if len(r) else 0.0))
    stats['time']['total'] = time.time() - t0
    return text_new, stats


def finish(u, text_new, stats):
    """write the record, replace the _nb CSV atomically (temp file in its own folder), mark done"""
    uid = u['uid']
    stats['new_md5'] = md5(text_new)
    stats['out_bytes'] = len(text_new.encode())
    os.makedirs(f'{WORK}/pending', exist_ok=True)
    pend = f'{WORK}/pending/{uid}.json'
    tmpj = f'{pend}.{os.getpid()}'
    json.dump(stats, open(tmpj, 'w'), indent=1)
    os.replace(tmpj, pend)
    dst = u['csv_out']
    assert '_nb/' in dst and os.path.isfile(dst) and not os.path.islink(dst), dst
    tmp = f'{os.path.dirname(dst)}/.{os.path.basename(dst)}.nb2tmp.{JOB}.{os.getpid()}'
    with open(tmp, 'w') as fh:
        fh.write(text_new)
    os.replace(tmp, dst)
    os.replace(pend, f'{WORK}/units/{uid}.json')


def recover(u):
    """a unit interrupted after its rename: the pending record names the md5 now on disk"""
    pend = f'{WORK}/pending/{u["uid"]}.json'
    if not os.path.exists(pend):
        return False
    s = json.load(open(pend))
    if md5(open(u['csv_out']).read()) == s['new_md5']:
        os.replace(pend, f'{WORK}/units/{u["uid"]}.json')
        return True
    os.remove(pend)
    return False


def est(u):
    return 6e-3 * int(u['n']) + 60


def claim(u):
    p = f'{WORK}/claims/{u["uid"]}'
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'{JOB} {RST} {time.time():.0f} {os.uname()[1]}\n'.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                jid, rst, t, _h = open(p).read().split()
            except Exception:
                return False
            stale = (jid == JOB and int(rst) < RST) or (time.time() - float(t) > 3 * est(u) + 900)
            if not stale:
                return False
            try:
                os.rename(p, f'{p}.stale.{JOB}.{RST}.{os.getpid()}')
            except FileNotFoundError:
                return False
    return False


def work(u):
    uid = u['uid']
    try:
        if os.path.exists(f'{WORK}/units/{uid}.json'):
            return uid, 'skip', 0.0
        if time.time() - T0 + est(u) > TLIMIT - 300:
            return uid, 'notime', 0.0
        if not claim(u):
            return uid, 'claimed', 0.0
        if recover(u):
            return uid, 'recovered', 0.0
        t0 = time.time()
        text_new, stats = compute(u)
        finish(u, text_new, stats)
        return uid, 'done', time.time() - t0
    except Exception:
        return uid, 'FAIL ' + traceback.format_exc().replace('\n', ' | '), 0.0


def test(u):
    t0 = time.time()
    text_new, stats = compute(u)
    os.makedirs(f'{WORK}/test', exist_ok=True)
    open(f'{WORK}/test/{u["uid"]}.csv', 'w').write(text_new)
    stats['new_md5'] = md5(text_new)
    json.dump(stats, open(f'{WORK}/test/{u["uid"]}.json', 'w'), indent=1)
    return u['uid'], 'test', time.time() - t0


if __name__ == '__main__':
    import multiprocessing as mp
    rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
    units = [dict(zip(rows[0], r)) for r in rows[1:]]
    os.makedirs(f'{WORK}/claims', exist_ok=True); os.makedirs(f'{WORK}/units', exist_ok=True)
    if MODE == 'test':
        sel = [u for u in units if u['uid'] in set(UIDS)]
        with mp.Pool(min(NPROC, len(sel))) as pool:
            for uid, st, dt in pool.imap_unordered(test, sel):
                print(uid, st, f'{dt:.1f} s', flush=True)
        sys.exit(0)
    # big units (4e5 events) spread through the list, so that few of them are in memory at once
    small = [u for u in units if int(u['n']) <= 80000]
    big = [u for u in units if int(u['n']) > 80000]
    small.sort(key=lambda u: -int(u['n']))
    order = []
    step = max(1, len(small) // max(1, len(big)))
    for k, u in enumerate(small):
        if k % step == 0 and big:
            order.append(big.pop(0))
        order.append(u)
    order += big
    if UIDS:
        order = [u for u in order if u['uid'] in set(UIDS)]
    print(f'worker job {JOB} restart {RST} on {os.uname()[1]}: {len(order)} candidate units, {NPROC} processes, '
          f'time limit {TLIMIT:.0f} s, {time.strftime("%c")}', flush=True)
    cnt = {}
    with mp.Pool(NPROC, maxtasksperchild=20) as pool:
        for uid, st, dt in pool.imap_unordered(work, order):
            k = st.split()[0]
            cnt[k] = cnt.get(k, 0) + 1
            if k in ('done', 'FAIL', 'recovered'):
                print(f'{uid} {st} {dt:.1f} s (elapsed {time.time() - T0:.0f} s; {cnt})', flush=True)
    print(f'WORKER END {JOB}: {cnt}, {time.time() - T0:.0f} s, {time.strftime("%c")}', flush=True)
    sys.exit(1 if cnt.get('FAIL') else 0)
