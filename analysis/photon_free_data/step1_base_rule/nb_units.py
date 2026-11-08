#!/usr/bin/env python3
"""Recompute 1_minus_thrust, B_total and rho_heavy without the beam photons for the units of one
array task (see nb_core.py for the method). One unit is one real particle file and its CSV.

    python nb_units.py WORK TASK NTASK NPROC [MAXEV]

WORK holds units.tsv (from inventory.py, with a 'task' column added by plan.py) and receives
units/<uid>.json (statistics, written last, so its presence marks the unit done), logs and tmp/.
The new CSV is written to tmp/ and renamed into place. A unit whose json and csv exist is skipped,
so a requeued task resumes. MAXEV > 0 processes only the first MAXEV events and writes nothing
but WORK/test/<uid>.json (timing tests).
"""
import os, sys, time, json, traceback
import numpy as np

WORK, TASK, NTASK, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
MAXEV = int(sys.argv[5]) if len(sys.argv) > 5 else 0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_core as C                                        # noqa: E402

THRUST_EDGES = np.array([0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .12, .14, .16, .18, .20,
                         .25, .30, .35, .40, .50])
RHO_EDGES = np.r_[np.arange(0, 0.1, 0.01), 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.4]
B_EDGES = np.round(np.arange(0, 0.4201, 0.02), 4)
EDGES = {'1_minus_thrust': THRUST_EDGES, 'B_total': B_EDGES, 'rho_heavy': RHO_EDGES}
RES_EDGES = np.arange(-12, 0.01, 0.25)                     # log10 |with - stored|
N_CONTROL, N_FIXED = 200, 300


def hist(v, e):
    h = np.histogram(v, e)[0]
    return [int((v < e[0]).sum())] + h.tolist() + [int((v >= e[-1]).sum())]


def res_stats(r):
    a = np.abs(r)
    out = dict(n=int(len(a)), n_zero=int((a == 0).sum()))
    if len(a):
        out.update(max=float(a.max()), q50=float(np.quantile(a, .5)), q90=float(np.quantile(a, .9)),
                   q99=float(np.quantile(a, .99)), q999=float(np.quantile(a, .999)),
                   gt={f'{t:g}': int((a > t).sum()) for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)},
                   loghist=np.histogram(np.log10(a[a > 0]), RES_EDGES)[0].tolist(),
                   log_under=int(((a > 0) & (a < 10 ** RES_EDGES[0])).sum()),
                   log_over=int((a >= 10 ** RES_EDGES[-1]).sum()))
    return out


def rebuilt_with(f):
    zz, mm, UU = C.parts(f)
    return C.three(C.four(zz, mm, UU, C.scale_of(zz, mm, UU)[0]))


def process(u):
    uid = u['uid']
    done = f'{WORK}/units/{uid}.json'
    if not MAXEV and os.path.exists(done) and os.path.exists(u['csv_out']):
        return uid, 'skip', 0.0
    t0 = time.time()
    z = np.load(u['npz'])
    P, M = z['particles'], z['mask'].astype(bool)
    N, PMAX = M.shape
    text = open(u['csv_in']).read()
    lines = text.split('\n')
    trail = lines[-1] == ''
    body = lines[1:-1] if trail else lines[1:]
    head = lines[0].split(',')
    if len(body) != N:
        raise RuntimeError(f'{uid}: csv {len(body)} rows, npz {N} events')
    col = [head.index(o) for o in C.OBS]
    if MAXEV:
        P, M, body, N = P[:MAXEV], M[:MAXEV], body[:MAXEV], min(N, MAXEV)
    t_load = time.time() - t0
    B = C.beam_mask(P, M)
    nb = B.sum(1)
    idx = np.where(nb > 0)[0]
    S = np.empty((N, 3))
    for i in range(N):
        q = body[i].split(',')
        S[i] = [float(q[c]) for c in col]
    W = np.empty((len(idx), 3)); WO = np.empty((len(idx), 3)); sc = np.empty(len(idx)); clamp = 0
    for k, i in enumerate(idx):
        f = P[i][M[i]]; b = B[i][M[i]]
        zz, mm, UU = C.parts(f)
        s, cl = C.scale_of(zz, mm, UU)
        clamp += cl
        ev = C.four(zz, mm, UU, s)
        W[k] = C.three(ev); WO[k] = C.three(ev[~b]); sc[k] = s
    t_main = time.time() - t0 - t_load
    stored = S[idx]
    newv = S.copy()
    newv[idx] = stored - (W - WO)
    # control: events without a beam photon, the rebuild alone against the stored values
    ctl = np.where(nb == 0)[0][:N_CONTROL]
    WC = np.array([rebuilt_with(P[i][M[i]]) for i in ctl]).reshape(-1, 3)
    # the same delta with the fixed event energy 91.2 GeV, on the first N_FIXED beam events
    dfix = []
    for k, i in enumerate(idx[:N_FIXED]):
        f = P[i][M[i]]; b = B[i][M[i]]
        zz, mm, UU = C.parts(f)
        ev = C.four(zz, mm, UU, C.ECM)
        dfix.append(np.array(C.three(ev)) - np.array(C.three(ev[~b])) - (W[k] - WO[k]))
    dfix = np.array(dfix).reshape(-1, 3)
    corner = C.corner_counts(P, M, B)
    zb = np.where(B, P[..., 0].astype(np.float64), 0.0).sum(1)
    stats = dict(uid=uid, group=u['group'], rid=int(u['rid']), gen=u['gen'], npz=u['npz'], csv_in=u['csv_in'],
                 csv_out=u['csv_out'], n_events=int(N), pmax=int(PMAX), maxev=MAXEV,
                 n_beam_events=int(len(idx)), nflag_hist=np.bincount(np.minimum(nb, 4), minlength=5).tolist(),
                 sum_zbeam=float(zb.sum()), n_ebeam_gt1=int((zb * C.ECM > 1).sum()),
                 n_ebeam_gt5=int((zb * C.ECM > 5).sum()), n_truncated=int((M.sum(1) == PMAX).sum()),
                 corner=corner, scale=dict(lt91_1=int((sc < 91.1).sum()), lt88=int((sc < 88).sum()),
                                           lt80=int((sc < 80).sum()), clamped=int(clamp),
                                           mean=float(sc.mean()) if len(sc) else None),
                 obs={}, time=dict(load=t_load, main=t_main))
    for j, o in enumerate(C.OBS):
        e = EDGES[o]
        stats['obs'][o] = dict(
            sum_stored=float(S[:, j].sum()), sum_new=float(newv[:, j].sum()),
            sumsq_stored=float((S[:, j] ** 2).sum()), sumsq_new=float((newv[:, j] ** 2).sum()),
            n_nan_stored=int(np.isnan(S[:, j]).sum()), n_neg_new=int((newv[:, j] < 0).sum()),
            min_new=float(newv[:, j].min()),
            hist_stored=hist(S[:, j], e), hist_new=hist(newv[:, j], e),
            residual_with=res_stats(W[:, j] - stored[:, j]),
            residual_control=res_stats(WC[:, j] - S[ctl, j]) if len(ctl) else None,
            delta=dict(mean=float((W[:, j] - WO[:, j]).mean()) if len(idx) else 0.0,
                       max=float((W[:, j] - WO[:, j]).max()) if len(idx) else 0.0,
                       min=float((W[:, j] - WO[:, j]).min()) if len(idx) else 0.0),
            fixed_vs_balance=res_stats(dfix[:, j]))
    if MAXEV:
        os.makedirs(f'{WORK}/test', exist_ok=True)
        stats['time']['total'] = time.time() - t0
        json.dump(stats, open(f'{WORK}/test/{uid}.json', 'w'), indent=1)
        return uid, 'test', time.time() - t0
    for k, i in enumerate(idx):
        q = body[i].split(',')
        for j, c in enumerate(col):
            q[c] = repr(float(newv[i, j]))
        body[i] = ','.join(q)
    out = '\n'.join([lines[0]] + body) + ('\n' if trail else '')
    os.makedirs(os.path.dirname(u['csv_out']), exist_ok=True)
    tmp = f'{WORK}/tmp/{uid}.{os.getpid()}.csv'
    with open(tmp, 'w') as fh:
        fh.write(out)
    os.replace(tmp, u['csv_out'])
    stats['time']['total'] = time.time() - t0
    stats['out_bytes'] = os.path.getsize(u['csv_out'])
    tj = f'{WORK}/tmp/{uid}.{os.getpid()}.json'
    json.dump(stats, open(tj, 'w'), indent=1)
    os.replace(tj, done)
    return uid, 'done', time.time() - t0


def safe(u):
    try:
        return process(u)
    except Exception:
        return u['uid'], 'FAIL ' + traceback.format_exc().replace('\n', ' | '), 0.0


if __name__ == '__main__':
    import multiprocessing as mp
    rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
    head = rows[0]
    units = [dict(zip(head, r)) for r in rows[1:]]
    mine = [u for u in units if int(u['task']) == TASK]
    if MAXEV:
        mine = mine[:NPROC]
    mine.sort(key=lambda u: -int(u['n']))
    t0 = time.time()
    print(f'task {TASK}/{NTASK} on {os.uname()[1]}: {len(mine)} units, {sum(int(u["n"]) for u in mine)} events, '
          f'{NPROC} processes, {time.strftime("%c")}', flush=True)
    nfail = 0
    with mp.Pool(NPROC) as pool:
        for k, (uid, st, dt) in enumerate(pool.imap_unordered(safe, mine)):
            nfail += st.startswith('FAIL')
            print(f'[{k + 1}/{len(mine)}] {uid} {st} {dt:.1f} s (elapsed {time.time() - t0:.0f} s)', flush=True)
    print(f'TASK {TASK} END: {len(mine)} units, {nfail} failed, {time.time() - t0:.0f} s, {time.strftime("%c")}', flush=True)
    sys.exit(1 if nfail else 0)
