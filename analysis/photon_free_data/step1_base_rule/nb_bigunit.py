#!/usr/bin/env python3
"""The same unit computation as nb_units.process, with the events of ONE unit spread over NPROC
forked processes (the particle arrays are shared copy-on-write), so a unit takes seconds instead of
minutes and a preemption loses at most the unit in hand. Per-event results are identical to
nb_units.py (same functions, same inputs).

    python nb_bigunit.py WORK NPROC TASKS      TASKS: comma list of task ids of units.tsv, or 'all'
Skips units whose json and csv exist.
"""
import os, sys, time, json
import numpy as np
WORK, NPROC, TASKS = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = [sys.argv[0], WORK, '0', '1', str(NPROC)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import nb_units as NU                                      # noqa: E402
from nb_units import C, MAXEV, N_CONTROL, N_FIXED, EDGES, hist, res_stats, rebuilt_with   # noqa: E402

_G = {}


def _chunk(rng):
    P, M, B, idx = _G['P'], _G['M'], _G['B'], _G['idx']
    a, b = rng
    out = np.empty((b - a, 7)); cl = 0
    for k in range(a, b):
        i = idx[k]
        f = P[i][M[i]]; bb = B[i][M[i]]
        zz, mm, UU = C.parts(f)
        s, c = C.scale_of(zz, mm, UU)
        cl += c
        ev = C.four(zz, mm, UU, s)
        out[k - a, :3] = C.three(ev); out[k - a, 3:6] = C.three(ev[~bb]); out[k - a, 6] = s
    return a, out, cl


def par_loop(P, M, B, idx):
    _G.update(P=P, M=M, B=B, idx=idx)
    n = len(idx)
    edges = np.linspace(0, n, min(NPROC * 4, max(n, 1)) + 1).astype(int)
    rngs = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    W = np.empty((n, 3)); WO = np.empty((n, 3)); sc = np.empty(n); clamp = 0
    ctx = mp.get_context('fork')
    with ctx.Pool(NPROC) as pool:
        for a, out, cl in pool.imap_unordered(_chunk, rngs):
            W[a:a + len(out)] = out[:, :3]; WO[a:a + len(out)] = out[:, 3:6]; sc[a:a + len(out)] = out[:, 6]
            clamp += cl
    _G.clear()
    return W, WO, sc, clamp


def process_par(u):
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
    W, WO, sc, clamp = par_loop(P, M, B, idx)
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




if __name__ == '__main__':
    rows = [l.rstrip('\n').split('\t') for l in open(f'{WORK}/units.tsv')]
    units = [dict(zip(rows[0], r)) for r in rows[1:]]
    if TASKS != 'all':
        keep = set(TASKS.split(','))
        units = [u for u in units if u['task'] in keep]
    units = [u for u in units if not (os.path.exists(f"{WORK}/units/{u['uid']}.json") and os.path.exists(u['csv_out']))]
    units = [u for u in units if int(u['n']) >= int(os.environ.get('NB_MINN', '0'))]
    units.sort(key=lambda u: -int(u['n']))
    t0 = time.time()
    print(f"bigunit on {os.uname()[1]}: {len(units)} units to do, {sum(int(u['n']) for u in units)} events, "
          f"{NPROC} processes per unit, {time.strftime('%c')}", flush=True)
    nf = 0
    for k, u in enumerate(units):
        try:
            uid, st, dt = process_par(u)
        except Exception:
            import traceback
            uid, st, dt = u['uid'], 'FAIL ' + traceback.format_exc().replace('\n', ' | '), 0.0
        nf += st.startswith('FAIL')
        print(f'[{k + 1}/{len(units)}] {uid} {st} {dt:.1f} s (elapsed {time.time() - t0:.0f} s)', flush=True)
    print(f'BIGUNIT END: {len(units)} units, {nf} failed, {time.time() - t0:.0f} s, {time.strftime("%c")}', flush=True)
    sys.exit(1 if nf else 0)
