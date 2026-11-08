#!/usr/bin/env python3
"""Independent check of the photon-free shapes (my own rebuild, beam test, exact thrust, masses,
broadenings; nothing imported from the task's scripts or compute_efps.py).

    python v_shapes.py RUN_DIR NSAMPLE SEED OUT_JSON
RUN_DIR holds particles_full_<rid>.npz, shapes_run_<rid>.csv (original), shapes_run_<rid>_nb.csv (new).
"""
import sys, os, glob, json, time
import numpy as np

ECM = 91.2
OBS = ['1_minus_thrust', 'B_total', 'rho_heavy']
TE = np.array([0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .12, .14, .16, .18, .20, .25, .30, .35, .40, .50])


def read(path):
    lines = open(path).read().split('\n')
    if lines[-1] == '':
        lines = lines[:-1]
    return lines[0].split(','), lines[1:]


def thrust(p):
    """exact thrust of 3-momenta p (n,3): every pair defines a plane; the particles on each side,
    with the four sign choices of the pair, give the candidate axes; then sign iteration to polish."""
    n = len(p)
    mag = np.sqrt((p * p).sum(1)); S = mag.sum()
    if n == 1:
        return p[0] / mag[0], 1.0
    i, j = np.triu_indices(n, 1)
    nrm = np.cross(p[i], p[j])
    sg = np.sign(nrm @ p.T)
    r = np.arange(len(i))
    sg[r, i] = 0.0; sg[r, j] = 0.0
    base = sg @ p
    best, bv = -1.0, None
    for a in (1.0, -1.0):
        for b in (1.0, -1.0):
            v = base + a * p[i] + b * p[j]
            m2 = (v * v).sum(1); k = int(np.argmax(m2))
            if m2[k] > best:
                best, bv = m2[k], v[k]
    ax = bv / np.sqrt(best)
    T = np.abs(p @ ax).sum() / S
    for _ in range(50):                     # polish: a sign iteration can only increase sum |p.n|
        v = (np.sign(p @ ax)[:, None] * p).sum(0); a2 = v / np.linalg.norm(v)
        T2 = np.abs(p @ a2).sum() / S
        if T2 <= T + 1e-15:
            break
        ax, T = a2, T2
    return ax, T


def shapes(E, p):
    ax, T = thrust(p)
    Ev = E.sum(); S = np.sqrt((p * p).sum(1)).sum()
    plus = (p @ ax) >= 0
    rh = []
    for h in (plus, ~plus):
        if h.sum() == 0:
            rh.append(0.0); continue
        e = E[h].sum(); q = p[h].sum(0)
        rh.append(max(e * e - q @ q, 0.0) / Ev ** 2)
    B = np.sqrt((np.cross(p, ax) ** 2).sum(1)).sum() / (2 * S)
    return np.array([1 - T, B, max(rh)])


def rebuild(f, mode):
    f = f.astype(np.float64)
    z, c, ph = f[:, 0], f[:, 1], f[:, 2]
    m = np.where(f[:, 3] <= -3.0, 0.0, 10.0 ** f[:, 3])          # at the 1e-3 clip: massless
    st = np.sqrt(np.clip(1 - c * c, 0, None))
    u = np.stack([st * np.cos(ph), st * np.sin(ph), c], 1)
    def mom(Ev):
        e = z * Ev; return e, np.sqrt(np.clip(e * e - m * m, 0, None))[:, None] * u
    Ev = ECM
    if mode == 'balance':                   # E_vis = ECM - |p_miss| (one invisible massless body)
        for _ in range(40):
            e, p = mom(Ev); En = ECM - np.linalg.norm(p.sum(0))
            if abs(En - Ev) < 1e-12:
                break
            Ev = En
    e, p = mom(Ev)
    return e, p


def main():
    d, nsamp, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rid = os.path.basename(glob.glob(f'{d}/particles_full_*.npz')[0])[15:-4]
    t0 = time.time()
    z = np.load(f'{d}/particles_full_{rid}.npz'); P = z['particles']; M = z['mask'].astype(bool)
    N = len(M)
    ho, ro = read(f'{d}/shapes_run_{rid}.csv'); hn, rn = read(f'{d}/shapes_run_{rid}_nb.csv')
    res = dict(rid=rid, n=N, header_same=ho == hn, nrows=(len(ro), len(rn)))
    col = [ho.index(o) for o in OBS]; other = [k for k in range(len(ho)) if k not in col]
    so = np.array([[float(r.split(',')[k]) for k in col] for r in ro])
    sn = np.array([[float(r.split(',')[k]) for k in col] for r in rn])
    changed = np.array([a != b for a, b in zip(ro, rn)])
    oth_same = all([a.split(',')[k] for k in other] == [b.split(',')[k] for k in other] for a, b in zip(ro, rn))
    # beam photons: strict (mass exactly at the clip, phi exactly float32(+-pi/2)) and the task's tolerance
    neu = M & (P[..., 4] == 0) & (P[..., 5] == 0) & (P[..., 3] == np.float32(-3.0))
    strict = neu & (np.abs(P[..., 2]) == np.float32(np.pi / 2))
    tol = M & (P[..., 4] == 0) & (P[..., 5] == 0) & (np.abs(P[..., 3].astype(np.float64) + 3) < 1e-4) & \
        (np.abs(np.abs(P[..., 2].astype(np.float64)) - np.pi / 2) < 1e-7)
    # axis exactly on the beam: no flagged photon, a massless neutral at |cos| = 1 with phi exactly 0 or pi
    onaxis = neu & (np.abs(P[..., 1]) == 1.0) & ((P[..., 2] == 0) | (np.abs(P[..., 2]) == np.float32(np.pi)))
    ev_tol = tol.any(1); ev_strict = strict.any(1); ev_axis = onaxis.any(1) & ~ev_tol
    mult = np.array([float(r.split(',')[ho.index('mult_total')]) for r in ro])
    nk = M.sum(1)
    res.update(other_columns_identical=bool(oth_same), changed_rows=int(changed.sum()),
               changed_equals_flagged=bool(np.array_equal(changed, ev_tol)),
               changed_not_flagged=int((changed & ~ev_tol).sum()), flagged_not_changed=int((ev_tol & ~changed).sum()),
               unchanged_rows_textual=bool(all(a == b for a, b, c in zip(ro, rn, changed) if not c)),
               ev_flag_tol=int(ev_tol.sum()), ev_flag_strict=int(ev_strict.sum()),
               photons_tol_not_strict=int((tol & ~strict).sum()), ev_axis_on_beam=int(ev_axis.sum()),
               mult_total_eq_kept=bool(np.all(mult[nk < P.shape[1]] == nk[nk < P.shape[1]])),
               n_truncated=int((nk == P.shape[1]).sum()),
               mean_old=so.mean(0).tolist(), mean_new=sn.mean(0).tolist(),
               rel_change_mean=(sn.mean(0) / so.mean(0) - 1).tolist(),
               neg_new=(sn < 0).sum(0).tolist(), nan_new=int(np.isnan(sn).sum()))
    res['thrust_bins_old'] = np.histogram(so[:, 0], TE)[0].tolist()
    res['thrust_bins_new'] = np.histogram(sn[:, 0], TE)[0].tolist()
    # my own recomputation on a random sample of flagged events, plus every axis-on-beam event
    rng = np.random.default_rng(seed)
    idx = np.where(ev_tol)[0]
    samp = np.sort(rng.choice(idx, min(nsamp, len(idx)), replace=False))
    rows = []
    for i in samp:
        f = P[i][M[i]]; b = tol[i][M[i]]
        for mode in ('balance',):
            e, p = rebuild(f, mode)
            w = shapes(e, p); wo = shapes(e[~b], p[~b])
            rows.append(np.r_[i, w, wo])
    A = np.array(rows)
    I = A[:, 0].astype(int); W = A[:, 1:4]; WO = A[:, 4:7]
    st = so[I]; nw = sn[I]
    def q(x):
        a = np.abs(x)
        return dict(q50=float(np.quantile(a, .5)), q90=float(np.quantile(a, .9)), q99=float(np.quantile(a, .99)),
                    max=float(a.max()), gt1e4=float((a > 1e-4).mean()), gt1e3=float((a > 1e-3).mean()),
                    gt1e2=float((a > 1e-2).mean()))
    res['sample'] = len(I)
    res['mine_with_vs_stored'] = {o: q(W[:, k] - st[:, k]) for k, o in enumerate(OBS)}
    res['mine_delta_vs_csv_delta'] = {o: q((W[:, k] - WO[:, k]) - (st[:, k] - nw[:, k])) for k, o in enumerate(OBS)}
    res['mine_without_vs_new'] = {o: q(WO[:, k] - nw[:, k]) for k, o in enumerate(OBS)}
    res['mean_delta_sample'] = {o: float((st[:, k] - nw[:, k]).mean()) for k, o in enumerate(OBS)}
    # the axis-on-beam events: what removal would do
    ax_rows = []
    for i in np.where(ev_axis)[0]:
        f = P[i][M[i]]; b = onaxis[i][M[i]]
        e, p = rebuild(f, 'balance')
        w = shapes(e, p); wo = shapes(e[~b], p[~b])
        ax_rows.append(dict(i=int(i), nflag=int(b.sum()), zbeam=float(f[b, 0].sum()), stored=so[i].tolist(),
                            csv_nb=sn[i].tolist(), with_=w.tolist(), without=wo.tolist(),
                            corrected=(so[i] - (w - wo)).tolist()))
    res['axis_on_beam_events'] = ax_rows
    res['time'] = time.time() - t0
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('axis_on_beam_events', 'thrust_bins_old', 'thrust_bins_new')}, indent=None))


if __name__ == '__main__':
    main()
