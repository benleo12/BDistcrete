#!/usr/bin/env python3
"""The maximum-entropy tilt, actually run on the sample (Sec. maxent of the paper).

CORRECTED VERSION of maxent_tilt.py. The only change is on the TARGET side: the event
shapes of the MEPS@NLO files (events_nom / events_shup / events_shdn) are now computed with
the project's main extractor, compute_efps.compute_hemisphere_shapes (the same code path
compute_shapes_only.py wraps and the same code that produced output/models/B_ref.npz), instead
of the private reimplementation that used to live in this file. The old private thrust routine
seeded the axis from the six hardest particles plus the momentum sum and took the best,
whereas compute_efps seeds from the hardest particle only; the two disagreed on ~3.9% of
events, so part of the fitted tilt was repairing an algorithmic offset between the two sides
of the constraint equation rather than physics. Both sides now use ONE implementation.

Everything else is identical to maxent_tilt.py: same Stage B cache and box centre, same
classifier weights w0, same per-event MC weights in the target average, same Newton solve on
the convex dual with the same 1e-10 tolerance, same spectator observables.

Writes output/maxent_tilt_fixed.json (maxent_tilt.json is left untouched)."""
import os, json, numpy as np
from ab_analysis import Cond
from compute_efps import compute_hemisphere_shapes, parse_hepmc3

NU = {12, 14, 16}
EVDIR = os.environ['CSS_EVENTS_DIR']
CONSTRAINTS = ['1_minus_thrust', 'B_total', 'rho_heavy']
SPECTATORS = ['mult_total', 'nbaryon']


def sample_moments(path, ret_raw=False):
    """Weighted moments of CONSTRAINTS over a HepMC file.

    Particle selection and the per-event MC weight are read exactly as before; the SHAPES
    come from compute_efps.compute_hemisphere_shapes, i.e. the project's main extractor."""
    ws = []; V = []
    w = None; rows = []

    def flush():
        if w is not None and len(rows) >= 2:
            d = compute_hemisphere_shapes(np.asarray(rows, dtype=np.float64))
            ws.append(w); V.append([d[o] for o in CONSTRAINTS])
    for line in open(path):
        t = line[0]
        if t == 'E':
            flush(); w = None; rows = []
        elif t == 'W':
            f = line.split()
            if '\\' in line or f[1][0].isalpha(): continue
            w = float(f[1])
        elif t == 'P':
            f = line.split()
            if int(f[-1]) == 1 and abs(int(f[3])) not in NU:
                # compute_efps event layout is [E, px, py, pz]
                rows.append([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
    flush()
    ws = np.array(ws); V = np.array(V); W = ws.sum()
    m = (ws[:, None]*V).sum(0)/W
    neff = W*W/np.sum(ws**2)
    err = np.sqrt(((ws[:, None]*(V-m)**2).sum(0)/W)/neff)
    if ret_raw:
        return m, err, ws, V
    return m, err


def audit(path):
    """Independent check that this file's parse+shape route reproduces, event by event, what
    compute_shapes_only.py writes when it is pointed at the same HepMC file."""
    import pandas as pd, os
    csv = 'output/shapes_defectfix/shapes_run_0000.csv'
    if not os.path.exists(csv):
        return
    df = pd.read_csv(csv)
    _, _, ws, V = sample_moments(path, ret_raw=True)
    ref = df[CONSTRAINTS].to_numpy()
    ok = ref.shape[0] == V.shape[0] and np.allclose(ref, V, rtol=0, atol=1e-12)
    print(f'audit vs compute_shapes_only CSV: {ref.shape[0]} vs {V.shape[0]} events, '
          f'identical = {ok}')
    print(f'audit unweighted mean (CSV) {np.round(ref.mean(0), 6).tolist()}  '
          f'weighted mean {np.round((ws[:, None]*V).sum(0)/ws.sum(), 6).tolist()}')


def main():
    # -------- target moments c_i and shower-scale bands (project extractor) --------
    c, stat = sample_moments(f'{EVDIR}/events_nom')
    dn, _ = sample_moments(f'{EVDIR}/events_shdn')
    up, _ = sample_moments(f'{EVDIR}/events_shup')
    band = 0.5*np.abs(up-dn)
    print('MEPS@NLO target moments (obs: c +- stat +- shower-band):')
    for i, o in enumerate(CONSTRAINTS):
        print(f'  <{o:>14s}> = {c[i]:.5f} +- {stat[i]:.5f} +- {band[i]:.5f}')
    audit(f'{EVDIR}/events_nom')

    # -------- sample side: reference reweighted to the box centre --------
    cond = Cond('B'); ref = np.load('output/models/B_ref.npz')
    th0 = cond.norm[:, 0].copy()
    f0 = cond.f_at(th0)
    w0 = np.exp((f0 - f0.max())/cond.T); w0 /= w0.sum()
    M = np.stack([ref[o].astype(float) for o in CONSTRAINTS])          # (3, Nref)
    S = {o: ref[o].astype(float) for o in SPECTATORS if o in ref}
    m0 = (w0*M).sum(1)
    print(f'sample moments at nu0 (before tilt): {np.round(m0, 5).tolist()}')
    print(f'target                             : {np.round(c, 5).tolist()}')

    # -------- Newton solve on the convex dual --------
    lam = np.zeros(len(CONSTRAINTS))
    for it in range(60):
        e = np.exp(lam @ M - (lam @ M).max())
        wt = w0*e; wt /= wt.sum()
        g = (wt*M).sum(1) - c                                          # gradient
        Mc = M - (wt*M).sum(1, keepdims=True)
        H = (wt*Mc) @ Mc.T                                             # Hessian = Cov
        step = np.linalg.solve(H, g)
        lam -= step
        if np.max(np.abs(g)) < 1e-10: break
    e = np.exp(lam @ M - (lam @ M).max()); wt = w0*e; wt /= wt.sum()
    mt = (wt*M).sum(1)
    print(f'Newton converged in {it+1} iters, lambda = {np.round(lam, 3).tolist()}')
    print(f'moments after tilt                 : {np.round(mt, 6).tolist()}')
    print(f'max |constraint violation|         : {np.max(np.abs(mt-c)):.2e}')

    # -------- verification: information added and distortion --------
    kl = float(np.sum(wt*np.log(np.clip(wt/np.clip(w0, 1e-300, None), 1e-300, None))))
    neff0 = 1.0/np.sum(w0**2); nefft = 1.0/np.sum(wt**2)
    print(f'KL(tilt||classifier-only) = {kl:.5f} nats   N_eff {neff0:.0f} -> {nefft:.0f}')
    dist = {}
    for o, v in S.items():
        mu0 = (w0*v).sum(); mut = (wt*v).sum()
        sd = np.sqrt((w0*(v-mu0)**2).sum())
        dist[o] = dict(before=float(mu0), after=float(mut), shift_in_sd=float((mut-mu0)/sd))
        print(f'  spectator <{o}>: {mu0:.4f} -> {mut:.4f}  ({dist[o]["shift_in_sd"]:+.3f} sd)')

    json.dump(dict(constraints=CONSTRAINTS, c=c.tolist(), stat=stat.tolist(), band=band.tolist(),
                   m_before=m0.tolist(), m_after=mt.tolist(), lam=lam.tolist(),
                   n_newton=int(it+1),
                   max_violation=float(np.max(np.abs(mt-c))), kl=kl,
                   neff_before=float(neff0), neff_after=float(nefft), spectators=dist,
                   shape_implementation='compute_efps.compute_hemisphere_shapes'),
              open('output/maxent_tilt_fixed.json', 'w'), indent=1)
    print('MAXENT TILT (FIXED) DONE')


if __name__ == '__main__':
    main()
