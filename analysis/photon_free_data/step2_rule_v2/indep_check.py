#!/usr/bin/env python3
"""Independent recomputation of the rewritten _nb CSVs on a sample, with nothing taken from
nb_core.py, compute_efps.py or beam_photons.py:
  - the beam photons by a separate per-event implementation of the same rule (plain loops),
    compared with the flags behind the CSV (rows changed = events with a beam photon);
  - the shapes by the exact thrust (pair enumeration), hemisphere masses and broadening of the
    Phase A verifier (lmxA/verify_observables/scripts/v_shapes.py: thrust, shapes, rebuild),
    rebuilt with the per-event visible energy;
  - per sampled event: new = stored - (with - without) where |with - stored| <= 1e-2 and the result
    is not negative, else stored; mult_total = stored - n_beam.
The two codes differ in their rebuild (iterations of the energy balance) and thrust algorithm, so
the shapes agree to rounding, not bit for bit; the fallback decisions and the multiplicity exactly.

    python indep_check.py WORK NUNITS NPER SEED NPROC
Units drawn at random from WORK/units/*.json; NPER beam-photon events per unit plus every event of
the unit where the mask differs from Phase A's rule or a shape kept its stored value."""
import os, sys, json, glob, math, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from indep_shapes import shapes, rebuild           # the verifier's independent code (copied, unchanged)

WORK, NU, NPER, SEED, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
OBS = ['1_minus_thrust', 'B_total', 'rho_heavy']
PI32 = float(np.float32(math.pi))


def mask_event(f):
    """flags of one event's kept particles (f: (n, 6) float32), written out without numpy masks"""
    n = len(f)
    E = [float(f[j, 0]) * 91.2 for j in range(n)]
    mn = [f[j, 4] == 0 and f[j, 5] == 0 and abs(float(f[j, 3]) + 3.0) < 1e-4 for j in range(n)]
    base = [mn[j] and abs(abs(float(f[j, 2])) - math.pi / 2) < 1e-7 for j in range(n)]
    if not any(base):
        return [mn[j] and abs(float(f[j, 1])) >= 1 - 1e-7 and (float(f[j, 2]) == 0.0 or abs(float(f[j, 2])) == PI32)
                for j in range(n)]
    flags = list(base)
    hard = [j for j in range(n) if base[j] and E[j] > 1e-3]
    if len(hard) < 2:
        return flags
    c = {}
    for j in range(n):
        if base[j]:
            s = 1.0 if float(f[j, 2]) > 0 else -1.0
            c[j] = -float(f[j, 1]) * s * (E[j] + 1e-12) / max(E[j], 1e-300)
    if max(c[j] for j in hard) - min(c[j] for j in hard) <= 1e-5:
        return flags
    soft = [j for j in c if j not in hard]
    best, ref = None, None
    for k in hard:
        nh = sum(1 for j in hard if abs(c[j] - c[k]) <= 1e-5)
        ns = sum(1 for j in soft if abs(c[j] - c[k]) <= 1e-5 + 0.3 * 1e-12 / max(E[j], 1e-300))
        emax = max(E[j] for j in hard if abs(c[j] - c[k]) <= 1e-5)
        key = (nh, ns, emax)
        if best is None or key > best:
            best, ref = key, c[k]
    for j in hard:
        if abs(c[j] - ref) > 1e-5:
            flags[j] = False
    return flags


def phaseA_mask(f):
    return [f[j, 4] == 0 and f[j, 5] == 0 and abs(float(f[j, 3]) + 3.0) < 1e-4 and
            abs(abs(float(f[j, 2])) - math.pi / 2) < 1e-7 for j in range(len(f))]


def one(path):
    u = json.load(open(path))
    z = np.load(u['npz']); P, M = z['particles'], z['mask'].astype(bool)
    a = open(u['csv_in']).read().split('\n'); b = open(u['csv_out']).read().split('\n')
    head = a[0].split(','); col = [head.index(o) for o in OBS]; cm = head.index('mult_total')
    ra, rb = a[1:-1], b[1:-1]
    N = len(M)
    rng = np.random.default_rng(SEED + u['rid'])
    # the mask on every event of the unit, and which rows the CSV changed
    flags = [mask_event(P[i][M[i]]) for i in range(N)]
    nb = np.array([sum(x) for x in flags])
    changed = np.array([ra[i] != rb[i] for i in range(N)])
    out = dict(uid=u['uid'], events=N, rows_changed=int(changed.sum()), beam_events=int((nb > 0).sum()),
               rows_changed_without_beam=int((changed & (nb == 0)).sum()),
               beam_events_unchanged=int((~changed & (nb > 0)).sum()))
    # the mult column: stored - n_beam in every row
    mult_bad = sum(1 for i in range(N) if float(rb[i].split(',')[cm]) != float(ra[i].split(',')[cm]) - nb[i])
    out['mult_mismatch'] = int(mult_bad)
    # the sample: NPER random beam events, every event whose mask differs from Phase A's rule, every
    # event where a shape kept its stored text
    idx = np.where(nb > 0)[0]
    samp = set(rng.choice(idx, min(NPER, len(idx)), replace=False).tolist())
    diffA = [i for i in idx if flags[i] != phaseA_mask(P[i][M[i]])]
    kept = [i for i in idx if any(ra[i].split(',')[c] == rb[i].split(',')[c] for c in col)]
    samp |= set(diffA) | set(kept)
    d_all, dec_mismatch, rows = [], 0, []
    for i in sorted(samp):
        f = P[i][M[i]]; fl = np.array(flags[i])
        e, p = rebuild(f, 'balance')
        w = shapes(e, p); wo = shapes(e[~fl], p[~fl])
        st = np.array([float(ra[i].split(',')[c]) for c in col])
        nw = np.array([float(rb[i].split(',')[c]) for c in col])
        cand = st - (w - wo)
        fb = ~(np.abs(w - st) <= 1e-2) | ~(cand >= 0)
        mine = np.where(fb, st, cand)
        csv_kept = nw == st
        dis = fb != csv_kept
        dec_mismatch += int(dis.sum())
        d_all.append(mine - nw)
        rows.append(dict(i=int(i), diffA=bool(i in diffA), fallback_mine=fb.tolist(), csv_kept=csv_kept.tolist(),
                         with_minus_stored=(w - st).tolist(), max_abs_diff=float(np.abs(mine - nw).max())))
    D = np.abs(np.array(d_all)).reshape(-1, 3)
    out.update(sampled=len(samp), mask_differs_from_phaseA=len(diffA), fallback_in_csv=len(kept),
               fallback_decision_mismatch=dec_mismatch,
               diff={o: dict(q50=float(np.quantile(D[:, k], .5)), q99=float(np.quantile(D[:, k], .99)),
                             max=float(D[:, k].max()), gt1e6=int((D[:, k] > 1e-6).sum()))
                     for k, o in enumerate(OBS)} if len(D) else {},
               worst=sorted(rows, key=lambda r: -r['max_abs_diff'])[:3])
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    recs = sorted(glob.glob(f'{WORK}/units/*.json'))
    random.seed(SEED)
    pick = random.sample(recs, min(NU, len(recs)))
    with mp.Pool(NPROC) as pool:
        res = pool.map(one, pick, chunksize=1)
    tot = dict(units=len(res), events=sum(r['events'] for r in res), sampled=sum(r['sampled'] for r in res),
               rows_changed_without_beam=sum(r['rows_changed_without_beam'] for r in res),
               beam_events_unchanged=sum(r['beam_events_unchanged'] for r in res),
               mult_mismatch=sum(r['mult_mismatch'] for r in res),
               mask_differs_from_phaseA=sum(r['mask_differs_from_phaseA'] for r in res),
               fallback_in_csv=sum(r['fallback_in_csv'] for r in res),
               fallback_decision_mismatch=sum(r['fallback_decision_mismatch'] for r in res),
               max_abs_diff={o: max(r['diff'][o]['max'] for r in res if r['diff']) for o in OBS},
               q99_max={o: max(r['diff'][o]['q99'] for r in res if r['diff']) for o in OBS},
               events_gt1e6={o: sum(r['diff'][o]['gt1e6'] for r in res if r['diff']) for o in OBS})
    json.dump(dict(total=tot, units=res), open(f'{WORK}/results/indep_check_{SEED}.json', 'w'), indent=1)
    print(json.dumps(tot, indent=1))
