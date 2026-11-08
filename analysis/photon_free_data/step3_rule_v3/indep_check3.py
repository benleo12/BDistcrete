#!/usr/bin/env python3
"""Independent recomputation of the v3 _nb CSVs on a sample (B0's indep_check.py for rule v3), with
nothing taken from nb_core.py, compute_efps.py or beam_photons*.py:
  - the beam photons by a separate per-event implementation of rule v3 (plain loops; the visible energy,
    missing momentum and |p| from the verifier's own rebuild; rule (ii) only in whole events, with the
    summed transverse momentum of the rest of the hemisphere, each angle with the largest error its
    float32 cos theta allows), compared with the CSV: the rows that
    differ from the original are exactly the events with a v3 beam photon; mult_total = stored - n_beam;
  - B0's own loop implementation of v2 (copied from indep_check.py), to find the events v3 changes;
  - the shapes by the exact thrust (pair enumeration), hemisphere masses and broadening of the Phase A
    verifier (indep_shapes.py), rebuilt with the per-event visible energy;
  - per sampled event: new = stored - (with - without) where |with - stored| <= 1e-2 and the result
    is not negative, else stored.
The two codes differ in their rebuild and thrust algorithm, so the shapes agree to rounding, not bit
for bit; the fallback decisions and the multiplicity exactly.

    python indep_check3.py WORK NUNITS NPER SEED NPROC
Units: every unit v3 rewrote (events_changed > 0), up to NUNITS, then units at random from
WORK/units/*.json; per unit NPER random beam-photon events plus every event whose independent v3 mask
differs from the independent v2 mask, and every event where a shape kept its stored value."""
import os, sys, json, glob, math, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from indep_shapes import shapes, rebuild           # the verifier's independent code (copied, unchanged)

WORK, NU, NPER, SEED, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
OBS = ['1_minus_thrust', 'B_total', 'rho_heavy']
PI32 = float(np.float32(math.pi))


def mask_event_v2(f):
    """B0's independent loop for rule v2 (indep_check.py, unchanged)"""
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


def angle_err(c):
    """the largest polar-angle error of a stored float32 cos theta c (rounded to the nearest float32): its
    rounding interval is c +- half the float32 step (2**-25 below 1 where c reads +-1)"""
    a = abs(float(c))
    h = 2.0 ** -25 if a >= 1.0 else float(np.spacing(np.float32(a))) / 2
    t = math.acos(min(a, 1.0))
    return max(t - math.acos(min(a + h, 1.0)), math.acos(max(a - h, -1.0)) - t)


def mask_event_v3(f, whole=True):
    """rule v3 for one event's kept particles (f: (n, 6) float32), written out with plain loops; whole: the
    file kept every particle of the event (fewer than its 100), which rule (ii) needs"""
    n = len(f)
    z = [float(f[j, 0]) for j in range(n)]
    cs = [float(f[j, 1]) for j in range(n)]
    ph = [float(f[j, 2]) for j in range(n)]
    mn = [f[j, 4] == 0 and f[j, 5] == 0 and abs(float(f[j, 3]) + 3.0) < 1e-4 for j in range(n)]
    base = [mn[j] and abs(abs(ph[j]) - math.pi / 2) < 1e-7 for j in range(n)]
    on_axis_phi = [ph[j] == 0.0 or abs(ph[j]) == PI32 for j in range(n)]
    strict = [mn[j] and on_axis_phi[j] and abs(cs[j]) >= 1 - 1e-7 for j in range(n)]
    if not any(base) and not any(strict):
        return [False] * n
    e, p = rebuild(f, 'balance')                   # the verifier's visible energy (one invisible body)
    s = max(float(e.sum()) / max(sum(z), 1e-300), 45.6)
    E = [z[j] * s for j in range(n)]
    if not any(base):                              # (i)
        return [mn[j] and on_axis_phi[j] and (abs(cs[j]) >= 1 - 1e-7 or 1 - abs(cs[j]) <= 1e-7 + 1.5e-12 / max(E[j], 1e-300))
                for j in range(n)]
    flags = list(base)
    miss = float(np.linalg.norm(p.sum(0)))
    ms = [0.0 if float(f[j, 3]) <= -3.0 + 1e-4 else 10.0 ** float(f[j, 3]) for j in range(n)]
    pmag = [math.sqrt(max(E[j] ** 2 - ms[j] ** 2, 0.0)) for j in range(n)]
    sn = [math.sqrt(max(1.0 - cs[j] ** 2, 0.0)) for j in range(n)]
    u = [(sn[j] * math.cos(ph[j]), sn[j] * math.sin(ph[j]), cs[j]) for j in range(n)]
    for j in range(n):                             # (ii)
        if not (whole and base[j] and abs(cs[j]) >= 1 - 1e-7 and miss < 1e-4 and E[j] >= max(pmag)):
            continue
        pt = 0.0                                   # the others of its hemisphere, transverse to it
        for k in range(n):
            if k != j and cs[k] * cs[j] > 0:
                a, b = u[k], u[j]
                cx, cy, cz = a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
                pt += pmag[k] * (math.sqrt(cx * cx + cy * cy + cz * cz) + angle_err(f[k, 1]))
        if pt < 1e-6 * E[j]:
            flags[j] = False
    J = [j for j in range(n) if flags[j]]
    if len(J) < 2:
        return flags
    c, tol = {}, {}
    for j in J:                                    # (iii)
        c[j] = -cs[j] * (1.0 if ph[j] > 0 else -1.0) * (E[j] + 1e-12) / max(E[j], 1e-300)
        tol[j] = 1e-4 if E[j] < 1e-7 else 1e-5

    def agree(a, b):
        return abs(c[a] - c[b]) <= max(tol[a], tol[b])
    if all(agree(a, b) for a in J for b in J):
        return flags
    best, ref = None, None
    for k in J:
        grp = [j for j in J if agree(j, k)]
        key = (len(grp), max(E[j] for j in grp))
        if best is None or key > best:
            best, ref = key, c[k]
    for j in J:
        if abs(c[j] - ref) > 1e-3:
            flags[j] = False
    return flags


def one(path):
    u = json.load(open(path))
    z = np.load(u['npz']); P, M = z['particles'], z['mask'].astype(bool)
    a = open(u['csv_in']).read().split('\n'); b = open(u['csv_out']).read().split('\n')
    head = a[0].split(','); col = [head.index(o) for o in OBS]; cm = head.index('mult_total')
    ra, rb = a[1:-1], b[1:-1]
    N = len(M)
    rng = np.random.default_rng(SEED + u['rid'])
    flags = [mask_event_v3(P[i][M[i]], M[i].sum() < M.shape[1]) for i in range(N)]
    nb = np.array([sum(x) for x in flags])
    changed = np.array([ra[i] != rb[i] for i in range(N)])
    out = dict(uid=u['uid'], events=N, events_changed_record=u.get('events_changed'), rows_changed=int(changed.sum()),
               beam_events=int((nb > 0).sum()), rows_changed_without_beam=int((changed & (nb == 0)).sum()),
               beam_events_unchanged=int((~changed & (nb > 0)).sum()))
    out['mult_mismatch'] = int(sum(1 for i in range(N) if float(rb[i].split(',')[cm]) != float(ra[i].split(',')[cm]) - nb[i]))
    # the sample: NPER random beam events, every event whose v3 mask differs from v2, every fallback
    idx = np.where(nb > 0)[0]
    samp = set(rng.choice(idx, min(NPER, len(idx)), replace=False).tolist()) if len(idx) else set()
    v3v2 = [i for i in range(N) if flags[i] != mask_event_v2(P[i][M[i]])]
    kept = [i for i in idx if any(ra[i].split(',')[c] == rb[i].split(',')[c] for c in col)]
    samp |= set(v3v2) | set(kept)
    d_all, dec_mismatch, rows = [], 0, []
    for i in sorted(samp):
        fl = np.array(flags[i])
        if not fl.any():                           # a row v3 left without beam photons: the original text
            rows.append(dict(i=int(i), v3_differs_v2=True, reverted_to_original=ra[i] == rb[i], max_abs_diff=0.0))
            dec_mismatch += int(ra[i] != rb[i])
            continue
        f = P[i][M[i]]
        e, p = rebuild(f, 'balance')
        w = shapes(e, p); wo = shapes(e[~fl], p[~fl])
        st = np.array([float(ra[i].split(',')[c]) for c in col])
        nw = np.array([float(rb[i].split(',')[c]) for c in col])
        cand = st - (w - wo)
        fb = ~(np.abs(w - st) <= 1e-2) | ~(cand >= 0)
        mine = np.where(fb, st, cand)
        csv_kept = nw == st
        dec_mismatch += int((fb != csv_kept).sum())
        d_all.append(mine - nw)
        rows.append(dict(i=int(i), v3_differs_v2=bool(i in v3v2), fallback_mine=fb.tolist(), csv_kept=csv_kept.tolist(),
                         with_minus_stored=(w - st).tolist(), max_abs_diff=float(np.abs(mine - nw).max())))
    D = np.abs(np.array(d_all)).reshape(-1, 3)
    out.update(sampled=len(samp), v3_differs_v2=len(v3v2), fallback_in_csv=len(kept), fallback_decision_mismatch=dec_mismatch,
               diff={o: dict(q50=float(np.quantile(D[:, k], .5)), q99=float(np.quantile(D[:, k], .99)),
                             max=float(D[:, k].max()), gt1e6=int((D[:, k] > 1e-6).sum()))
                     for k, o in enumerate(OBS)} if len(D) else {},
               v3_rows=[r for r in rows if r['v3_differs_v2']][:50],
               worst=sorted(rows, key=lambda r: -r['max_abs_diff'])[:3])
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    recs = sorted(glob.glob(f'{WORK}/units/*.json'))
    random.seed(SEED)
    chg = [f for f in recs if json.load(open(f)).get('events_changed', 0) > 0]
    rest = [f for f in recs if f not in set(chg)]
    pick = random.sample(chg, min(NU, len(chg)))
    pick += random.sample(rest, min(NU - len(pick), len(rest)))
    with mp.Pool(max(1, min(NPROC, len(pick)))) as pool:
        res = pool.map(one, pick, chunksize=1)
    tot = dict(units=len(res), units_rewritten_by_v3=sum(1 for r in res if r['events_changed_record']),
               events=sum(r['events'] for r in res), sampled=sum(r['sampled'] for r in res),
               rows_changed_without_beam=sum(r['rows_changed_without_beam'] for r in res),
               beam_events_unchanged=sum(r['beam_events_unchanged'] for r in res),
               mult_mismatch=sum(r['mult_mismatch'] for r in res),
               v3_differs_v2=sum(r['v3_differs_v2'] for r in res),
               v3_differs_v2_record=sum(r['events_changed_record'] or 0 for r in res),
               fallback_in_csv=sum(r['fallback_in_csv'] for r in res),
               fallback_decision_mismatch=sum(r['fallback_decision_mismatch'] for r in res),
               max_abs_diff={o: max((r['diff'][o]['max'] for r in res if r['diff']), default=0.0) for o in OBS},
               q99_max={o: max((r['diff'][o]['q99'] for r in res if r['diff']), default=0.0) for o in OBS},
               events_gt1e6={o: sum(r['diff'][o]['gt1e6'] for r in res if r['diff']) for o in OBS})
    os.makedirs(f'{WORK}/results', exist_ok=True)
    json.dump(dict(total=tot, units=res), open(f'{WORK}/results/indep_check3_{SEED}.json', 'w'), indent=1)
    print(json.dumps(tot, indent=1))
