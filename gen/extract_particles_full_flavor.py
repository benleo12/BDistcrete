#!/usr/bin/env python3
"""Full-event, flavor-aware particle extractor in the thrust frame (both hemispheres).

Combines extract_particles_full.py (FULL event, both hemispheres, signed cos_theta) with
extract_particles_flavor.py (per-particle flavor features from the HepMC particle id/mass).
Keeping ALL final-state particles makes the whole event structure visible (the signed
cos(theta) carries the hemisphere), while the flavor features let the network reweight
flavor-changing hadronization knobs (e.g. BARYON_FRACTION) that leave kinematics unchanged.

Output npz:
    particles : [N, P_max, 6] float32  per-particle features, columns in order
        z             = E_i / E_total          energy fraction of the whole event
        cos_theta     = p_hat_i . n_thrust      signed, in [-1, 1] (both hemispheres)
        phi           = azimuth around n_thrust
        log10_mass    = log10(max(mass, 1e-3))  from the HepMC P-line
        baryon_number = +/-1 for (anti)baryons, else 0
        charge        = +/-1 for |q|=1 species, else 0
    mask  : [N, P_max] bool
    counts: [N] int32
    nbaryon : [N] int32  number of baryons in the FULL event (baryon-sensitive closure observable)
"""
import argparse, gzip, numpy as np
from pathlib import Path

NEUTRINO = {12, 14, 16}
CHG1 = {211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13, 15, 411, 431, 521}  # |charge|=1 species


def charge(pid):
    a = abs(pid)
    return (1 if pid > 0 else -1) if a in CHG1 else 0      # neutrals (gamma,pi0,K0,n,Lambda,...) -> 0


def baryonnum(pid):
    a = abs(pid)
    return (1 if pid > 0 else -1) if 1000 <= a < 1000000 else 0


def thrust_axis(p3):
    if len(p3) == 0:
        return np.array([0, 0, 1.0])
    mags = np.linalg.norm(p3, axis=1)
    n = p3[np.argmax(mags)] / (mags[np.argmax(mags)] + 1e-12)
    for _ in range(20):
        signs = np.sign(p3 @ n)
        n_new = (signs[:, None] * p3).sum(0)
        nn = np.linalg.norm(n_new)
        if nn < 1e-12:
            break
        n_new = n_new / nn
        if np.dot(n_new, n) > 1 - 1e-12:
            break
        n = n_new
    return n


def _hepmc_flavour(filepath, opener):
    """'hepmc2' or 'hepmc3', read off the file's own header, or None if it does not say."""
    with opener(filepath, 'rt') as fh:
        for _ in range(40):
            line = fh.readline()
            if not line:
                break
            if 'Asciiv3' in line or 'HepMC::Version 3' in line:
                return 'hepmc3'
            if 'IO_GenEvent' in line or 'HepMC::Version 2' in line:
                return 'hepmc2'
            if line.startswith('E '):
                break
    return None


def parse_hepmc(filepath, max_events=None, fmt='auto'):
    """Parse a HepMC file, keeping final-state non-neutrino particles.
    Each row is [E, px, py, pz, pid, mass].

    fmt='auto' reads HepMC2 against HepMC3 from the file header. It used to choose by counting
    fields on the P line, taking the HepMC3 layout whenever there were ten or more, and a HepMC2
    P line carries twelve or thirteen, so an auto-detected HepMC2 file had its momentum read as
    the particle id, matched no status and produced nothing. Production never hit it because the
    Herwig runs pass fmt='herwig' and Sherpa writes HepMC3, but a user regenerating would.
    'herwig' is kept as an alias for 'hepmc2', which is what it always meant."""
    opener = gzip.open if str(filepath).endswith('.gz') else open
    if fmt in ('auto', None):
        fmt = _hepmc_flavour(filepath, opener)
        if fmt is None:
            raise ValueError(f'{filepath}: the header does not say whether this is HepMC2 or '
                             f'HepMC3. Pass --fmt hepmc2 or --fmt hepmc3.')
    if fmt == 'herwig':
        fmt = 'hepmc2'
    if fmt not in ('hepmc2', 'hepmc3'):
        raise ValueError(f'fmt={fmt!r} is not one of auto, hepmc2, hepmc3, herwig')
    # HepMC2: P barcode pid px py pz E m status ...   HepMC3: P id vertex pid px py pz E m status
    i_pid, i_stat, i_mom = (2, 8, 3) if fmt == 'hepmc2' else (3, 9, 4)
    events, cur, seen = [], [], 0
    with opener(filepath, 'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if cur:
                    events.append(np.asarray(cur, dtype=np.float64)); seen += len(cur); cur = []
                    if max_events is not None and len(events) >= max_events:
                        break
                continue
            if not line.startswith('P '):
                continue
            parts = line.split()
            if len(parts) <= i_stat:
                continue
            try:
                pid = int(parts[i_pid]); status = int(parts[i_stat])
                px, py, pz, E, m = (float(parts[i_mom]), float(parts[i_mom + 1]),
                                    float(parts[i_mom + 2]), float(parts[i_mom + 3]),
                                    float(parts[i_mom + 4]))
                if status == 1 and abs(pid) not in NEUTRINO:
                    cur.append([E, px, py, pz, pid, m])
            except (ValueError, IndexError):
                continue
    if cur:
        events.append(np.asarray(cur, dtype=np.float64)); seen += len(cur)
    if not seen:
        raise ValueError(f'{filepath}: parsed as {fmt} and found no final-state particles. '
                         f'Either it holds no events or the layout is not what {fmt} implies.')
    return events


def exact_thrust_tau(p3):
    """1 - T computed exactly for a small set of momenta.

    Thrust is max over unit vectors of sum|p.n| / sum|p|, and for n particles the maximizing
    axis is parallel to sum_i s_i p_i for some sign assignment s, so the exact value is the
    maximum of |sum_i s_i p_i| / sum|p_i| over all sign assignments (equal to
    2|sum_{i in S} p_i| / sum|p_i| over subsets S when the momenta sum to zero, which the
    parton set does only up to photon radiation, so the sign form is used). For the five to
    eight partons of a shower final state that is a few hundred assignments, enumerated with the
    first sign fixed to halve the count. Above 16 particles the iterative finder is used."""
    n = len(p3)
    if n == 0:
        return float('nan')
    if n == 1:
        return 0.0
    if n > 16:
        ax = thrust_axis(p3)
        return 1.0 - np.abs(p3 @ ax).sum() / np.linalg.norm(p3, axis=1).sum()
    m = n - 1
    codes = np.arange(1 << m)[:, None] >> np.arange(m)[None, :] & 1          # (2^m, m) bits
    signs = np.hstack([np.ones((1 << m, 1)), 2.0*codes - 1.0])              # first sign fixed
    best = np.linalg.norm(signs @ p3, axis=1).max()
    return 1.0 - best / np.linalg.norm(p3, axis=1).sum()


def parton_level_tau(filepath, fmt='auto', max_events=None):
    """Per-event thrust of the END-OF-SHOWER partons, one number per event, plus their count.

    Needed to anchor the sample at parton level (the calculation supplies the perturbative shape,
    the generator's hadronization does the transfer), which is impossible once the HepMC record
    is gone, so it is stored at extraction time next to the hadron-level arrays.

    The parton set is defined by the vertex topology of each generator's record, verified on
    2026-09-26 (Sherpa 3.1) and 2026-09-28 (Herwig 7.3.0):
      Sherpa, HepMC3: shower partons carry status 11 and vertex status 4 is a shower vertex, 5 a
        fragmentation vertex. A parton is end-of-shower if it is a status-11 quark or gluon whose
        end vertex is not a shower vertex (in practice it is a fragmentation vertex).
      Herwig, HepMC2: every shower parton is status 11, clusters are PDG 81, and vertex statuses
        are all 0. A parton is end-of-shower if its end vertex produces a cluster directly. Herwig
        then splits gluons nonperturbatively into a quark pair just before clustering, and those
        splittings are merged back into the gluon so that the set is the shower's own final state
        (thrust barely sees the difference, 0.0009 in the mean, but the count does).
    Returns (tau, nparton) as float32 and int16 arrays; events with no partons get NaN and 0."""
    opener = gzip.open if str(filepath).endswith('.gz') else open
    if fmt in ('auto', None):
        fmt = _hepmc_flavour(filepath, opener)
    if fmt == 'herwig':
        fmt = 'hepmc2'
    QG = lambda pid: abs(pid) <= 6 or pid == 21
    taus, ns = [], []

    def finish(P, V):
        if fmt == 'hepmc3':
            # Sherpa 3.0.4 writes a pass-through status-5 vertex whose outgoing partons repeat
            # the shower's final partons with identical momenta before the real fragmentation
            # vertex, so "not a shower vertex" alone counts every parton twice (11.5 per event
            # instead of 7.5, with tau unchanged since duplicated momenta leave the thrust
            # invariant). The end-of-shower parton is the one whose end vertex produces
            # something that is not a parton.
            endv = {}
            for vid, v in V.items():
                for i in v['in']:
                    endv[i] = vid
            outs = {}
            for i, q in P.items():
                outs.setdefault(q['vtx'], []).append(i)
            def hadronizes(vid):
                return vid is not None and any(not QG(P[j]['pid']) for j in outs.get(vid, []))
            sel = [i for i, q in P.items() if q['st'] == 11 and QG(q['pid'])
                   and V.get(endv.get(i), {}).get('st') != 4 and hadronizes(endv.get(i))]
        else:
            def outs(v):
                return V[v]['out'] if v in V else []
            A = set(i for i, q in P.items() if QG(q['pid']) and q['end'] != 0
                    and any(P[j]['pid'] == 81 for j in outs(q['end'])))
            B = set(A)
            for i, q in P.items():
                if q['pid'] == 21 and q['end'] != 0:
                    o = outs(q['end'])
                    if len(o) == 2 and all(abs(P[j]['pid']) <= 6 and j in A for j in o):
                        B -= set(o); B.add(i)
            sel = list(B)
        if not sel:
            taus.append(np.nan); ns.append(0); return
        p3 = np.array([P[i]['p'] for i in sel])
        taus.append(exact_thrust_tau(p3))
        ns.append(len(sel))

    P, V, curv, started = {}, {}, None, False
    with opener(filepath, 'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if started:
                    finish(P, V)
                    if max_events is not None and len(taus) >= max_events:
                        started = False; break
                P, V, curv, started = {}, {}, None, True
            elif not started:
                continue
            elif fmt == 'hepmc3' and line.startswith('V '):
                head, _, rest = line.partition('[')
                h = head.split()
                V[int(h[1])] = dict(st=int(h[2]), in_=None,
                                    **{'in': [int(x) for x in rest.split(']')[0].split(',') if x.strip()]})
            elif fmt == 'hepmc3' and line.startswith('P '):
                q = line.split()
                P[int(q[1])] = dict(pid=int(q[3]), p=np.array([float(q[4]), float(q[5]), float(q[6])]),
                                    st=int(q[9]), vtx=int(q[2]))
            elif fmt == 'hepmc2' and line.startswith('V '):
                q = line.split(); curv = int(q[1]); V[curv] = dict(out=[])
            elif fmt == 'hepmc2' and line.startswith('P '):
                q = line.split(); i = int(q[1])
                P[i] = dict(pid=int(q[2]), p=np.array([float(q[3]), float(q[4]), float(q[5])]),
                            st=int(q[8]), end=int(q[11]) if len(q) > 11 else 0)
                if curv is not None:
                    V[curv]['out'].append(i)
    if started:
        finish(P, V)
    return np.asarray(taus, np.float32), np.asarray(ns, np.int16)


def full_event_particles(event, max_particles):
    """Return (features, mask, nbaryon) for the WHOLE event in the thrust frame.
    event rows are [E, px, py, pz, pid, mass]. features columns:
    [z, cos_theta, phi, log10_mass, baryon_number, charge]."""
    out = np.zeros((max_particles, 6), dtype=np.float32)
    mask = np.zeros(max_particles, dtype=bool)
    if event.shape[0] == 0:
        return out, mask, 0
    p3 = event[:, 1:4]
    n = thrust_axis(p3)
    E = event[:, 0]
    E_total = E.sum()
    z = E / max(E_total, 1e-12)
    p_hat = p3 / (np.linalg.norm(p3, axis=1, keepdims=True) + 1e-12)
    cos_theta = p_hat @ n                               # signed, both hemispheres
    z_axis = np.array([0., 0., 1.])
    e1 = np.cross(n, z_axis)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1., 0., 0.])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    phi = np.arctan2(p_hat @ e2, p_hat @ e1)
    pid = event[:, 4].astype(int); mass = event[:, 5]
    bn = np.array([baryonnum(x) for x in pid], np.float32)
    ch = np.array([charge(x) for x in pid], np.float32)
    feats = np.stack([z, cos_theta, phi,
                      np.log10(np.maximum(mass, 1e-3)), bn, ch], axis=1).astype(np.float32)
    P = feats.shape[0]
    Pk = min(P, max_particles)
    order = np.argsort(-z)[:Pk]                          # keep hardest if truncating
    out[:Pk] = feats[order]; mask[:Pk] = True
    return out, mask, int((np.abs(bn) > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--max-particles', type=int, default=100)
    # 'auto' reads HepMC2 against HepMC3 from the file header. The parser's own error message
    # names hepmc2 and hepmc3, so argparse has to accept them.
    ap.add_argument('--fmt', choices=['auto', 'hepmc2', 'hepmc3', 'herwig'], default='auto')
    ap.add_argument('--no-parton', action='store_true',
                    help='skip the parton-level thrust (a second pass over the file)')
    args = ap.parse_args()
    print(f"Parsing {args.input} ...", flush=True)
    events = parse_hepmc(args.input, max_events=args.max_events, fmt=args.fmt)
    N = len(events)
    print(f"  {N} events; extracting FULL-event flavor particles ...", flush=True)
    parts = np.zeros((N, args.max_particles, 6), dtype=np.float32)
    masks = np.zeros((N, args.max_particles), dtype=bool)
    counts = np.zeros(N, dtype=np.int32)
    nbar = np.zeros(N, dtype=np.int32)
    for i, ev in enumerate(events):
        p, m, nb = full_event_particles(ev, args.max_particles)
        parts[i] = p; masks[i] = m; counts[i] = m.sum(); nbar[i] = nb
        if i and i % 50000 == 0:
            print(f"    {i}/{N}", flush=True)
    print(f"  counts: min={counts.min()} max={counts.max()} mean={counts.mean():.1f} "
          f"mean event-baryons={nbar.mean():.3f}", flush=True)
    extra = {}
    if not args.no_parton:
        print("  parton-level thrust ...", flush=True)
        tp, npar = parton_level_tau(args.input, fmt=args.fmt, max_events=args.max_events)
        assert len(tp) == N, f'parton pass saw {len(tp)} events, hadron pass {N}'
        ok = np.isfinite(tp)
        print(f"  parton level: {npar[ok].mean():.1f} partons/event, <tau> {np.nanmean(tp):.4f}, "
              f"{(~ok).sum()} events without partons", flush=True)
        extra = dict(tau_parton=tp, n_parton=npar)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, particles=parts, mask=masks, counts=counts, nbaryon=nbar, **extra)
    print(f"  saved {args.output}")


if __name__ == '__main__':
    main()
