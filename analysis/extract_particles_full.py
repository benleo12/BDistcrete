#!/usr/bin/env python3
"""
Full-event particle extractor in the thrust frame (both hemispheres).

This is the full-event counterpart of extract_particles.py, which kept only the heavy
hemisphere. Keeping only one hemisphere hides half the event from the network, which is why
the total jet broadening B_total (a sum over both hemispheres) and the hemisphere masses are
the hardest observables to reweight. Here we keep ALL final-state particles and parametrise
each in the event thrust frame, so the signed cos(theta) carries the hemisphere and the full
event structure is visible.

Output npz (same layout as extract_particles.py, drop-in):
    particles : [N, P_max, 3] float32  per-particle (z, cos_theta, phi)
        z         = E_i / E_total            energy fraction of the whole event
        cos_theta = p_hat_i . n_thrust       signed, in [-1, 1] (both hemispheres)
        phi       = azimuth around n_thrust
    mask  : [N, P_max] bool
    counts: [N] int32

The downstream feature builder turns (z, cos_theta, phi) into
    [cos_theta, phi, log z, log(1 - cos_theta^2)]
where log(1 - cos_theta^2) = log(sin^2 theta) is the (squared) transverse angle to the
thrust axis. It diverges to -inf at BOTH axis ends, so it exposes collinearity to either
hemisphere and makes the per-particle broadening contribution z*sin(theta) accessible.
"""
import argparse, gzip, numpy as np
from pathlib import Path

NEUTRINO = {12, 14, 16}


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


def parse_hepmc(filepath, max_events=None):
    opener = gzip.open if str(filepath).endswith('.gz') else open
    events, cur = [], []
    with opener(filepath, 'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if cur:
                    events.append(np.asarray(cur, dtype=np.float64)); cur = []
                    if max_events is not None and len(events) >= max_events:
                        break
                continue
            if not line.startswith('P '):
                continue
            parts = line.split()
            try:
                if len(parts) >= 10:
                    pid = int(parts[3]); status = int(parts[9])
                    px, py, pz, E = (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
                elif len(parts) >= 9:
                    pid = int(parts[2]); status = int(parts[8])
                    px, py, pz, E = (float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]))
                else:
                    continue
                if status == 1 and abs(pid) not in NEUTRINO:
                    cur.append([E, px, py, pz])
            except (ValueError, IndexError):
                continue
    if cur:
        events.append(np.asarray(cur, dtype=np.float64))
    return events


def full_event_particles(event4, max_particles):
    """Return (features, mask) for the WHOLE event in the thrust frame."""
    if event4.shape[0] == 0:
        return (np.zeros((max_particles, 3), dtype=np.float32),
                np.zeros(max_particles, dtype=bool))
    p3 = event4[:, 1:4]
    n = thrust_axis(p3)
    E_total = event4[:, 0].sum()
    z = event4[:, 0] / max(E_total, 1e-12)
    p_hat = p3 / (np.linalg.norm(p3, axis=1, keepdims=True) + 1e-12)
    cos_theta = p_hat @ n                              # signed, both hemispheres
    z_axis = np.array([0., 0., 1.])
    e1 = np.cross(n, z_axis)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1., 0., 0.])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    phi = np.arctan2(p_hat @ e2, p_hat @ e1)
    feats = np.stack([z, cos_theta, phi], axis=1).astype(np.float32)
    P = feats.shape[0]
    out = np.zeros((max_particles, 3), dtype=np.float32)
    mask = np.zeros(max_particles, dtype=bool)
    Pk = min(P, max_particles)
    order = np.argsort(-z)[:Pk]                          # keep hardest if truncating
    out[:Pk] = feats[order]; mask[:Pk] = True
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--max-particles', type=int, default=100)
    args = ap.parse_args()
    print(f"Parsing {args.input} ...", flush=True)
    events = parse_hepmc(args.input, max_events=args.max_events)
    N = len(events)
    print(f"  {N} events; extracting FULL-event particles ...", flush=True)
    parts = np.zeros((N, args.max_particles, 3), dtype=np.float32)
    masks = np.zeros((N, args.max_particles), dtype=bool)
    counts = np.zeros(N, dtype=np.int32)
    for i, ev in enumerate(events):
        p, m = full_event_particles(ev, args.max_particles)
        parts[i] = p; masks[i] = m; counts[i] = m.sum()
        if i and i % 50000 == 0:
            print(f"    {i}/{N}", flush=True)
    print(f"  counts: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, particles=parts, mask=masks, counts=counts)
    print(f"  saved {args.output}")


if __name__ == '__main__':
    main()
