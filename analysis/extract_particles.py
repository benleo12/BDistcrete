#!/usr/bin/env python3
"""
Extract per-event heavy-hemisphere particle lists from a HepMC file.

Output: a compressed npz with two arrays
    particles : [N, P_max, 3]  float32   per-particle (E/Q_hemi, cos_theta, phi)
    mask      : [N, P_max]     bool      True for real particle, False for pad
where P_max is the maximum number of hemisphere particles in any event
(typically <= 60 for LEP Z -> qq events).

The representation is IRC-safe: the event embedding used downstream is
    sum_i z_i * f(hat p_i)  with  z_i = E_i / Q_hemi  and f smooth,
so collinear splittings (same hat p, split z) and soft particles (z -> 0)
do not change the embedding up to NN smoothness.

We parametrise particle directions by (cos theta_thrust, phi_thrust) in the
event-thrust frame, which is rotationally invariant around the thrust axis
in expectation and reduces the input to a 2D angular feature.
"""
import argparse, gzip, numpy as np, sys
from pathlib import Path

NEUTRINO = {12, 14, 16}


def thrust_axis(p3):
    """Compute thrust axis for a list of 3-momenta (Nx3 array). Simple
    approximate algorithm: iterate the sign-weighted sum."""
    # Start from the direction of highest-|p| particle
    if len(p3) == 0:
        return np.array([0, 0, 1.0])
    mags = np.linalg.norm(p3, axis=1)
    n = p3[np.argmax(mags)] / (mags[np.argmax(mags)] + 1e-12)
    for _ in range(20):
        signs = np.sign(p3 @ n)
        n_new = (signs[:, None] * p3).sum(0)
        nn = np.linalg.norm(n_new)
        if nn < 1e-12: break
        n_new = n_new / nn
        if np.dot(n_new, n) > 1 - 1e-12: break
        n = n_new
    return n


def parse_hepmc(filepath, max_events=None):
    opener = gzip.open if str(filepath).endswith('.gz') else open
    events = []
    cur = []
    with opener(filepath, 'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if cur:
                    events.append(np.asarray(cur, dtype=np.float64))
                    cur = []
                    if max_events is not None and len(events) >= max_events:
                        break
                continue
            if not line.startswith('P '):
                continue
            parts = line.split()
            try:
                if len(parts) >= 10:
                    pid = int(parts[3]); status = int(parts[9])
                    px, py, pz, E = (float(parts[4]), float(parts[5]),
                                     float(parts[6]), float(parts[7]))
                elif len(parts) >= 9:
                    pid = int(parts[2]); status = int(parts[8])
                    px, py, pz, E = (float(parts[3]), float(parts[4]),
                                     float(parts[5]), float(parts[6]))
                else:
                    continue
                if status == 1 and abs(pid) not in NEUTRINO:
                    cur.append([E, px, py, pz])
            except (ValueError, IndexError):
                continue
    if cur: events.append(np.asarray(cur, dtype=np.float64))
    return events


def hemisphere_particles(event4, max_particles):
    """Return (features, mask) for the heavy hemisphere of this event.

    features : (max_particles, 3)  rows of (z_i, cos_theta_i, phi_i)
    mask     : (max_particles,)   True for real particle
    """
    if event4.shape[0] == 0:
        return (np.zeros((max_particles, 3), dtype=np.float32),
                np.zeros(max_particles, dtype=bool))
    p3 = event4[:, 1:4]
    n = thrust_axis(p3)
    signs = np.sign(p3 @ n)                         # +/-1 per particle
    # Pick the hemisphere with larger invariant mass^2
    def hemi_mass2(mask):
        p = event4[mask]
        if len(p) == 0: return -1.0
        P = p.sum(axis=0)   # (E, px, py, pz)
        return P[0]**2 - (P[1]**2 + P[2]**2 + P[3]**2)
    heavy_sign = 1 if hemi_mass2(signs > 0) >= hemi_mass2(signs < 0) else -1
    keep = (signs == heavy_sign)
    hemi = event4[keep]
    if hemi.shape[0] == 0:
        return (np.zeros((max_particles, 3), dtype=np.float32),
                np.zeros(max_particles, dtype=bool))
    # Energy fraction relative to hemisphere total
    E_hemi = hemi[:, 0].sum()
    z = hemi[:, 0] / max(E_hemi, 1e-12)
    p_hat = hemi[:, 1:4] / (np.linalg.norm(hemi[:, 1:4], axis=1, keepdims=True) + 1e-12)
    cos_theta = p_hat @ n
    # Angle phi in plane orthogonal to n: build two basis vectors
    # e1 = (n cross z_axis) normalized; fallback if n is parallel to z_axis
    z_axis = np.array([0., 0., 1.])
    e1 = np.cross(n, z_axis)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1., 0., 0.])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    px_perp = p_hat @ e1
    py_perp = p_hat @ e2
    phi = np.arctan2(py_perp, px_perp)
    feats = np.stack([z, cos_theta, phi], axis=1).astype(np.float32)
    # Truncate/pad
    P = feats.shape[0]
    out = np.zeros((max_particles, 3), dtype=np.float32)
    mask = np.zeros(max_particles, dtype=bool)
    Pk = min(P, max_particles)
    # Sort by descending z so truncation drops the softest particles (minimal IRC impact)
    order = np.argsort(-z)[:Pk]
    out[:Pk] = feats[order]
    mask[:Pk] = True
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='HepMC file')
    ap.add_argument('--output', required=True, help='output .npz')
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--max-particles', type=int, default=60)
    args = ap.parse_args()

    print(f"Parsing {args.input} ...", flush=True)
    events = parse_hepmc(args.input, max_events=args.max_events)
    N = len(events)
    print(f"  {N} events loaded; extracting heavy-hemisphere particles ...", flush=True)
    parts = np.zeros((N, args.max_particles, 3), dtype=np.float32)
    masks = np.zeros((N, args.max_particles), dtype=bool)
    counts = np.zeros(N, dtype=np.int32)
    for i, ev in enumerate(events):
        p, m = hemisphere_particles(ev, args.max_particles)
        parts[i] = p; masks[i] = m; counts[i] = m.sum()
        if i and i % 50000 == 0:
            print(f"    {i}/{N}", flush=True)
    print(f"  counts: min={counts.min()}  max={counts.max()}  "
          f"mean={counts.mean():.1f}  median={np.median(counts):.0f}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, particles=parts, mask=masks, counts=counts)
    print(f"  saved {args.output}  ({parts.nbytes/1e6:.1f} MB raw)")


if __name__ == '__main__':
    main()
