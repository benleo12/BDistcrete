#!/usr/bin/env python3
"""Add the missing physics family to the closure: the inclusive scaled-momentum spectrum x_p.

The ladder already closes an event shape (1-T), a counting observable (multiplicity) and a
hadron-species observable (baryon number). The scaled-momentum spectrum x_p = 2|p|/Q (here
z = E/Q, massless hadrons) is the fourth, distinct family, the fragmentation spectrum. It is
evaluated post-hoc on the ALREADY-trained Stage C conditional: the reference is reweighted to
each held-out point with the exported network, and the per-particle x_p histogram is compared
to the held-out run with the same sqrt(chi^2/ndf) ruler (null 1).

Because many particles come from one event, the per-bin error uses the effective number of
EVENTS (Kish) contributing to the bin, v_e = w_e * n_e(bin), not the raw particle count, so an
event dumping several particles into a bin counts as one correlated unit rather than several
independent ones. Writes the x_p width into output/ladder_final.json under C.per_obs.
"""
import json, numpy as np
from ab_analysis import Cond

STAGE = 'C'
DATA = 'data_stageC'
HELD = {6900: (0.116, 0.38, 1.00), 6901: (0.124, 0.55, 1.50), 6902: (0.120, 0.46, 1.21),
        6903: (0.114, 0.60, 0.95), 6904: (0.126, 0.35, 1.60)}
XP_BINS = np.linspace(0.0, 0.35, 22)          # scaled-momentum range holding the bulk of hadrons
MIN_EFF = 5


def per_event_bincontents(zmat, mask, bins):
    """n_e(bin): particles of each event falling in each bin -> (nevent, nbin).
    Vectorized with a single 2D histogram over (event index, x_p) rather than a per-event loop."""
    ne = zmat.shape[0]
    m = mask.astype(bool)
    ev = np.repeat(np.arange(ne), m.sum(1))         # event id of each valid particle
    z = zmat[m]                                     # flat valid x_p values
    H, _, _ = np.histogram2d(ev, z, bins=[np.arange(ne+1)-0.5, bins])
    return H


def density_and_eff(counts_ev, w):
    """Weighted density over bins and Kish-effective EVENT count per bin.
    counts_ev: (nevent, nbin) particles per event per bin; w: per-event weight (normalized)."""
    bw = np.diff(XP_BINS)
    v = counts_ev * w[:, None]                 # per-event contribution to each bin
    content = v.sum(0)                          # total (weighted) particles per bin
    v2 = (counts_ev**2) * (w[:, None]**2)
    neff = np.where(v2.sum(0) > 0, content**2/np.where(v2.sum(0) > 0, v2.sum(0), 1.0), 0.0)
    W = content.sum()
    dens = content/(W*bw) if W > 0 else np.zeros_like(content)
    return dens, neff


def width_xp(ref_counts, wref, tgt_counts):
    wt = np.full(len(tgt_counts), 1.0/len(tgt_counts))
    da, na = density_and_eff(ref_counts, wref)
    db, nb = density_and_eff(tgt_counts, wt)
    bw = np.diff(XP_BINS)
    qa, qb = da*bw, db*bw
    Na, Nb = na.sum(), nb.sum()          # not used directly; pooled q below
    qhat = (qa*na + qb*nb)/np.maximum(na+nb, 1e-9)
    safe_a = np.where(na > 0, na, 1.0); safe_b = np.where(nb > 0, nb, 1.0)
    var = qhat**2*(1-qhat)*(1/safe_a + 1/safe_b)/bw**2
    sig = np.sqrt(np.maximum(var, 0))
    keep = (sig > 0) & (na >= MIN_EFF) & (nb >= MIN_EFF)
    pull = (da[keep]-db[keep])/sig[keep]
    ndf = max(keep.sum()-1, 1)
    return float(np.sqrt(np.sum(pull**2)/ndf)), int(keep.sum())


def main():
    cond = Cond(STAGE)
    ref = np.load(f'output/models/{STAGE}_ref.npz')
    # reference particles: z = E/Q is column 0 of the stored particle tensor
    ref_counts = per_event_bincontents(ref['particles'][:, :, 0], ref['mask'], XP_BINS)
    T = cond.T
    widths = []
    for rid, theta in HELD.items():
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        p, m = d['particles'], d['mask']
        tgt_counts = per_event_bincontents(p[:, :, 0], m, XP_BINS)
        f = cond.f_at(theta); w = np.exp((f-f.max())/T); w /= w.sum()
        wdt, nb = width_xp(ref_counts, w, tgt_counts)
        widths.append(wdt)
        print(f'  {theta}: x_p width={wdt:.3f} ({nb} bins)')
    xp_width = float(np.mean(widths))
    print(f'[C] x_p closure width = {xp_width:.3f} (null 1.0)')
    lad = json.load(open('output/ladder_final.json'))
    lad['C']['per_obs']['x_p'] = xp_width
    json.dump(lad, open('output/ladder_final.json', 'w'), indent=1)
    print('X_P CLOSURE DONE ->', xp_width)


if __name__ == '__main__':
    main()
