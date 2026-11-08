"""Photon-free event shapes from the stored particle features (decision D4 of the linear-mix rerun).

Beam photon: a kept particle with baryon number 0, charge 0, log10 mass -3 (the clip) within 1e-4,
and ||phi| - pi/2| < 1e-7. In the thrust frame of extract_particles_full_flavor.py (e1 = n x z),
a particle exactly along the beam lies in the plane of the thrust axis and the beam, so its azimuth
is +-pi/2 (the float32 value sits 4.4e-8 above pi/2).

For every event with at least one beam photon, the three-momenta are rebuilt from the stored
features: E_i = z_i * E_vis, |p_i| = sqrt(E_i^2 - m_i^2) (m = 0 at the clip, i.e. photons and
electrons), direction from cos(theta) and phi about the stored thrust axis. E_vis is 91.2 GeV minus
the missing momentum of the rebuilt event (a fixed point; exact for one neutrino, an upper bound
otherwise). 1-T, B_total and rho_heavy are computed with and without the beam photons by
compute_hemisphere_shapes of gen/compute_efps.py (exact thrust axis), and the new value is
    stored - (with - without),
so what the rebuild misses (more than one neutrino, the truncation to 100 particles, float32
features) cancels to first order. Every other column, and every event without a beam photon,
keeps the original text of the CSV.
"""
import os, sys
import numpy as np

GEN = os.environ.get('NB_GEN', '/mnt/beegfs/bassi/reweighting/gen')
sys.path.insert(0, GEN)
import compute_efps as CE                      # noqa: E402  gen/compute_efps.py (exact thrust)

ECM = 91.2                                      # GeV: BEAM_ENERGIES 45.6 (Sherpa), Energy 91.2 (Herwig)
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy')
PHI_TOL, LM_TOL = 1e-7, 1e-4
SCALE_MODE = os.environ.get('NB_SCALE', 'balance')      # 'balance' (per event) or 'fixed' (91.2)


def beam_mask(P, M, chunk=20000):
    """(N, Pmax) bool: the beam photons of every event, in chunks to bound the float64 temporaries."""
    out = np.zeros(M.shape, bool)
    for a in range(0, len(M), chunk):
        p = P[a:a + chunk]
        lm = p[..., 3].astype(np.float64)
        ph = p[..., 2].astype(np.float64)
        out[a:a + chunk] = (M[a:a + chunk] & (p[..., 4] == 0) & (p[..., 5] == 0)
                            & (np.abs(lm + 3.0) < LM_TOL) & (np.abs(np.abs(ph) - np.pi / 2) < PHI_TOL))
    return out


def parts(f):
    """(n, 6) features -> z, mass, unit directions (n, 3), all float64."""
    f = f.astype(np.float64)
    z, ct, ph, lm = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    m = np.where(np.abs(lm + 3.0) < LM_TOL, 0.0, 10.0 ** lm)
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
    U = np.column_stack([st * np.cos(ph), st * np.sin(ph), ct])
    return z, m, U


def four(z, m, U, s):
    E = z * s
    p = np.sqrt(np.maximum(E * E - m * m, 0.0))
    return np.column_stack([E, p[:, None] * U])


def scale_of(z, m, U):
    """E_vis = ECM - |sum p| at E_vis (fixed point). Returns (s, clamped)."""
    if SCALE_MODE == 'fixed':
        return ECM, False
    s = ECM
    for _ in range(12):
        E = z * s
        p = np.sqrt(np.maximum(E * E - m * m, 0.0))
        miss = np.linalg.norm((p[:, None] * U).sum(0))
        s_new = ECM - miss
        if abs(s_new - s) < 1e-10:
            s = s_new
            break
        s = s_new
    if s < 0.5 * ECM:
        return 0.5 * ECM, True
    return s, False


def three(ev4):
    d = CE.compute_hemisphere_shapes(ev4)
    return d['1_minus_thrust'], d['B_total'], d['rho_heavy']


def event_delta(f, b, s=None):
    """with, without (each a 3-tuple) and the scale, for one event: f (n, 6) kept features,
    b (n,) beam-photon flags."""
    z, m, U = parts(f)
    if s is None:
        s, _ = scale_of(z, m, U)
    ev = four(z, m, U, s)
    return three(ev), three(ev[~b]), s


def corner_counts(P, M, B):
    """Two failure modes of the stored-feature criterion (counts of events), for the record.
    axis_on_beam: no photon flagged, but a massless neutral sits at |cos| >= 1 - 1e-7 with phi
      exactly 0 or +-pi: the thrust axis lies exactly on the beam (a hard ISR photon alone in its
      hemisphere), the frame falls back to e1 = x and the beam photons get phi 0, so they are missed.
    inconsistent: two or more flagged photons whose implied cos(axis, beam) = -cos*sign(phi),
      corrected for the 1e-12 regularization of p_hat, differ by more than 1e-4: one of them is not
      on the beam (the only other particle in the hemisphere of a hard ISR photon lies in the plane
      of the axis and the beam)."""
    out = dict(axis_on_beam=0, inconsistent=0)
    for a in range(0, len(M), 20000):
        p, mk, b = P[a:a + 20000], M[a:a + 20000], B[a:a + 20000]
        ct = p[..., 1].astype(np.float64); ph = p[..., 2].astype(np.float64)
        lm = p[..., 3].astype(np.float64)
        nm = mk & (p[..., 4] == 0) & (p[..., 5] == 0) & (np.abs(lm + 3.0) < LM_TOL)
        ax = nm & (np.abs(ct) >= 1 - 1e-7) & ((ph == 0) | (np.abs(np.abs(ph) - np.pi) < 1e-6)) & ~b
        out['axis_on_beam'] += int((ax.any(1) & (b.sum(1) == 0)).sum())
        pm = p[..., 0].astype(np.float64) * ECM
        shrink = pm / (pm + 1e-12)
        c = -ct * np.sign(ph) / np.where(shrink > 0, shrink, 1.0)
        for i in np.where(b.sum(1) >= 2)[0]:
            if np.ptp(c[i][b[i]]) > 1e-4:
                out['inconsistent'] += 1
    return out
