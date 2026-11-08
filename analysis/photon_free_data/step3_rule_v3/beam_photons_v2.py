"""Beam photons in the stored particle features (D4: they are removed from every event shape).

A beam photon is a photon radiated by the incoming e+ or e- along the beam. The particle files
(particles_full_<rid>.npz, from extract_particles_full_flavor.py) keep per particle
P = (z, cos theta, phi, log10 m, baryon, charge) in the thrust frame: angles about the thrust axis
n, with e1 = n x beam (x when n lies within 1e-6 of the beam) and e2 = n x e1. A particle along the
beam therefore sits in the plane of n and the beam, at phi = +-pi/2. The rule:

  base  a kept, massless (log10 m = -3, the clip, within 1e-4) and neutral (baryon 0, charge 0)
        particle with ||phi| - pi/2| < 1e-7.
  (i)   in an event where the base rule flags nothing, also every massless neutral particle with
        |cos theta| >= 1 - 1e-7 and phi exactly 0 or +-pi: the thrust axis lies on the beam (a hard
        beam photon is the hardest particle and sets it), so the frame falls back to e1 = x.
  (ii)  where the base rule flags two or more photons above 1 MeV (E = 91.2 GeV z), each flagged
        photon implies cos(axis, beam) = -cos theta sign(phi), after undoing p/(|p| + 1e-12 GeV).
        Photons above 1 MeV whose value differs by more than 1e-5 from the value shared by the most
        flagged photons above 1 MeV lose their flag (ties go to the value shared by more flagged
        photons below 1 MeV, within 1e-5 plus 30% of their undone shrink, then to the hardest
        photon's). Flags below 1 MeV stay.

    mask = beam_photon_mask(P, M)      # (N, Pmax) bool
    nb = n_beam(P, M)                  # (N,) int, the flagged photons of each event

Plain numpy, in chunks of events. `python beam_photons.py particles_full_<rid>.npz ...` runs the
self-test on real events.
"""
import numpy as np

ECM = 91.2                          # GeV, the event energy that turns z into E
LM_TOL = 1e-4                       # log10 m within this of -3: the 1 MeV clip, i.e. massless
PHI_TOL = 1e-7                      # ||phi| - pi/2| below this: in the plane of the axis and the beam
COS_AXIS = 1 - 1e-7                 # (i): |cos theta| at least this
PI32 = float(np.float32(np.pi))     # phi = +-pi as stored (float32)
E_MIN = 1e-3                        # (ii): photons above 1 MeV take part
SHRINK_UNC = 0.3                    # (ii): relative uncertainty of the undone shrink (E_vis < 91.2)
C_TOL = 1e-5                        # (ii): the same implied cos(axis, beam) within this
CHUNK = 50000


def _implied_cos(ct, ph, e):
    """cos(axis, beam) implied by a particle on the beam: a photon along +beam has phi = -pi/2 and
    cos theta = cos(axis, beam), one along -beam phi = +pi/2 and cos theta = -cos(axis, beam).
    The stored cos theta is shrunk by |p|/(|p| + 1e-12), which is undone with |p| = E."""
    return -ct * np.sign(ph) * (e + 1e-12) / np.maximum(e, 1e-300)


def _chunk(P, M):
    """flags and per-event diagnostics for one chunk of events"""
    M = np.asarray(M, bool)
    ct = P[..., 1].astype(np.float64)
    ph32 = P[..., 2]
    ph = ph32.astype(np.float64)
    neu = (M & (P[..., 4] == 0) & (P[..., 5] == 0)
           & (np.abs(P[..., 3].astype(np.float64) + 3.0) < LM_TOL))
    base = neu & (np.abs(np.abs(ph) - np.pi / 2) < PHI_TOL)
    n_base = base.sum(1)
    # (i) the thrust axis on the beam
    on_axis = (neu & (np.abs(ct) >= COS_AXIS) & ((ph32 == 0) | (np.abs(ph) == PI32))
               & (n_base == 0)[:, None])
    flag = base | on_axis
    # (ii) consistency of the implied cos(axis, beam) among the base flags above 1 MeV
    e = P[..., 0].astype(np.float64) * ECM
    hard = base & (e > E_MIN)
    n_hard = hard.sum(1)
    c = _implied_cos(ct, ph, e)
    spread = np.where(hard, c, -np.inf).max(1) - np.where(hard, c, np.inf).min(1)
    dropped = np.zeros(len(M), np.int64)
    for i in np.where((n_hard >= 2) & (spread > C_TOL))[0]:
        j = np.where(base[i])[0]
        cj, ej = c[i, j], e[i, j]
        above = ej > E_MIN
        tol = C_TOL + SHRINK_UNC * 1e-12 / np.maximum(ej, 1e-300)
        best, ref = None, None
        for k in np.where(above)[0]:
            same = np.abs(cj - cj[k]) <= C_TOL
            vote = ~above & (np.abs(cj - cj[k]) <= tol)
            key = (int((same & above).sum()), int(vote.sum()), float(ej[same & above].max()))
            if best is None or key > best:
                best, ref = key, cj[k]
        out = j[above & (np.abs(cj - ref) > C_TOL)]
        flag[i, out] = False
        dropped[i] = len(out)
    info = dict(n_base=n_base, rule_i=on_axis.any(1), n_dropped=dropped)
    return flag, info


def beam_photon_mask(P, M, chunk=CHUNK, detail=False):
    """(N, Pmax) bool: the beam photons among the kept particles. P: (N, Pmax, 6) features
    (z, cos theta, phi, log10 m, baryon, charge); M: (N, Pmax) mask of kept particles.
    detail=True also returns per-event arrays: n_base (base-rule flags), rule_i (the axis-on-beam
    rule fired) and n_dropped (flags removed by the consistency test)."""
    N = len(M)
    flag = np.zeros(np.shape(M), bool)
    info = dict(n_base=np.zeros(N, np.int64), rule_i=np.zeros(N, bool), n_dropped=np.zeros(N, np.int64))
    for a in range(0, N, chunk):
        f, d = _chunk(np.asarray(P[a:a + chunk]), np.asarray(M[a:a + chunk]))
        flag[a:a + chunk] = f
        for k in info:
            info[k][a:a + chunk] = d[k]
    return (flag, info) if detail else flag


def n_beam(P, M, chunk=CHUNK):
    """(N,) int: the number of beam photons among the kept particles of each event."""
    return beam_photon_mask(P, M, chunk=chunk).sum(1)


def _selftest(path, nchk=3000):
    """Checks on the events of one particle file; returns the statistics, raises on a failure."""
    z = np.load(path)
    P, M = z['particles'], z['mask'].astype(bool)
    flag, info = beam_photon_mask(P, M, detail=True)
    N = len(M)
    f = P.astype(np.float64)
    neu = M & (P[..., 4] == 0) & (P[..., 5] == 0) & (np.abs(f[..., 3] + 3.0) < LM_TOL)
    base = neu & (np.abs(np.abs(f[..., 2]) - np.pi / 2) < PHI_TOL)
    nb = flag.sum(1)
    # 1. only kept, massless, neutral particles are flagged
    assert not (flag & ~neu).any(), 'a flag on a massive, charged or masked particle'
    # 2. where the base rule flags something, the final flags are a subset of its flags
    has = base.any(1)
    assert not (flag[has] & ~base[has]).any(), 'a flag outside the base rule in an event it covers'
    assert np.array_equal(base.sum(1)[has] - info['n_dropped'][has], nb[has]), 'drop count'
    # 3. the axis-on-beam photons: |cos| = 1 within 1e-7, phi 0 or pi, only where nothing else is
    ax = flag & ~has[:, None]
    assert np.all(np.abs(f[..., 1][ax]) >= COS_AXIS), 'axis rule: |cos|'
    assert np.all((P[..., 2][ax] == 0) | (np.abs(f[..., 2][ax]) == PI32)), 'axis rule: phi'
    # 4. every event's flagged photons above 1 MeV agree on cos(axis, beam)
    e = f[..., 0] * ECM
    c = _implied_cos(f[..., 1], f[..., 2], e)
    hard = flag & has[:, None] & (e > E_MIN)
    sp = np.where(hard, c, -np.inf).max(1) - np.where(hard, c, np.inf).min(1)
    assert np.all(sp[hard.sum(1) >= 2] <= 2 * C_TOL), 'inconsistent flags remain'
    # 5. chunking does not matter, n_beam agrees
    k = min(N, nchk)
    assert np.array_equal(beam_photon_mask(P[:k], M[:k], chunk=97), flag[:k]), 'chunk dependence'
    assert np.array_equal(n_beam(P[:k], M[:k]), nb[:k]), 'n_beam'
    return dict(events=N, with_beam_photon=int((nb > 0).sum()),
                n_hist=np.bincount(np.minimum(nb, 4), minlength=5).tolist(),
                base_photons=int(base.sum()), photons=int(flag.sum()),
                axis_rule_events=int(info['rule_i'].sum()), axis_rule_photons=int(ax.sum()),
                consistency_events=int((info['n_dropped'] > 0).sum()),
                consistency_dropped=int(info['n_dropped'].sum()),
                energy_GeV=float((e * flag).sum() / N), truncated=int((M.sum(1) == M.shape[1]).sum()))


if __name__ == '__main__':
    import sys, os
    paths = sys.argv[1:] or [p for p in (
        '/mnt/beegfs/bassi/reweighting/gen/data_release/particles_full_7001.npz',
        '/mnt/beegfs/bassi/reweighting/gen/data_release/particles_full_8001.npz') if os.path.exists(p)]
    if not paths:
        sys.exit('usage: python beam_photons.py particles_full_<rid>.npz [...]')
    for p in paths:
        s = _selftest(p)
        print(f'{p}: OK')
        print('   ' + ', '.join(f'{k} {v}' for k, v in s.items()))
