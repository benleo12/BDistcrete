"""The photons radiated by the beams, among the stored particles (the paper's rule, version 3).

The paper leaves these photons out of every observable it computes from the particles of an
event, while the networks read every particle, as they were trained. gentune.bands.observables
uses this rule, and so can any observable of your own: mask = beam_photon_mask(F, M) for the
features F and mask M of gentune.features.features_from_events, then use M & ~mask.

This module is a copy of analysis/beam_photons.py, the file of the paper's code at RULE_VERSION 3
(md5 ea7f82ed77c067ee727114a72d038d93). Everything from RULE_VERSION to n_beam() is that file's
code, unchanged. Only this docstring and the self-test at the end are written for the release.
Pure numpy.

A beam photon is a photon radiated by the incoming e+ or e- along the beam. The features are
P = (z, cos theta, phi, log10 m, baryon, charge) per particle in the thrust frame, the same for
gentune.features and for the training samples (generators/extract_particles_full_flavor.py):
angles about the thrust axis n, with e1 = n x beam (x when n lies within 1e-6 of the beam),
e2 = n x e1 and p_hat = p/(|p| + 1e-12), which shrinks cos theta by 1e-12/E. A particle along the
beam therefore sits in the plane of n and the beam, at phi = +-pi/2. Below, E = z s with s the
event's visible energy from momentum balance (s = 91.2 GeV - |sum p|, as in the paper's
photon-free event shapes). The rule, RULE_VERSION 3:

  base  a kept, massless (log10 m = -3, the clip, within 1e-4) and neutral (baryon 0, charge 0)
        particle with ||phi| - pi/2| < 1e-7.
  (i)   only where the base rule flags nothing: every massless neutral particle with phi exactly 0 or
        +-pi and 1 - |cos theta| <= 1e-7 + 1.5e-12/E, if one has |cos theta| >= 1 - 1e-7. A hard beam
        photon alone in its hemisphere sets the axis exactly on the beam, so the frame falls back to
        e1 = x; a soft beam photon on the axis sits 1e-12/E below |cos theta| = 1 (the shrink).
  (ii)  a base-flagged photon with |cos theta| >= 1 - 1e-7 loses its flag if it is the hardest particle
        (largest |p|) of a whole event (fewer kept particles than the file holds, so none was dropped)
        without missing momentum (|sum p| < 1e-4 GeV), and the other kept particles of its hemisphere
        carry a summed transverse momentum, relative to the photon's direction, below 1e-6 of its energy
        (0 for a photon alone). Each of their angles counts with the largest error its float32 cos theta
        allows (_angle_err): a particle within 2.44e-4 rad of the axis is stored at cos theta = 1, its
        stored angles give it no transverse momentum, and it counts at least 2.44e-4 times its |p|.
        Why: the thrust iteration starts at n0 = p/(|p| + 1e-12) of the hardest particle, bit for bit
        the photon's own p_hat. Its first step points along 2 (sum p of the hemisphere) + p_miss, here
        within 1e-6 of n0 up to the missing momentum (the counted transverse momentum bounds the true
        one), and it stops with n = n0 when that angle is below sqrt(2e-12 (1 - 1/E)) (E in GeV), which
        is above 1e-6 for E > 2 GeV (1.26e-6 at 5 GeV; the hardest particle was above 3 GeV in all 1.12M
        events checked). A beam photon has py = 0 and |px| < 1e-6 E (px = 0 in Sherpa, below 2e-16 E in
        Herwig), so n is within 1e-6 of the beam, e1 = x, p_hat.e2 = +-0 and phi = atan2(+-0, p_hat.e1)
        is 0 or +-pi, never +-pi/2. Limits: particles the file dropped would tilt the first step unseen,
        so a truncated event is left out; a true missing momentum below the 1e-4 GeV cut tilts it by
        about |p_miss|/(2E), so a beam photon of an event whose neutrinos carry 1e-6 to 1e-4 GeV in all
        can still lose its flag.
  (iii) every other base-flagged photon implies cos(axis, beam) = -cos theta sign(phi), the shrink
        undone. Two values agree within 1e-5 (1e-4 for a photon below 1e-7 GeV, whose undone shrink
        depends on s). Where they do not all agree, the deciding value is that of the largest group of
        agreeing photons, then of the group with the hardest photon; where no two flags agree (two
        photons sharing a hemisphere, or flags in different hemispheres), that of the hardest photon.
        A photon more than 1e-3 from the deciding value loses its flag: it lies in the plane by chance,
        as the only other particle in the hemisphere of a beam photon does. Closer ones keep it.

    mask = beam_photon_mask(P, M)      # (N, Pmax) bool
    nb = n_beam(P, M)                  # (N,) int, the flagged photons of each event

The self-test runs the rule on stored events and checks what the rule says, event by event:

    python -m gentune.beam_photons models/MIX17_selftest.npz [more files]

Any npz with the arrays 'particles' and 'mask' will do, such as the self-test bundles of
models/ or the particle files of generators/extract_particles_full_flavor.py. For a file that
fails it prints FAIL and the reason, such as the first event that breaks the rule, and stops with
exit 1. For a file that passes it prints OK and these counts: events; with_beam_photon, the
events with at least one beam photon; n_hist, the events with 0, 1, 2, 3 and 4 or more;
base_photons, the photons the base rule flags, and photons, those flagged in the end;
axis_rule_events, axis_rule_photons and axis_rule_soft_photons, the events where (i) applies, the
photons it flags, and those of them below |cos theta| = 1 - 1e-7; lone_events, the events where
(ii) removes a flag; conflict_events, conflict_dropped and tie_events, the events where (iii)
removes flags, the flags it removes, and those of these events where the hardest photon decides;
whole, the events that hold all their particles (fewer than the file has room for), truncated, the
others, and whole_one_per_side, the whole events with one beam photon along each beam; energy_GeV,
the energy of the beam photons summed over all events, divided by their number.
"""
import numpy as np

RULE_VERSION = 3
ECM = 91.2                          # GeV, the event energy
LM_TOL = 1e-4                       # log10 m within this of -3: the 1 MeV clip, i.e. massless
PHI_TOL = 1e-7                      # ||phi| - pi/2| below this: in the plane of the axis and the beam
COS_AXIS = 1 - 1e-7                 # (i), (ii): |cos theta| at least this: on the axis
AXIS_SHRINK = 1.5e-12               # (i): a soft photon on the axis, 1 - |cos theta| <= 1e-7 + this / E
PI32 = float(np.float32(np.pi))     # phi = +-pi as stored (float32)
P_MISS = 1e-4                       # (ii): GeV, no missing momentum below this
PT_REL = 1e-6                       # (ii): the rest of the hemisphere below this times E in summed pT
C_TOL = 1e-5                        # (iii): two implied values agree within this,
C_TOL_SOFT, E_SOFT = 1e-4, 1e-7     #        within this if a photon is below E_SOFT GeV
C_GAP = 1e-3                        # (iii): further than this from the deciding value: no beam photon
CHUNK = 50000


def _implied_cos(ct, ph, e):
    """cos(axis, beam) implied by a particle on the beam: a photon along +beam has phi = -pi/2 and
    cos theta = cos(axis, beam), one along -beam phi = +pi/2 and cos theta = -cos(axis, beam).
    The stored cos theta is shrunk by |p|/(|p| + 1e-12), which is undone with |p| = E."""
    return -ct * np.sign(ph) * (e + 1e-12) / np.maximum(e, 1e-300)


def _balance(P, M):
    """per event, the visible energy s and the missing momentum |sum p(s)|, s = ECM - |sum p(s)| solved
    from s = ECM as in nb_core.scale_of (12 steps, an event stops once s moves by less than 1e-10;
    s >= ECM/2), and |p| of every kept particle at s"""
    f = np.asarray(P, np.float64)
    M = np.asarray(M, bool)
    z = np.where(M, f[..., 0], 0.0)
    m = np.where(np.abs(f[..., 3] + 3.0) < LM_TOL, 0.0, 10.0 ** f[..., 3])
    st = np.sqrt(np.maximum(1.0 - f[..., 1] ** 2, 0.0))
    U = np.stack([st * np.cos(f[..., 2]), st * np.sin(f[..., 2]), f[..., 1]], -1)
    s, miss = np.full(len(M), ECM), np.zeros(len(M))
    run = np.ones(len(M), bool)
    for _ in range(12):
        i = np.where(run)[0]
        if len(i) == 0:
            break
        E = z[i] * s[i, None]
        p = np.sqrt(np.maximum(E * E - m[i] * m[i], 0.0))
        miss[i] = np.linalg.norm((p[..., None] * U[i]).sum(1), axis=1)
        new = ECM - miss[i]
        run[i[np.abs(new - s[i]) < 1e-10]] = False
        s[i] = new
    s = np.maximum(s, 0.5 * ECM)
    E = z * s[:, None]
    return s, miss, np.where(M, np.sqrt(np.maximum(E * E - m * m, 0.0)), 0.0)


def _angle_err(ct):
    """per value of a stored (float32, rounded to the nearest) cos theta: the largest error of the polar angle
    it allows, in rad. 2.44e-4 where it reads +-1 (any angle below that), 1.0e-4 at the next float32 value,
    then falling as 3e-8 / sin theta"""
    c32 = np.abs(np.asarray(ct, np.float32))
    h = np.where(c32 >= 1, 2.0 ** -25, np.spacing(c32).astype(np.float64) / 2)    # half the float32 step
    c = c32.astype(np.float64)
    t = np.arccos(np.minimum(c, 1.0))
    return np.maximum(t - np.arccos(np.minimum(c + h, 1.0)), np.arccos(np.maximum(c - h, -1.0)) - t)


def _hemi_parts(P, M, j, pabs):
    """for particle j of one event (P (Pmax, 6), M (Pmax,), pabs (Pmax,) the |p| of _balance), each other kept
    particle of j's hemisphere (the same sign of cos theta): its transverse momentum relative to j's
    direction, the angle counted with the largest error its stored cos theta allows (_angle_err), and
    whether it is stored at cos theta = +-1 (within 2.44e-4 rad of the axis)"""
    ct = P[:, 1].astype(np.float64)
    ph = P[:, 2].astype(np.float64)
    st = np.sqrt(np.maximum(1.0 - ct ** 2, 0.0))
    U = np.stack([st * np.cos(ph), st * np.sin(ph), ct], -1)
    o = np.asarray(M, bool) & (np.sign(ct) == np.sign(ct[j]))
    o[j] = False
    sine = np.linalg.norm(np.cross(U[o], U[j]), axis=1) + _angle_err(P[o, 1])
    return pabs[o] * sine, np.abs(ct[o]) == 1.0


def _hemi_pt(P, M, j, pabs):
    """for particle j of one event: the summed transverse momentum of _hemi_parts"""
    return float(_hemi_parts(P, M, j, pabs)[0].sum())


def _decide(c, e, tol):
    """(iii) for one event's flagged photons (implied values c, energies e, tolerances tol): the
    photons that lose their flag, and whether no two photons agreed (the hardest photon decided)"""
    agree = np.abs(c[:, None] - c[None, :]) <= np.maximum(tol[:, None], tol[None, :])
    size = agree.sum(1)
    k = max(range(len(c)), key=lambda q: (size[q], e[agree[q]].max()))
    return np.abs(c - c[k]) > C_GAP, size[k] < 2


def _chunk(P, M):
    """flags and per-event diagnostics for one chunk of events"""
    M = np.asarray(M, bool)
    N = len(M)
    ct = P[..., 1].astype(np.float64)
    ph32 = P[..., 2]
    ph = ph32.astype(np.float64)
    e91 = P[..., 0].astype(np.float64) * ECM
    neu = (M & (P[..., 4] == 0) & (P[..., 5] == 0)
           & (np.abs(P[..., 3].astype(np.float64) + 3.0) < LM_TOL))
    base = neu & (np.abs(np.abs(ph) - np.pi / 2) < PHI_TOL)
    n_base = base.sum(1)
    flag = base.copy()
    info = dict(n_base=n_base, rule_i=np.zeros(N, bool), n_axis_soft=np.zeros(N, np.int64),
                n_lone=np.zeros(N, np.int64), n_conflict=np.zeros(N, np.int64), tie=np.zeros(N, bool))
    # (i) the thrust axis on the beam
    axp = neu & ((ph32 == 0) | (np.abs(ph) == PI32)) & (n_base == 0)[:, None]
    i = np.where((axp & (np.abs(ct) >= COS_AXIS)).any(1))[0]
    if len(i):
        s = _balance(P[i], M[i])[0]
        e = P[i, :, 0].astype(np.float64) * s[:, None]
        soft = 1 - np.abs(ct[i]) <= 1 - COS_AXIS + AXIS_SHRINK / np.maximum(e, 1e-300)
        on = axp[i] & ((np.abs(ct[i]) >= COS_AXIS) | soft)
        flag[i] = on
        info['rule_i'][i] = True
        info['n_axis_soft'][i] = (on & (np.abs(ct[i]) < COS_AXIS)).sum(1)
    # (ii) a photon on the axis, the hardest particle of a whole event, no missing momentum, the rest of its
    # hemisphere below PT_REL of its energy in summed transverse momentum
    lone = base & (np.abs(ct) >= COS_AXIS) & (M.sum(1) < M.shape[1])[:, None]
    i = np.where(lone.any(1))[0]
    if len(i):
        s, miss, pm = _balance(P[i], M[i])
        e = P[i, :, 0] * s[:, None]
        cand = lone[i] & (miss < P_MISS)[:, None] & (e >= pm.max(1)[:, None])
        drop = np.zeros_like(cand)
        for k, j in zip(*np.where(cand)):
            drop[k, j] = _hemi_pt(P[i[k]], M[i[k]], j, pm[k]) < PT_REL * e[k, j]
        flag[i] &= ~drop
        info['n_lone'][i] = drop.sum(1)
    # (iii) the implied cos(axis, beam) of the flags left. The visible energy moves a value by at most
    # 1e-12/E (at 91.2 GeV), so where the spread plus that stays within C_TOL all agree for any s
    left = flag & base
    c = _implied_cos(ct, ph, e91)
    spread = np.where(left, c, -np.inf).max(1) - np.where(left, c, np.inf).min(1)
    shift = 1e-12 / np.where(left, np.maximum(e91, 1e-300), np.inf).min(1)
    i = np.where((left.sum(1) >= 2) & (spread + shift > C_TOL))[0]
    if len(i):
        s = _balance(P[i], M[i])[0]
        e = P[i, :, 0].astype(np.float64) * s[:, None]
        cs = _implied_cos(ct[i], ph[i], e)
        tol = np.where(e < E_SOFT, C_TOL_SOFT, C_TOL)
        li = left[i]
        hard = li & (tol == C_TOL)
        sp_all = np.where(li, cs, -np.inf).max(1) - np.where(li, cs, np.inf).min(1)
        sp_hard = np.where(hard, cs, -np.inf).max(1) - np.where(hard, cs, np.inf).min(1)
        for k in np.where((sp_all > C_TOL_SOFT) | (sp_hard > C_TOL))[0]:
            j = np.where(li[k])[0]
            out, tie = _decide(cs[k, j], e[k, j], tol[k, j])
            flag[i[k], j[out]] = False
            info['n_conflict'][i[k]] = int(out.sum())
            info['tie'][i[k]] = tie and out.any()
    info['n_dropped'] = info['n_lone'] + info['n_conflict']
    return flag, info


def beam_photon_mask(P, M, chunk=CHUNK, detail=False):
    """(N, Pmax) bool: the beam photons among the kept particles. P: (N, Pmax, 6) features
    (z, cos theta, phi, log10 m, baryon, charge); M: (N, Pmax) mask of kept particles.
    detail=True also returns per-event arrays: n_base (base-rule flags), rule_i (the axis-on-beam rule
    fired), n_axis_soft (its photons below |cos theta| = 1 - 1e-7), n_lone (flags removed by (ii)),
    n_conflict (by (iii)), n_dropped (both) and tie ((iii) decided by the hardest photon)."""
    N = len(M)
    flag = np.zeros(np.shape(M), bool)
    info = dict(n_base=np.zeros(N, np.int64), rule_i=np.zeros(N, bool),
                n_axis_soft=np.zeros(N, np.int64), n_lone=np.zeros(N, np.int64),
                n_conflict=np.zeros(N, np.int64), n_dropped=np.zeros(N, np.int64), tie=np.zeros(N, bool))
    for a in range(0, N, chunk):
        f, d = _chunk(np.asarray(P[a:a + chunk]), np.asarray(M[a:a + chunk]))
        flag[a:a + chunk] = f
        for k in info:
            info[k][a:a + chunk] = d[k]
    return (flag, info) if detail else flag


def n_beam(P, M, chunk=CHUNK):
    """(N,) int: the number of beam photons among the kept particles of each event."""
    return beam_photon_mask(P, M, chunk=chunk).sum(1)



# -- the self-test ---------------------------------------------------------------------------------
# The checks of _selftest in analysis/beam_photons.py, one for one. They raise RuleCheckError rather
# than use assert, so that python -O cannot switch them off.
class RuleCheckError(AssertionError):
    """A stored event where the flags do not do what the rule says."""


def _need(ok, what):
    if not ok:
        raise RuleCheckError(what)


def check(P, M, nchk=3000):
    """Run the rule on the events (P, M) and check it event by event. Returns the counts the paper's
    self-test prints, and raises RuleCheckError at the first event that breaks the rule."""
    P, M = np.asarray(P), np.asarray(M, bool)
    flag, info = beam_photon_mask(P, M, detail=True)
    N = len(M)
    f = P.astype(np.float64)
    neu = M & (P[..., 4] == 0) & (P[..., 5] == 0) & (np.abs(f[..., 3] + 3.0) < LM_TOL)
    base = neu & (np.abs(np.abs(f[..., 2]) - np.pi / 2) < PHI_TOL)
    nb = flag.sum(1)
    e = f[..., 0] * ECM
    # 1. only kept, massless, neutral particles are flagged
    _need(not (flag & ~neu).any(), 'a flag on a massive, charged or masked particle')
    # 2. where the base rule flags something, the final flags are a subset of its flags
    has = base.any(1)
    _need(not (flag[has] & ~base[has]).any(), 'a flag outside the base rule in an event it covers')
    _need(np.array_equal(base.sum(1)[has] - info['n_dropped'][has], nb[has]), 'drop count')
    # 3. the axis-on-beam photons: phi 0 or pi, |cos| within 1e-7 + 1.5e-12/E of 1 (s >= 45.6 GeV),
    #    only where nothing else is, and one of them within 1e-7
    ax = flag & ~has[:, None]
    _need(np.all((P[..., 2][ax] == 0) | (np.abs(f[..., 2][ax]) == PI32)), 'axis rule: phi')
    _need(np.all(1 - np.abs(f[..., 1][ax]) <= 1e-7 + 2 * AXIS_SHRINK / e[ax]), 'axis rule: |cos|')
    _need(np.all((ax & (np.abs(f[..., 1]) >= COS_AXIS)).any(1) == ax.any(1)), 'axis rule: none on axis')
    # 4. no flagged photon on the axis that is the hardest particle of a whole event without missing
    #    momentum, the rest of its hemisphere below PT_REL of its energy in summed pT; and every flag
    #    that (ii) removed was such a photon (none in a truncated event)
    ct = f[..., 1]
    whole = M.sum(1) < M.shape[1]
    _need(not (info['n_lone'] > 0)[~whole].any(), '(ii) in a truncated event')
    lone = base & (has & whole)[:, None] & (np.abs(ct) >= COS_AXIS)
    for i in np.where(lone.any(1))[0]:
        s, miss, pm = _balance(P[i:i + 1], M[i:i + 1])
        n_ii = 0
        for j in np.where(lone[i])[0]:
            E = f[i, j, 0] * s[0]
            ii = miss[0] < P_MISS and E >= pm[0].max() and _hemi_pt(P[i], M[i], j, pm[0]) < PT_REL * E
            _need(not (ii and flag[i, j]), f'photon (ii) should unflag kept, event {i}')
            n_ii += ii
        _need(n_ii == info['n_lone'][i], f'(ii) count, event {i}')
    # 5. no flag left further than C_GAP from two or more agreeing flags (the values with the event's
    #    own s wherever they could disagree at all)
    fl = flag & has[:, None]
    c = _implied_cos(ct, f[..., 2], e)
    spread = np.where(fl, c, -np.inf).max(1) - np.where(fl, c, np.inf).min(1)
    shift = 1e-12 / np.where(fl, e, np.inf).min(1)
    for i in np.where((fl.sum(1) >= 2) & (spread + shift > C_TOL))[0]:
        j = np.where(fl[i])[0]
        es = f[i, j, 0] * _balance(P[i:i + 1], M[i:i + 1])[0][0]
        cj, tj = _implied_cos(ct[i, j], f[i, j, 2], es), np.where(es < E_SOFT, C_TOL_SOFT, C_TOL)
        near = np.abs(cj[:, None] - cj[None, :]) <= np.maximum(tj[:, None], tj[None, :])
        for a in range(len(j)):
            w = np.where(np.abs(cj - cj[a]) > C_GAP)[0]
            _need(not any(near[b, d] for b in w for d in w if b < d), f'contradicted flag, event {i}')
    # 6. chunking does not matter, n_beam agrees
    k = min(N, nchk)
    _need(np.array_equal(beam_photon_mask(P[:k], M[:k], chunk=97), flag[:k]), 'chunk dependence')
    _need(np.array_equal(n_beam(P[:k], M[:k]), nb[:k]), 'n_beam')
    # per side: phi < 0 (+beam) and phi > 0 (-beam) for base flags; cos > 0 and < 0 for (i)
    neg = np.where(has, (flag & (f[..., 2] < 0)).sum(1), (flag & (ct > 0)).sum(1))
    pos = np.where(has, (flag & (f[..., 2] > 0)).sum(1), (flag & (ct < 0)).sum(1))
    return dict(rule_version=RULE_VERSION, events=N, with_beam_photon=int((nb > 0).sum()),
                n_hist=np.bincount(np.minimum(nb, 4), minlength=5).tolist(),
                base_photons=int(base.sum()), photons=int(flag.sum()),
                axis_rule_events=int(info['rule_i'].sum()), axis_rule_photons=int(ax.sum()),
                axis_rule_soft_photons=int(info['n_axis_soft'].sum()),
                lone_events=int((info['n_lone'] > 0).sum()),
                conflict_events=int((info['n_conflict'] > 0).sum()),
                conflict_dropped=int(info['n_conflict'].sum()),
                tie_events=int(info['tie'].sum()),
                whole_one_per_side=int((whole & (neg == 1) & (pos == 1)).sum()), whole=int(whole.sum()),
                energy_GeV=float((e * flag).sum() / N), truncated=int((~whole).sum()))


def check_file(path, nchk=3000):
    """check() on the arrays 'particles' and 'mask' of an npz file. Raises ValueError for a file
    that is not an npz file holding both."""
    import zipfile
    try:
        z = np.load(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        raise ValueError(f'cannot read it as an npz file ({e})') from None
    if not hasattr(z, 'files') or not {'particles', 'mask'} <= set(z.files):
        raise ValueError('needs the arrays particles and mask')
    return check(z['particles'], z['mask'].astype(bool), nchk)


def main(argv):
    """The command line: 0 when the rule passes its checks on every file, 1 at the first file that
    fails or cannot be read, 2 for a usage error. -h or --help prints this module's docstring."""
    paths = argv[1:]
    if '-h' in paths or '--help' in paths:
        print(__doc__)
        return 0
    if not paths or any(p.startswith('-') for p in paths):
        print('usage: python -m gentune.beam_photons FILE.npz [...], each with the arrays particles '
              'and mask, such as models/MIX17_selftest.npz (--help says more)')
        return 2
    for p in paths:
        try:
            s = check_file(p)
        except (RuleCheckError, ValueError) as e:
            print(f'{p}: FAIL, {e}')
            return 1
        print(f'{p}: OK, rule version {RULE_VERSION}')
        print('   ' + ', '.join(f'{k} {v}' for k, v in s.items()))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
