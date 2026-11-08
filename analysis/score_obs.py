"""The scoring side of the rerun: observables without the beam photons (D4), input checks, and the
refitted calibration constants (D11). Shared by r2_ladder.py and every closure, learning-curve,
calibration and learning-band script.

D4 at score time. BEAM_PHOTONS=keep, the default, reproduces the published scoring: every function
here then returns exactly what the published scripts computed. BEAM_PHOTONS=remove takes the beam
photons (beam_photons.py) out of what is computed at score time from all kept particles, on both
sides of every comparison (the held-out run and the reweighted reference):
  strange_frac  kaons over all kept particles: the beam photons leave the denominator (they are
                massless, so never in the numerator)
  mult_total    the multiplicity of all kept particles. The shapes files count every final-state
                particle but the neutrinos, so the beam photons among the kept ones are subtracted.
                An event with more particles than the file keeps (the hardest 100) may have lost
                its soft beam photons from the file; there nothing is subtracted for them.
  kept          the particles that enter the energy-fraction spectrum x_p, or any other sum over
                all kept particles
The event shapes 1-T, B_total and rho_heavy come photon-free from the shapes files of the _nb
folders and from refreshed reference bundles. check_csv and check_ref_shapes stop a script that
would read the old ones, and check_csv also stops BEAM_PHOTONS=keep from reading photon-free shapes.
A cache of a bundle's particle observables (POBS_DIR) records the md5 of beam_photons.py, and one
made with another rule stops the script; describe() gives the md5 to every result under remove. Under
remove a refreshed bundle must also be listed in RUN/output/bundle_rule.json with the beam_photons.py
in use (bundle_rule.check, I2), wherever reference_obs or check_ref_shapes reads one.

D11. With T_DIR set, temperature(key) is the refitted calibration constant of the model `key`,
from T_DIR/<key>.json (r2_ladder.py with LADDER_SCORE_T=refit or recalibrate_T.py writes it), and
check_export_T makes a script fail when an export does not carry it. Unset, nothing changes.
"""
import os, re, json
import numpy as np

MODE = os.environ.get('BEAM_PHOTONS') or 'keep'          # unset or empty: as published
if MODE not in ('keep', 'remove'):
    raise SystemExit(f'BEAM_PHOTONS={MODE!r}: use keep (as published) or remove (D4)')
REMOVE = MODE == 'remove'
# a shapes file is photon-free when it resolves into a folder whose name ends in _nb (the mirrors
# of Phase A, gen/data_*_nb and maxent_code/data_*_nb)
NB_DIR = re.compile(os.environ.get('PHOTON_FREE_DIR_RE', r'_nb\d*$'))
ALLOW_MIXED = os.environ.get('BEAM_PHOTONS_ALLOW_MIXED', '') == '1'
SHAPES = ('1_minus_thrust', 'B_total', 'rho_heavy')
# the mean of a reference sample's photon-free shape against the old one, (lowest, highest)
# relative change accepted: Phase A measured -0.85 to -1.23% (1-T), -0.41 to -0.57% (B_total) and
# -0.55 to -0.82% (rho_heavy) over the paper's samples
SHIFT = {'1_minus_thrust': (-0.03, -0.004), 'B_total': (-0.02, -0.002), 'rho_heavy': (-0.03, -0.002)}
T_DIR = os.environ.get('T_DIR', '')


def describe():
    d = dict(beam_photons=MODE, t_dir=T_DIR or None)
    if REMOVE:
        d['beam_photons_md5'] = beam_photons_md5()
    return d


def need(*paths):
    """Fail at once, naming the file, when an input is missing."""
    for p in paths:
        if not os.path.exists(p):
            extra = f' (a link to {os.path.realpath(p)})' if os.path.islink(p) else ''
            raise FileNotFoundError(f'missing input {p}{extra}')
    return paths[0] if len(paths) == 1 else paths


def photon_free(path):
    """True when the file resolves into a photon-free (_nb) folder."""
    return any(NB_DIR.search(part) for part in os.path.realpath(path).split(os.sep)[:-1])


def check_csv(path):
    """A shapes file must be photon-free under remove, and the original under keep."""
    need(path)
    nb = photon_free(path)
    if REMOVE and not nb:
        raise RuntimeError(f'BEAM_PHOTONS=remove, but {path} resolves to {os.path.realpath(path)}, '
                           f'which is not in a photon-free (_nb) folder')
    if not REMOVE and nb and not ALLOW_MIXED:
        raise RuntimeError(f'BEAM_PHOTONS=keep, but {path} is photon-free ({os.path.realpath(path)}): its '
                           f'shapes would be photon-free and the multiplicity and the kaon fraction not')
    return path


def read_shapes(path, usecols=None):
    import pandas as pd
    return pd.read_csv(check_csv(path), usecols=usecols)


# ---- the beam photons ----------------------------------------------------------------------------
_BP = None


def _bp():
    global _BP
    if _BP is None:
        try:
            import beam_photons as m
        except ImportError as e:
            raise SystemExit('BEAM_PHOTONS=remove needs beam_photons.py next to the scripts') from e
        _BP = m
    return _BP


def check_rule(path):
    """Under remove, a refreshed bundle must be listed in RUN/output/bundle_rule.json, made with the
    beam_photons.py in use (bundle_rule.py, I2); under keep nothing is checked."""
    if not REMOVE:
        return
    try:
        import bundle_rule
    except ImportError as e:
        raise SystemExit('BEAM_PHOTONS=remove needs bundle_rule.py next to the scripts') from e
    bundle_rule.check(path)


def beam_photons_md5():
    """The md5 of the beam_photons.py in use, read now: the rule a result or a cache was made with."""
    import hashlib
    with open(_bp().__file__, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def beam_mask(P, M):
    """(N, Pmax) bool, the beam photons among the kept particles"""
    return _bp().beam_photon_mask(P, np.asarray(M, bool))


def n_beam(P, M):
    """(N,) the number of beam photons among the kept particles of each event"""
    return _bp().n_beam(P, np.asarray(M, bool))


def kept(P, M):
    """The particles that enter an observable summed over all kept particles (x_p)."""
    M = np.asarray(M, bool)
    return M & ~beam_mask(P, M) if REMOVE else M


def strange_frac(p, mask, nb=None):
    """Kaons (log10 m in (-0.36, -0.20)) over all kept particles, as r2_ladder.strange_frac; under
    remove without the beam photons in the denominator. nb: n_beam(p, mask) if already known."""
    lm = p[..., 3]; m = mask.astype(bool); ink = ((lm > -0.36) & (lm < -0.20)) & m
    den = m.sum(1)
    if REMOVE:
        den = den - (n_beam(p, m) if nb is None else nb)
    return ink.sum(1) / np.maximum(den, 1)


def mult_total(mult, P, M, where='', nb=None):
    """The multiplicity of all kept particles. `mult` is a shapes file's (or a bundle's) mult_total
    for the same events as the particle arrays P, M. Under keep it is returned as it is. Under
    remove the beam photons are subtracted, after checking on the events the file keeps whole that
    `mult` counts them (it equals the kept count there); a `mult` that already leaves them out (it
    equals the kept count less the beam photons) is returned unchanged; anything else fails."""
    mult = np.asarray(mult)
    if not REMOVE:
        return mult
    M = np.asarray(M, bool)
    k = M.sum(1)
    nb = n_beam(P, M) if nb is None else np.asarray(nb)
    # events stored whole; below two particles the shapes code writes 0
    full = (k < M.shape[1]) & (k >= 2)
    d = np.rint(mult[full]).astype(np.int64) - k[full]
    if not np.any(d):
        return mult - nb
    if np.array_equal(d, -nb[full]):
        return mult
    bad = int(np.count_nonzero((d != 0) & (d != -nb[full])))
    raise RuntimeError(f'{where}: mult_total equals neither the kept count nor the kept count without the '
                       f'beam photons in {bad} of {int(full.sum())} events stored whole (wrong event order?)')


# ---- reference bundles ---------------------------------------------------------------------------
def particle_obs(path, cache=None, chunk=200000):
    """What a reference bundle's particles give per event: the kept count, the beam photons, the
    kaon fraction with and without them, the charged count and the baryon count (for the event-order
    checks). `path` holds 'particles' and 'mask'. `cache`, if given, stores the result (an npz) with
    the md5 of beam_photons.py, and is reused when it is newer than `path` and was made from the same
    file; one made with another beam_photons.py (or before the md5 was recorded) stops the script."""
    need(path)
    src = os.path.realpath(path)
    rule = beam_photons_md5()
    if cache and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        c = np.load(cache)
        if str(c['source']) == src:
            made = str(c['beam_photons_md5']) if 'beam_photons_md5' in c.files else 'not recorded'
            if made != rule:
                raise RuntimeError(f'{cache} was made with beam_photons.py md5 {made}, but {_bp().__file__} has '
                                   f'md5 {rule}: remove the cache, and redo the results made with the other rule')
            return {k: c[k] for k in c.files if k not in ('source', 'beam_photons_md5')}
    z = np.load(path)
    if 'particles' not in z.files:
        raise RuntimeError(f'{path} holds no particles; give the bundle with the particles of the same events')
    P = z['particles']; M = z['mask'].astype(bool)
    out = {k: [] for k in ('n_kept', 'n_beam', 'strange_all', 'strange_nb', 'nch')}
    for a in range(0, len(M), chunk):
        p, m = P[a:a + chunk], M[a:a + chunk]
        nb = _bp().n_beam(p, m)
        lm = p[..., 3]; ink = (((lm > -0.36) & (lm < -0.20)) & m).sum(1); k = m.sum(1)
        out['n_kept'].append(k.astype(np.int16)); out['n_beam'].append(nb.astype(np.int16))
        out['strange_all'].append(ink / np.maximum(k, 1)); out['strange_nb'].append(ink / np.maximum(k - nb, 1))
        out['nch'].append(((p[..., 5] != 0) & m).sum(1).astype(np.int16))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['pmax'] = np.array(M.shape[1])
    if 'nbaryon' in z.files:
        out['nbaryon'] = np.asarray(z['nbaryon'])
    if cache:
        tmp = cache[:-4] + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, source=np.array(src), beam_photons_md5=np.array(rule), **out)
        os.replace(tmp, cache)
    return out


def reference_obs(bundle, obs, particles=None, orig=None, cache=None):
    """The observables `obs` of a reference bundle, {obs: float64 array}. Under keep, the stored
    arrays, as every published script read them. Under remove: the three event shapes from the
    bundle after check_ref_shapes against the old bundle `orig`; mult_total and the kaon fraction
    from the particles of the same events (`particles`, default the bundle itself), with the event
    order checked against the bundle's own arrays; the baryon count as stored."""
    need(bundle)
    check_rule(bundle)
    z = np.load(bundle)
    if not REMOVE:
        return {o: np.asarray(z[o], np.float64) for o in obs}
    po = particle_obs(particles or bundle, cache)
    n = len(po['n_beam'])
    out = {}
    for o in obs:
        if o == 'strange':
            out[o] = po['strange_nb'].astype(np.float64)
            if 'strange' in z.files:
                s = np.asarray(z['strange'], np.float64)
                if len(s) != n or not (np.allclose(s, po['strange_nb'], rtol=0, atol=1e-6)
                                       or np.allclose(s, po['strange_all'], rtol=0, atol=1e-6)):
                    raise RuntimeError(f'{bundle}: the stored kaon fraction is not that of the particles of '
                                       f'{particles or bundle} (wrong event order?)')
        elif o == 'mult_total':
            out[o] = np.asarray(_mult_from_counts(np.asarray(z['mult_total']), po, bundle), np.float64)
        else:
            out[o] = np.asarray(z[o], np.float64)
            if len(out[o]) != n:
                raise RuntimeError(f'{bundle}: {len(out[o])} events in {o}, {n} in the particles')
    for k in ('nbaryon', 'nch'):
        if k in z.files and k in po and not np.array_equal(np.rint(np.asarray(z[k], np.float64)),
                                                           np.rint(np.asarray(po[k], np.float64))):
            raise RuntimeError(f'{bundle}: {k} differs from that of {particles or bundle} (wrong event order)')
    check_ref_shapes(bundle, {o: out[o] for o in obs if o in SHAPES}, orig, po['n_beam'])
    return out


def _mult_from_counts(mult, po, where):
    """mult_total of a bundle without its particles at hand: as mult_total() from the counts."""
    k = po['n_kept'].astype(np.int64); nb = po['n_beam'].astype(np.int64)
    full = (k < int(po['pmax'])) & (k >= 2)
    d = np.rint(mult[full]).astype(np.int64) - k[full]
    if not np.any(d):
        return mult - nb
    if np.array_equal(d, -nb[full]):
        return mult
    raise RuntimeError(f'{where}: mult_total equals neither the kept count nor the kept count without the beam '
                       f'photons in {int(np.count_nonzero((d != 0) & (d != -nb[full])))} events (wrong event order?)')


def check_ref_shapes(path, values, orig, nb=None):
    """Under remove, the event shapes `values` ({obs: array}) of the bundle at `path` must be the
    photon-free ones: the bundle is not the old file `orig` nor a link to it, its means have moved
    against `orig` (the same events) by the shift the beam photons make, and the events without a
    beam photon (nb == 0) are unchanged but for a fraction 2e-3 (the rule of the shapes files and of
    beam_photons.py may differ in rare events)."""
    if not REMOVE or not values:
        return
    check_rule(path)
    if not orig:
        raise RuntimeError(f'{path}: give the old bundle of the same events (REF_ORIG) to check its shapes')
    need(orig)
    if os.path.realpath(path) == os.path.realpath(orig):
        raise RuntimeError(f'{path} is the old bundle {orig}: its shapes include the beam photons')
    o = np.load(orig)
    for k, new in values.items():
        old = np.asarray(o[k], np.float64)
        if len(old) != len(new):
            raise RuntimeError(f'{path}: {len(new)} events in {k}, {len(old)} in {orig}')
        r = float(new.mean() / old.mean() - 1)
        lo, hi = SHIFT[k]
        if not lo <= r <= hi:
            raise RuntimeError(f'{path}: <{k}> differs from {orig} by {100*r:+.3f}%, the beam photons move it by '
                               f'{100*lo:.1f} to {100*hi:.1f}%: not the photon-free shapes')
        if nb is not None:
            same = np.asarray(nb) == 0
            f = float(np.mean(new[same] != old[same])) if same.any() else 0.0
            if f > 2e-3:
                raise RuntimeError(f'{path}: {k} changed in {100*f:.2f}% of the events without a beam photon')
    print(f'[score_obs] {path}: photon-free shapes ({", ".join(values)}) checked against {orig}', flush=True)


# ---- the refitted calibration constants (D11) -----------------------------------------------------
def model_key(path):
    """The model a file belongs to: the name of the file it resolves to, without its suffix, so the
    links of one network (MIXP2v3fullfac0 -> MIXGEOV3S9, <tag>h400 -> <tag>) share one key."""
    b = os.path.basename(os.path.realpath(path))
    for s in ('_cond.npz', '_trunk.pt', '_concat.pt'):
        if b.endswith(s):
            return b[:-len(s)]
    raise ValueError(f'{path}: not a model file (_cond.npz, _trunk.pt or _concat.pt)')


def temperature(key, shipped=None):
    """The refitted temperature of model `key` from T_DIR (required when T_DIR is set), else `shipped`."""
    if not T_DIR:
        return shipped
    f = os.path.join(T_DIR, f'{key}.json')
    if not os.path.exists(f):
        raise FileNotFoundError(f'no refitted temperature for {key}: {f} is missing (D11: refit it first)')
    return float(json.load(open(f))['T'])


def check_export_T(export, T_export):
    """With T_DIR set, the export must carry the refitted temperature of its model, so every reader of
    it (including directlib) uses the same value. Returns the temperature to use."""
    if not T_DIR:
        return T_export
    T = temperature(model_key(export))
    if abs(T - float(T_export)) > 1e-12:
        raise RuntimeError(f'{export} carries T={T_export}, but the refit of {model_key(export)} is {T}: '
                           f'write it into the export first (apply_T.py --lmx)')
    return T


def write_temperature(key, rec):
    """Write T_DIR/<key>.json atomically (rec holds 'T' and its provenance)."""
    if not T_DIR:
        raise RuntimeError('T_DIR is not set')
    os.makedirs(T_DIR, exist_ok=True)
    f = os.path.join(T_DIR, f'{key}.json'); tmp = f + f'.tmp{os.getpid()}'
    json.dump(dict(model=key, **rec), open(tmp, 'w'), indent=1)
    os.replace(tmp, f)
    print(f'[score_obs] wrote {f}: T = {rec["T"]}', flush=True)
    return f
