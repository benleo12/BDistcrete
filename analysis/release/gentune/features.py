"""Turn events into the per-particle features the trunk expects.

The model reads each event in its own THRUST FRAME, as a variable length set of particles
with six features each. Getting this construction right matters more than anything else in
the release: a different axis convention, a different azimuth reference or a different
particle selection changes the input and silently changes every weight.

The conventions, all of which must be reproduced exactly:

  particles    every final state particle of the WHOLE event except neutrinos, both
               hemispheres, ordered by descending energy fraction when truncating
  z            E_i / sum_j E_j, the energy fraction of the whole event
  cos_theta    p_hat_i . n, SIGNED, so the hemisphere is carried by the sign
  phi          azimuth around n, measured from e1 = n x z_hat normalized
  log10_mass   log10(max(m, 1e-3)) with m from the generator record, in GeV
  units        momenta and masses in GeV. parse_hepmc reads the file's units line and
               converts a MeV file, since read as GeV it shifts log10_mass by 3
  baryon       +1 for baryons, -1 for antibaryons, 0 otherwise
  charge       +1 or -1 for the |q| = 1 species listed below, 0 otherwise
  P_max        100 particles, padded with zeros and flagged by the mask

The thrust axis n is found by the standard iteration: start from the hardest particle and
repeatedly replace n by the normalized sum of sign(p_i . n) p_i until it stops moving. The
sign of n itself is not fixed by this, and it does not need to be: the features are invariant
under n -> -n up to the sign of cos_theta and a reflection in phi, and the trained model saw
the same convention on both sides of every event.

A stable-hadron convention question sits underneath all of this. Sherpa keeps hadrons with
c*tau above 10 mm as final state particles and Herwig decays K0S, Lambda, Sigma, Xi and Omega
by default, so the two generators do not agree on what a final state particle IS unless you
say so. The released samples were all produced with the Sherpa convention, which in Herwig
means setting DecayHandler:MaxLifeTime to 10*mm. If your sample uses the other convention the
particle content differs and the weights are not the ones the model was trained to give.
"""
import gzip
import warnings
import numpy as np

NEUTRINO = {12, 14, 16}
# How many of each momentum unit a HepMC units line can name make one GeV. Momenta are divided
# by it, which rounds once, where multiplying by 1e-3 would round twice.
UNITS = {'GEV': 1.0, 'MEV': 1000.0}
PION_MASS = 0.13957        # GeV, the charged pion, which every generator writes on shell
# The |charge| = 1 species that appear in these samples. Everything else, which is the
# neutrals (photon, pi0, K0, neutron, Lambda, ...), takes charge 0.
CHG1 = {211, 321, 2212, 3222, 3112, 3312, 3334, 11, 13, 15, 411, 431, 521}
P_MAX = 100
N_FEATURES = 6


def charge(pid):
    a = abs(pid)
    return (1 if pid > 0 else -1) if a in CHG1 else 0


def baryon_number(pid):
    a = abs(pid)
    return (1 if pid > 0 else -1) if 1000 <= a < 1000000 else 0


def thrust_axis(p3, iters=20):
    """The axis the heads were TRAINED with, kept deliberately.

    This is the single-seed sign iteration, seeded from the hardest particle. It converges to a
    local maximum in about 4 percent of Z-pole events, which was found on 2026-09-28, and the
    exact algorithm now lives in generators/compute_efps.py for the event-shape observables. It
    is NOT used here on purpose: the per-particle features (cos theta signed against the axis,
    phi around it) that every released head was trained on were built with this finder, and a
    head reads events only through those features. Changing the frame for a user's events
    without retraining the head would mismatch the two. When the heads are retrained on
    regenerated samples this function and the training code change together.
    """
    # original docstring:
    # The thrust axis of one event, by the standard sign-flip iteration.

    if len(p3) == 0:
        return np.array([0.0, 0.0, 1.0])
    mags = np.linalg.norm(p3, axis=1)
    j = int(np.argmax(mags))
    n = p3[j]/(mags[j] + 1e-12)
    for _ in range(iters):
        new = (np.sign(p3 @ n)[:, None]*p3).sum(0)
        nn = np.linalg.norm(new)
        if nn < 1e-12:
            break
        new = new/nn
        if float(new @ n) > 1 - 1e-12:
            break
        n = new
    return n


def event_features(event, max_particles=P_MAX):
    """(features, mask, n_baryons) for one event.

    event is an (n, 6) array of rows [E, px, py, pz, pid, mass], already restricted to the
    final state particles you want the model to see."""
    out = np.zeros((max_particles, N_FEATURES), np.float32)
    mask = np.zeros(max_particles, bool)
    event = np.asarray(event, np.float64)
    if event.shape[0] == 0:
        return out, mask, 0
    p3 = event[:, 1:4]
    n = thrust_axis(p3)
    E = event[:, 0]
    z = E/max(E.sum(), 1e-12)
    p_hat = p3/(np.linalg.norm(p3, axis=1, keepdims=True) + 1e-12)
    cos_theta = p_hat @ n
    e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-6:                 # n along z, any perpendicular will do
        e1 = np.array([1.0, 0.0, 0.0])
    e1 = e1/np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    phi = np.arctan2(p_hat @ e2, p_hat @ e1)
    pid = event[:, 4].astype(int)
    bn = np.array([baryon_number(x) for x in pid], np.float32)
    ch = np.array([charge(x) for x in pid], np.float32)
    feats = np.stack([z, cos_theta, phi, np.log10(np.maximum(event[:, 5], 1e-3)), bn, ch],
                     1).astype(np.float32)
    keep = np.argsort(-z)[:min(len(z), max_particles)]     # keep the hardest if truncating
    out[:len(keep)] = feats[keep]
    mask[:len(keep)] = True
    return out, mask, int((np.abs(bn) > 0).sum())


def features_from_events(events, max_particles=P_MAX):
    """Stack event_features over a list of events into (features, mask, n_baryons)."""
    F = np.zeros((len(events), max_particles, N_FEATURES), np.float32)
    M = np.zeros((len(events), max_particles), bool)
    NB = np.zeros(len(events), np.int32)
    for i, ev in enumerate(events):
        F[i], M[i], NB[i] = event_features(ev, max_particles)
    return F, M, NB


def trunk_features(feats):
    """The seven columns the trunk actually reads, built from the six stored ones.

    The model does not see z and the mass column directly. It sees cos_theta, phi, log z and
    log sin^2(theta), which is the standard soft and collinear parametrization, plus log10 of
    the mass and the two species tags. Keeping this in one place is deliberate, because a
    reordering here would be invisible and would corrupt every weight."""
    f = np.asarray(feats, np.float32)
    eps = 1e-6
    z, cos, phi = f[..., 0], f[..., 1], f[..., 2]
    cols = [cos, phi,
            np.log(np.clip(z, eps, None)),
            np.log(np.clip(1 - cos**2 + eps, eps, None)),
            f[..., 3], f[..., 4], f[..., 5]]
    return np.stack(cols, -1).astype(np.float32)


def _hepmc_flavour(path, opener):
    """'hepmc2' or 'hepmc3', read off the file's own header.

    The START_EVENT_LISTING line names the format and the HepMC::Version line names the library
    that wrote it, which is not the same thing: Herwig 7.3 writes HepMC2 through HepMC3, so its
    files open with 'HepMC::Version 3.03.01' followed by 'HepMC::IO_GenEvent-START_EVENT_LISTING'.
    Reading the version line first took every such file for HepMC3 and found no particles in it,
    so the version line now decides only when there is no listing line."""
    version = None
    with opener(path, 'rt') as fh:
        for _ in range(40):
            line = fh.readline()
            if not line:
                break
            if 'Asciiv3' in line:
                return 'hepmc3'
            if 'IO_GenEvent' in line:
                return 'hepmc2'
            if 'HepMC::Version 3' in line:
                version = 'hepmc3'
            elif 'HepMC::Version 2' in line:
                version = 'hepmc2'
            if line.startswith('E '):
                break
    return version


def _hepmc2_weight(p):
    """The first weight on a split HepMC2 E line, or None when the line does not have the layout
    E n mpi scale a_qcd a_qed process vertex n_vertices beam1 beam2 n_rng rng.. n_w w.., which is
    checked by asking the two counts to account for every field on the line."""
    try:
        k = 12 + int(p[11])
        n = int(p[k])
        return float(p[k + 1]) if n > 0 and len(p) == k + 1 + n else None
    except (ValueError, IndexError):
        return None


def _check_units(path, events, unit, source):
    """Raise unless the parsed momenta are in GeV, judged by the charged pion mass.

    Every generator writes final state pions on shell, so their median mass is 0.1396 in GeV and
    139.6 in MeV, a factor of a thousand apart, and a factor of ten either way is a safe margin.
    An event record without pion masses falls back on the event energy, below 10 TeV at any
    electron-positron collider and far above it for a Z-pole event read in MeV."""
    rows = np.concatenate(events[:1000])
    m = rows[np.abs(rows[:, 4]) == 211, 5]
    m = m[m > 0]
    if len(m):
        mpi = float(np.median(m))
        if 0.1 < mpi/PION_MASS < 10:
            return
        what = f'the charged pions come out at a median mass of {mpi:.6g} GeV, not {PION_MASS}'
    else:
        e = float(np.median([ev[:, 0].sum() for ev in events[:1000]]))
        if 1.0 < e < 1e4:
            return
        what = f'the median event energy comes out at {e:.6g} GeV'
    if source == 'assumed':
        hint = ('The file has no units line, so GeV was assumed. Pass units="MEV" if it is in '
                'MeV, or add a "U MEV MM" line after each E line.')
    else:
        hint = f'Pass units="GEV" or units="MEV" if the {source} is wrong.'
    raise ValueError(f'{path}: read in {unit}, {what}. {hint}')


def parse_hepmc(path, max_events=None, fmt='auto', units=None):
    """Final state, non-neutrino particles per event from a HepMC ASCII file, gzip allowed.

    Returns an Events list, one array per event in the order of the file, with the event numbers
    of the file in its `numbers`. An event with no final-state particle other than neutrinos is
    left out, which an e+e- -> hadrons sample never has, so event i is the i-th of the file and
    numbers[i] says which it is in any case. Every weight gentune computes from these events
    follows the same order.

    Rows are [E, px, py, pz, pid, mass], in GeV. HepMC2 and HepMC3 put the particle id and the
    status in different columns of the P line, so the format has to be known. fmt='auto' reads it
    from the file's own header, which is what you want. Pass 'hepmc2' or 'hepmc3' to force it, and
    'herwig' is kept as an alias for 'hepmc2' because that is what it always meant.

    Units. Both formats state them on a line 'U GEV MM' or 'U MEV MM' after each E line, which
    HepMC 2.06 and HepMC3 always write. That line is read and a MeV file is converted to GeV. It
    used to be ignored, so a MeV file reached the trunk with every mass a thousand times too large
    and gave wrong weights with nothing to say so. A file without the line is read as GeV, the
    HepMC3 default and what Herwig and Sherpa write, and units='GEV' or 'MEV' overrides the file
    either way. The result is then checked: the charged pions must come out at their mass, 0.1396
    GeV, within a factor of ten, or ValueError is raised. Converting on that evidence alone would
    also have been possible. It is not done, because a factor of a thousand guessed for a file
    that does not state its units is the same kind of silent choice the units line replaces, so
    the check only refuses and the caller says what the file is.

    Weights. The event weights, at the end of a HepMC2 E line or on a HepMC3 W line, are ignored
    and every event counts once, as in the training samples, which are unweighted. A file whose
    weights are all equal, to one or to anything else, is unweighted too and needs nothing. When
    they differ between events a warning is issued, once per file, because the weights gentune
    returns are then relative to the file's own and must be multiplied by them, event by event,
    and renormalized.

    Raises ValueError rather than returning an empty list, because the failure this replaces was
    silent: a HepMC2 file has more than ten fields on its P lines, so it took the HepMC3 branch,
    read the momentum as the particle id, matched no status and yielded nothing at all.
    """
    opener = gzip.open if str(path).endswith('.gz') else open
    if fmt in ('auto', None):
        fmt = _hepmc_flavour(path, opener)
        if fmt is None:
            raise ValueError(
                f'{path}: cannot tell HepMC2 from HepMC3 out of the header, so the P line layout '
                f'is unknown. Pass fmt="hepmc2" or fmt="hepmc3" explicitly.')
    if fmt == 'herwig':
        fmt = 'hepmc2'
    if fmt not in ('hepmc2', 'hepmc3'):
        raise ValueError(f'fmt={fmt!r} is not one of auto, hepmc2, hepmc3, herwig')
    forced = None if units is None else str(units).upper()
    if forced is not None and forced not in UNITS:
        raise ValueError(f'units={units!r} is not one of GEV, MEV')
    # HepMC2: P barcode pdg px py pz E m status ...
    # HepMC3: P id vertex pdg px py pz E m status
    i_pid, i_stat, i_mom = (2, 8, 3) if fmt == 'hepmc2' else (3, 9, 4)
    need = i_stat + 1
    scale = UNITS[forced] if forced else 1.0
    stated = None                         # the unit the file's own units line names, if any
    events, cur, seen, wts, in_event = [], [], 0, [], False
    numbers, number = [], -1
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if cur:
                    events.append(np.asarray(cur, np.float64))
                    numbers.append(number)
                    seen += len(cur)
                    cur = []
                    if max_events is not None and len(events) >= max_events:
                        break
                in_event = True
                e = line.split()
                try:
                    number = int(e[1])          # the event number, second field in both formats
                except (ValueError, IndexError):
                    number = -1
                if fmt == 'hepmc2':
                    w = _hepmc2_weight(e)
                    if w is not None:
                        wts.append(w)
                continue
            if line.startswith('U '):
                u = (line.split() + [''])[1].upper()
                if u not in UNITS:
                    raise ValueError(f'{path}: the units line {line.strip()!r} names a momentum '
                                     f'unit other than GEV or MEV')
                stated = u
                if forced is None:
                    scale = UNITS[u]
                continue
            # a W line before the first event is the HepMC3 header naming the weights
            if line.startswith('W ') and in_event and fmt == 'hepmc3':
                try:
                    wts.append(float(line.split()[1]))
                except (ValueError, IndexError):
                    pass
                continue
            if not line.startswith('P '):
                continue
            p = line.split()
            if len(p) < need:
                continue
            try:
                pid, status = int(p[i_pid]), int(p[i_stat])
                px, py, pz, E, m = (float(p[i_mom]), float(p[i_mom + 1]), float(p[i_mom + 2]),
                                    float(p[i_mom + 3]), float(p[i_mom + 4]))
                if status == 1 and abs(pid) not in NEUTRINO:
                    cur.append([E/scale, px/scale, py/scale, pz/scale, pid, m/scale])
            except (ValueError, IndexError):
                continue
    if cur:
        events.append(np.asarray(cur, np.float64))
        numbers.append(number)
        seen += len(cur)
    if not seen:
        raise ValueError(
            f'{path}: parsed as {fmt} and found no final-state particles. Either the file holds '
            f'no events, or its P line layout is not the one {fmt} implies. Try the other fmt.')
    source = 'units argument' if forced else ('units line' if stated else 'assumed')
    _check_units(path, events, forced or stated or 'GEV', source)
    w = np.asarray(wts, np.float64)
    if len(w) > 1 and not np.allclose(w, w[0], rtol=1e-9, atol=0):
        warnings.warn(
            f'{path}: the event weights differ between events, from {w.min():.6g} to '
            f'{w.max():.6g}. parse_hepmc ignores them and counts every event once, as in the '
            f'training samples, so the weights gentune returns for these events must be '
            f'multiplied by the file\'s own, event by event, and renormalized. Or unweight the '
            f'sample first.', stacklevel=2)
    return Events(events, numbers)


class Events(list):
    """What parse_hepmc returns: a list with one (n, 6) array per event, in the order of the file,
    and `numbers`, the event number each one carries on its E line in the file, as an int64 array
    (-1 where the line has none). Weights computed from these events follow the same order, so
    weight i belongs to the event numbered numbers[i]."""

    def __init__(self, events=(), numbers=()):
        super().__init__(events)
        self.numbers = np.asarray(numbers, np.int64)
