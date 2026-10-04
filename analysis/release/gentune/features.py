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
import numpy as np

NEUTRINO = {12, 14, 16}
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
    """'hepmc2' or 'hepmc3', read off the file's own header."""
    with opener(path, 'rt') as fh:
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


def parse_hepmc(path, max_events=None, fmt='auto'):
    """Final state, non-neutrino particles per event from a HepMC ASCII file, gzip allowed.

    Rows are [E, px, py, pz, pid, mass]. HepMC2 and HepMC3 put the particle id and the status
    in different columns of the P line, so the format has to be known. fmt='auto' reads it from
    the file's own header, which is what you want. Pass 'hepmc2' or 'hepmc3' to force it, and
    'herwig' is kept as an alias for 'hepmc2' because that is what it always meant.

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
    # HepMC2: P barcode pdg px py pz E m status ...
    # HepMC3: P id vertex pdg px py pz E m status
    i_pid, i_stat, i_mom = (2, 8, 3) if fmt == 'hepmc2' else (3, 9, 4)
    need = i_stat + 1
    events, cur, seen = [], [], 0
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if cur:
                    events.append(np.asarray(cur, np.float64))
                    seen += len(cur)
                    cur = []
                    if max_events is not None and len(events) >= max_events:
                        return events
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
                    cur.append([E, px, py, pz, pid, m])
            except (ValueError, IndexError):
                continue
    if cur:
        events.append(np.asarray(cur, np.float64))
        seen += len(cur)
    if not seen:
        raise ValueError(
            f'{path}: parsed as {fmt} and found no final-state particles. Either the file holds '
            f'no events, or its P line layout is not the one {fmt} implies. Try the other fmt.')
    return events
