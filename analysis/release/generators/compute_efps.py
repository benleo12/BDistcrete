#!/usr/bin/env python3
"""
Compute Energy Flow Polynomials and event shapes from HepMC3 ASCII files.

Parses final-state particle 4-momenta, splits events into thrust-axis
hemispheres, and evaluates hemisphere EFPs (degree <= 4, 36 of them) plus standard
event shapes (thrust, C-parameter, heavy jet mass, wide broadening).

The photons radiated by the beams are left out of every shape and every EFP, and out of the
multiplicities, as in the paper's observables. Before anything is computed, each event loses the
photons that the paper's rule flags (gentune/beam_photons.py of the release). The rule
reads the per-particle features that extract_particles_full_flavor.py stores for the event, the
100 hardest particles in the frame of the single-seed thrust iteration, so the event's features
are built here in the same way first (remove_beam_photons). mult_total is then the number of the
other final-state particles. --beam-photons keep counts the beam photons everywhere, as the shape
files of an earlier draft did. The paper made its photon-free values from the stored features of
its runs, as the value with the beam photons minus the change their removal makes on momenta
rebuilt from those features; this script removes them from the event record itself, which those
values follow to first order in the removed momentum.

Usage:
    # Single run
    python compute_efps.py --input runs/run_000/events.hepmc --output-dir data --run-id 0

    # Batch over many runs
    python compute_efps.py --batch --runs-dir runs --output-dir data
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from energyflow import efp
except ImportError:
    efp = None

HERE = os.path.dirname(os.path.abspath(__file__))
# The particles the rule reads per event: the --max-particles of extract_particles_full_flavor.py,
# with which every training sample was extracted.
P_MAX = 100


# ---------------------------------------------------------------------------
#  HepMC3 ASCII parser
# ---------------------------------------------------------------------------

def _hepmc_flavour(filepath, opener):
    """'hepmc2' or 'hepmc3', read off the file's own header, or None if it does not say.

    The listing line names the format and the HepMC::Version line names the library that wrote
    it, which is not the same thing: Herwig 7.3 writes HepMC2 through HepMC3, so its files open
    with 'HepMC::Version 3.03.01' followed by 'HepMC::IO_GenEvent-START_EVENT_LISTING'. The
    version line decides only when there is no listing line, as in gentune/features.py."""
    version = None
    with opener(filepath, "rt") as fh:
        for _ in range(40):
            line = fh.readline()
            if not line:
                break
            if "Asciiv3" in line:
                return "hepmc3"
            if "IO_GenEvent" in line:
                return "hepmc2"
            if "HepMC::Version 3" in line:
                version = "hepmc3"
            elif "HepMC::Version 2" in line:
                version = "hepmc2"
            if line.startswith("E "):
                break
    return version


def parse_hepmc3(filepath, max_events=None, fmt='auto', with_id=False):
    """
    Parse a HepMC ASCII file (plain or gzipped) and return a list of events.

    Each event is an (N, 4) numpy array with columns [E, px, py, pz] for
    every final-state particle (status == 1), excluding neutrinos. with_id=True adds the
    particle id and the generated mass, [E, px, py, pz, pid, mass], the rows of
    extract_particles_full_flavor.parse_hepmc, which the beam-photon rule needs.

    fmt='auto' reads HepMC2 against HepMC3 from the file header. It used to pick the layout by
    counting fields on the P line, taking pid at index 3 and status at 9 whenever there were ten
    or more, and a HepMC2 P line carries twelve or thirteen, so an auto-detected HepMC2 file had
    its momentum read as the particle id, matched no status and returned an empty list with no
    error. The two branches were also labelled the wrong way round. 'herwig' is an alias for
    'hepmc2', which is what it always meant.

    Raises ValueError rather than returning an empty list, because compute_shapes_only.py writes
    its CSV from whatever this returns and would otherwise write an empty table and exit 0.
    """
    filepath = str(filepath)
    opener = gzip.open if filepath.endswith(".gz") else open
    if fmt in ("auto", None):
        fmt = _hepmc_flavour(filepath, opener)
        if fmt is None:
            raise ValueError(f"{filepath}: the header does not say whether this is HepMC2 or "
                             f"HepMC3. Pass fmt='hepmc2' or fmt='hepmc3'.")
    if fmt == "herwig":
        fmt = "hepmc2"
    if fmt not in ("hepmc2", "hepmc3"):
        raise ValueError(f"fmt={fmt!r} is not one of auto, hepmc2, hepmc3, herwig")
    # HepMC2: P barcode pid px py pz E m status ...
    # HepMC3: P id vertex pid px py pz E m status
    i_pid, i_stat, i_mom = (2, 8, 3) if fmt == "hepmc2" else (3, 9, 4)

    events = []
    particles = []
    seen = 0

    with opener(filepath, "rt") as fh:
        for line in fh:
            if line.startswith("E "):
                if particles:
                    events.append(np.asarray(particles, dtype=np.float64))
                    seen += len(particles)
                    particles = []
                    if max_events is not None and len(events) >= max_events:
                        break
                continue
            if not line.startswith("P "):
                continue
            parts = line.split()
            if len(parts) <= i_stat:
                continue
            try:
                pid = int(parts[i_pid])
                status = int(parts[i_stat])
                px, py, pz, E = (float(parts[i_mom]), float(parts[i_mom + 1]),
                                 float(parts[i_mom + 2]), float(parts[i_mom + 3]))
                if status == 1 and abs(pid) not in (12, 14, 16):
                    particles.append([E, px, py, pz, pid, float(parts[i_mom + 4])] if with_id
                                     else [E, px, py, pz])
            except (ValueError, IndexError):
                continue

    # Flush last event
    if particles:
        events.append(np.asarray(particles, dtype=np.float64))
        seen += len(particles)

    if not seen:
        raise ValueError(f"{filepath}: parsed as {fmt} and found no final-state particles. "
                         f"Either it holds no events or the layout is not what {fmt} implies.")
    return events


# ---------------------------------------------------------------------------
#  The photons radiated by the beams
# ---------------------------------------------------------------------------

def remove_beam_photons(events, max_particles=P_MAX, chunk=20000):
    """
    (events without their beam photons, as [E, px, py, pz] arrays; the number removed per event).

    events are rows [E, px, py, pz, pid, mass] (parse_hepmc3 with with_id=True). The rule is the
    paper's, gentune/beam_photons.py of the release, and it reads the per-particle features of
    an event, so each event's features are built first exactly as
    extract_particles_full_flavor.py stores them: full_event_particles(), the hardest
    max_particles particles in the thrust frame of the single-seed iteration, in float32. A
    flagged row of the features is mapped back to its particle by the same ordering, by
    descending energy fraction. A particle beyond the hardest max_particles is never flagged, as
    in the paper.
    """
    root = os.path.dirname(HERE)                  # the release, whose gentune holds the rule
    for d in (root, HERE):
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        from gentune.beam_photons import beam_photon_mask
    except ImportError as e:
        raise ImportError(f"the beam-photon rule is gentune/beam_photons.py of the release, looked "
                          f"for in {root}: keep generators/ next to gentune/, or pass "
                          f"--beam-photons keep to count the beam photons ({e})") from None
    from extract_particles_full_flavor import full_event_particles
    out, nbeam = [], np.zeros(len(events), np.int64)
    for a in range(0, len(events), chunk):
        part = events[a:a + chunk]
        F = np.zeros((len(part), max_particles, 6), np.float32)
        M = np.zeros((len(part), max_particles), bool)
        order = []
        for i, ev in enumerate(part):
            F[i], M[i], _ = full_event_particles(ev, max_particles)
            E = ev[:, 0]
            z = E / max(E.sum(), 1e-12)
            o = np.argsort(-z)[:min(len(z), max_particles)]
            if not np.array_equal(F[i, :len(o), 0], z[o].astype(np.float32)):
                raise RuntimeError(f"event {a + i}: the stored features are not in the order "
                                   f"of descending energy fraction, so a flag cannot be mapped "
                                   f"back to its particle")
            order.append(o)
        flags = beam_photon_mask(F, M)
        for i, ev in enumerate(part):
            drop = np.zeros(len(ev), bool)
            drop[order[i]] = flags[i, :len(order[i])]
            nbeam[a + i] = int(drop.sum())
            out.append(np.ascontiguousarray(ev[~drop, :4]))
    return out, nbeam


def load_events(filepath, max_events=None, fmt='auto', beam_photons='remove'):
    """parse_hepmc3, with the beam photons removed (remove_beam_photons) unless beam_photons is
    'keep'. Returns the events as [E, px, py, pz] arrays and prints how many photons went."""
    if beam_photons not in ('remove', 'keep'):
        raise ValueError(f"beam_photons={beam_photons!r}: 'remove' (the paper) or 'keep'")
    if beam_photons == 'keep':
        return parse_hepmc3(filepath, max_events=max_events, fmt=fmt)
    events, nbeam = remove_beam_photons(parse_hepmc3(filepath, max_events=max_events, fmt=fmt,
                                                     with_id=True))
    print(f"  beam photons left out: {int(nbeam.sum())} in {int((nbeam > 0).sum())} of "
          f"{len(events)} events")
    return events


# ---------------------------------------------------------------------------
#  Thrust axis via iterative method
# ---------------------------------------------------------------------------

def thrust_axis(three_momenta, max_iter=50, tol=1e-12):
    """
    Find the thrust axis *n* that maximises  T = sum_i |p_i . n| / sum_i |p_i|, EXACTLY.

    The maximizing axis is parallel to sum_i s_i p_i for the sign pattern s of one cell of the
    arrangement of great circles p_i . n = 0 on the sphere, and every cell has a vertex where
    two circles cross, i.e. an axis perpendicular to two momenta. So every candidate is reached
    by taking, for each pair (i, j), n0 = p_i x p_j, fixing the signs of all other particles by
    sign(p_k . n0), and trying the four sign choices for i and j. The largest |sum s p| over
    these O(n^2) candidates is the exact thrust. About 0.4 ms for a 40-particle event.

    This replaces the single-seed sign iteration used until 2026-09-28, which converges to a
    local maximum in a few percent of Z-pole events, biasing <1-T> and <rho_heavy> upward and
    throwing those events into the far tail (generators/README.md). max_iter and tol are kept
    for the signature and ignored. Returns the unit thrust axis (3-vector) and the thrust value T.
    """
    P = np.asarray(three_momenta, dtype=np.float64)
    if P.size == 0:
        return np.array([0.0, 0.0, 1.0]), 1.0
    p_mags = np.linalg.norm(P, axis=1)
    sum_p = p_mags.sum()
    if sum_p == 0:
        return np.array([0.0, 0.0, 1.0]), 1.0
    n = len(P)
    if n == 1:
        return P[0] / (p_mags[0] + 1e-30), 1.0
    I, J = np.triu_indices(n, 1)
    N0 = np.cross(P[I], P[J])
    S = np.sign(P @ N0.T).T
    S[np.arange(len(I)), I] = 0.0
    S[np.arange(len(J)), J] = 0.0
    base = S @ P
    best, bvec = -1.0, None
    for si in (-1.0, 1.0):
        for sj in (-1.0, 1.0):
            cand = base + si * P[I] + sj * P[J]
            mags = np.linalg.norm(cand, axis=1)
            k = int(np.argmax(mags))
            if mags[k] > best:
                best, bvec = mags[k], cand[k]
    n_hat = bvec / best
    T = np.abs(P @ n_hat).sum() / sum_p
    return n_hat, T


# ---------------------------------------------------------------------------
#  Hemisphere decomposition
# ---------------------------------------------------------------------------

def split_hemispheres(event, n_hat):
    """
    Split *event* (N x 4, columns [E, px, py, pz]) into two hemispheres
    using thrust axis *n_hat*.

    Returns (hemi_plus, hemi_minus) as (N+, 4) and (N-, 4) arrays.
    Particles with p . n >= 0 go into hemi_plus.
    """
    P = event[:, 1:4]
    proj = P @ n_hat
    mask_plus = proj >= 0.0
    return event[mask_plus], event[~mask_plus]


def hemisphere_invariant_mass(hemi):
    """Invariant mass of a hemisphere 4-momentum sum."""
    if len(hemi) == 0:
        return 0.0
    E_tot = hemi[:, 0].sum()
    p_tot = hemi[:, 1:4].sum(axis=0)
    m2 = E_tot ** 2 - np.dot(p_tot, p_tot)
    return np.sqrt(max(m2, 0.0))


def get_hemispheres(event):
    """
    Return (heavy_hemi, light_hemi, n_hat, T) for an event.

    Both hemispheres are (N, 4) arrays with [E, px, py, pz].
    """
    P = event[:, 1:4]
    n_hat, T = thrust_axis(P)
    h_plus, h_minus = split_hemispheres(event, n_hat)
    m_plus = hemisphere_invariant_mass(h_plus)
    m_minus = hemisphere_invariant_mass(h_minus)
    if m_plus >= m_minus:
        return h_plus, h_minus, n_hat, T
    return h_minus, h_plus, n_hat, T


# ---------------------------------------------------------------------------
#  Event shapes
# ---------------------------------------------------------------------------

def compute_thrust(event):
    """Return thrust T in [0.5, 1]."""
    P = event[:, 1:4]
    _, T = thrust_axis(P)
    return T


def compute_c_parameter(event):
    """
    C-parameter:  C = 3 (l1*l2 + l2*l3 + l3*l1)
    where l1, l2, l3 are the eigenvalues of the linearised sphericity tensor
    Theta_{ij} = sum_k (p_k^i p_k^j) / (|p_k| sum_l |p_l|).
    """
    P = event[:, 1:4]
    p_mags = np.linalg.norm(P, axis=1, keepdims=True)
    p_mags = np.where(p_mags > 0, p_mags, 1e-30)
    sum_p = p_mags.sum()

    # Linearised sphericity tensor (3x3)
    theta = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            theta[a, b] = np.sum(P[:, a] * P[:, b] / p_mags[:, 0]) / sum_p

    eigvals = np.linalg.eigvalsh(theta)
    l1, l2, l3 = sorted(eigvals)
    C = 3.0 * (l1 * l2 + l2 * l3 + l3 * l1)
    return C


def compute_heavy_jet_mass(event):
    """
    Heavy jet mass rho_H = M_heavy^2 / E_vis^2.
    """
    heavy, _, _, _ = get_hemispheres(event)
    m_heavy = hemisphere_invariant_mass(heavy)
    E_vis = event[:, 0].sum()
    if E_vis == 0:
        return 0.0
    return (m_heavy / E_vis) ** 2


def compute_wide_broadening(event):
    """
    Wide jet broadening  B_W = max(B_+, B_-)
    where B_pm = sum_{i in H_pm} |p_i x n| / (2 sum_j |p_j|).
    """
    P = event[:, 1:4]
    n_hat, _ = thrust_axis(P)
    sum_p = np.linalg.norm(P, axis=1).sum()
    if sum_p == 0:
        return 0.0

    h_plus, h_minus = split_hemispheres(event, n_hat)

    def _broadening(hemi):
        if len(hemi) == 0:
            return 0.0
        p3 = hemi[:, 1:4]
        cross = np.cross(p3, n_hat)  # (N, 3)
        return np.linalg.norm(cross, axis=1).sum() / (2.0 * sum_p)

    b_plus = _broadening(h_plus)
    b_minus = _broadening(h_minus)
    return max(b_plus, b_minus)


def compute_light_jet_mass(event):
    """Light jet mass rho_L = M_light^2 / E_vis^2."""
    _, light, _, _ = get_hemispheres(event)
    m_light = hemisphere_invariant_mass(light)
    E_vis = event[:, 0].sum()
    if E_vis == 0:
        return 0.0
    return (m_light / E_vis) ** 2


def compute_narrow_broadening(event):
    """Narrow jet broadening B_N = min(B_+, B_-)."""
    P = event[:, 1:4]
    n_hat, _ = thrust_axis(P)
    sum_p = np.linalg.norm(P, axis=1).sum()
    if sum_p == 0:
        return 0.0
    h_plus, h_minus = split_hemispheres(event, n_hat)

    def _broadening(hemi):
        if len(hemi) == 0:
            return 0.0
        p3 = hemi[:, 1:4]
        cross = np.cross(p3, n_hat)
        return np.linalg.norm(cross, axis=1).sum() / (2.0 * sum_p)

    return min(_broadening(h_plus), _broadening(h_minus))


def compute_d_parameter(event):
    """D-parameter: D = 27 * l1 * l2 * l3 (product of sphericity eigenvalues)."""
    P = event[:, 1:4]
    p_mags = np.linalg.norm(P, axis=1, keepdims=True)
    p_mags = np.where(p_mags > 0, p_mags, 1e-30)
    sum_p = p_mags.sum()
    theta = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            theta[a, b] = np.sum(P[:, a] * P[:, b] / p_mags[:, 0]) / sum_p
    eigvals = sorted(np.linalg.eigvalsh(theta))
    return 27.0 * eigvals[0] * eigvals[1] * eigvals[2]


def compute_sphericity(event):
    """Sphericity S = 3/2 * (l2 + l3), quadratic sphericity tensor."""
    P = event[:, 1:4]
    p2 = np.sum(P ** 2, axis=1, keepdims=True)
    p2 = np.where(p2 > 0, p2, 1e-30)
    sum_p2 = p2.sum()
    S_ij = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            S_ij[a, b] = np.sum(P[:, a] * P[:, b]) / sum_p2
    eigvals = sorted(np.linalg.eigvalsh(S_ij))
    return 1.5 * (eigvals[0] + eigvals[1])


def compute_aplanarity(event):
    """Aplanarity A = 3/2 * l1 (smallest eigenvalue of quadratic sphericity tensor)."""
    P = event[:, 1:4]
    p2 = np.sum(P ** 2, axis=1, keepdims=True)
    p2 = np.where(p2 > 0, p2, 1e-30)
    sum_p2 = p2.sum()
    S_ij = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            S_ij[a, b] = np.sum(P[:, a] * P[:, b]) / sum_p2
    eigvals = sorted(np.linalg.eigvalsh(S_ij))
    return 1.5 * eigvals[0]


def _hemisphere_broadening(hemi, n_hat, sum_p_total):
    """Jet broadening for one hemisphere: B = sum |p_i x n| / (2 sum |p|)."""
    if len(hemi) == 0:
        return 0.0
    p3 = hemi[:, 1:4]
    cross = np.cross(p3, n_hat)
    return np.linalg.norm(cross, axis=1).sum() / (2.0 * sum_p_total)


def _hemisphere_mass_sq(hemi, E_vis):
    """Hemisphere invariant mass squared, normalized: rho = M^2 / E_vis^2."""
    if len(hemi) == 0 or E_vis == 0:
        return 0.0
    E_tot = hemi[:, 0].sum()
    p_tot = hemi[:, 1:4].sum(axis=0)
    m2 = E_tot ** 2 - np.dot(p_tot, p_tot)
    return max(m2, 0.0) / (E_vis ** 2)


def compute_hemisphere_shapes(event):
    """
    Compute hemisphere-based observables for one event.
    All quantities are computed per-hemisphere (heavy/wide vs light/narrow)
    using the thrust axis to split.

    Returns a dict with all observables.
    """
    _zero = {
        "rho_heavy": 0.0, "rho_light": 0.0, "rho_diff": 0.0, "rho_sum": 0.0,
        "B_wide": 0.0, "B_narrow": 0.0, "B_total": 0.0, "B_diff": 0.0,
        "mult_heavy": 0.0, "mult_light": 0.0, "mult_total": 0.0,
        "thrust": 1.0, "1_minus_thrust": 0.0,
    }
    if len(event) < 2:
        return _zero

    P = event[:, 1:4]
    n_hat, T = thrust_axis(P)
    h_plus, h_minus = split_hemispheres(event, n_hat)
    E_vis = event[:, 0].sum()
    sum_p = np.linalg.norm(P, axis=1).sum()

    # Hemisphere masses
    rho_plus = _hemisphere_mass_sq(h_plus, E_vis)
    rho_minus = _hemisphere_mass_sq(h_minus, E_vis)
    rho_heavy = max(rho_plus, rho_minus)
    rho_light = min(rho_plus, rho_minus)

    # Hemisphere broadenings
    B_plus = _hemisphere_broadening(h_plus, n_hat, sum_p)
    B_minus = _hemisphere_broadening(h_minus, n_hat, sum_p)
    B_wide = max(B_plus, B_minus)
    B_narrow = min(B_plus, B_minus)

    # Multiplicities per hemisphere
    n_plus = float(len(h_plus))
    n_minus = float(len(h_minus))

    return {
        "rho_heavy": rho_heavy,
        "rho_light": rho_light,
        "rho_diff": rho_heavy - rho_light,
        "rho_sum": rho_heavy + rho_light,
        "B_wide": B_wide,
        "B_narrow": B_narrow,
        "B_total": B_wide + B_narrow,
        "B_diff": B_wide - B_narrow,
        "mult_heavy": max(n_plus, n_minus),
        "mult_light": min(n_plus, n_minus),
        "mult_total": n_plus + n_minus,
        "thrust": T,
        "1_minus_thrust": 1.0 - T,
    }


# ---------------------------------------------------------------------------
#  EFP computation via EFPSet (degree <= 4, 36 EFPs)
# ---------------------------------------------------------------------------

_EFPSET_CACHE = {}


def get_efpset(dmax=4, measure="hadr", beta=1.0, normed=True):
    """Return a cached EFPSet for hemisphere EFP computation."""
    key = (dmax, measure, beta, normed)
    if key not in _EFPSET_CACHE:
        from energyflow import EFPSet
        _EFPSET_CACHE[key] = EFPSet(('d<=', dmax), measure=measure,
                                     beta=beta, normed=normed)
    return _EFPSET_CACHE[key]


def efp_column_names(dmax=4):
    """Return column names EFP_0, EFP_1, ..., EFP_{N-1} for d<=dmax."""
    s = get_efpset(dmax)
    return [f"EFP_{i}" for i in range(s.count())]


def compute_efps_for_event(hemi, efpset):
    """Evaluate all EFPs in *efpset* on hemisphere *hemi* [N,4]."""
    if len(hemi) < 2:
        return np.zeros(efpset.count())
    try:
        return efpset.compute(hemi)
    except Exception:
        return np.zeros(efpset.count())


# ---------------------------------------------------------------------------
#  Main processing loop
# ---------------------------------------------------------------------------

def process_events(events, efpset, hemisphere="heavy"):
    """
    Process a list of events.  For each event:
      - find the thrust axis and split into hemispheres
      - compute EFPs on the chosen hemisphere (d <= 4, 36 EFPs via EFPSet)
      - compute hemisphere-based event shapes

    Returns (efp_array, shape_rows):
      efp_array: np.ndarray [n_events, n_efps]
      shape_rows: list of dicts
    """
    n_efps = efpset.count()
    efp_array = np.zeros((len(events), n_efps))
    shape_rows = []
    n_events = len(events)

    for idx, event in enumerate(events):
        if idx % 5000 == 0 and idx > 0:
            print(f"  processed {idx}/{n_events} events ...")

        if event.shape[0] < 2:
            shape_rows.append(compute_hemisphere_shapes(event))
            continue

        heavy, light, n_hat, T = get_hemispheres(event)
        hemi = heavy if hemisphere == "heavy" else light

        # EFPs on the hemisphere
        efp_array[idx] = compute_efps_for_event(hemi, efpset)

        # Hemisphere-based event shapes
        shape_rows.append(compute_hemisphere_shapes(event))

    return efp_array, shape_rows


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def save_results(efp_array, shape_rows, output_dir, run_id, efp_columns):
    """
    Save EFP and event-shape DataFrames to CSV files.

    Files:  data/efps_run_{run_id:04d}.csv
            data/shapes_run_{run_id:04d}.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    efp_df = pd.DataFrame(efp_array, columns=efp_columns)
    shape_df = pd.DataFrame(shape_rows)

    efp_path = out / f"efps_run_{run_id:04d}.csv"
    shape_path = out / f"shapes_run_{run_id:04d}.csv"

    efp_df.to_csv(efp_path, index=False)
    shape_df.to_csv(shape_path, index=False)

    print(f"  EFPs   -> {efp_path}  ({len(efp_df)} events, {len(efp_df.columns)} columns)")
    print(f"  Shapes -> {shape_path}  ({len(shape_df)} events, {len(shape_df.columns)} columns)")
    return efp_path, shape_path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def run_single(input_path, output_dir, run_id, dmax=4,
               measure="hadr", normed=True,
               hemisphere="heavy", max_events=None, beam_photons="remove"):
    """Process one HepMC file end-to-end."""
    print(f"[run {run_id:04d}] parsing {input_path} ...")
    events = load_events(input_path, max_events=max_events, beam_photons=beam_photons)
    if not events:
        print(f"  WARNING: no events found in {input_path}", file=sys.stderr)
        return

    print(f"  {len(events)} events loaded, building EFPSet (d<={dmax}) ...")
    efpset = get_efpset(dmax=dmax, measure=measure, normed=normed)
    col_names = efp_column_names(dmax)
    print(f"  {efpset.count()} EFPs (degree <= {dmax}), hemisphere = {hemisphere}")

    efp_array, shape_rows = process_events(events, efpset, hemisphere=hemisphere)
    save_results(efp_array, shape_rows, output_dir, run_id, col_names)


def run_batch(runs_dir, output_dir, dmax=4, measure="hadr", normed=True,
              hemisphere="heavy", max_events=None, beam_photons="remove"):
    """
    Discover all run directories under *runs_dir* and process each one.

    Expected layout:  runs_dir/run_NNN/events.hepmc[.gz]
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        print(f"ERROR: runs directory {runs_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(runs_dir.glob("run_*"))
    if not run_dirs:
        print(f"ERROR: no run_* directories found in {runs_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run directories in {runs_dir}")

    # Build EFPSet once for all runs
    efpset = get_efpset(dmax=dmax, measure=measure, normed=normed)
    col_names = efp_column_names(dmax)
    print(f"{efpset.count()} EFPs (degree <= {dmax}), hemisphere = {hemisphere}\n")

    for run_dir in run_dirs:
        # Extract run id from directory name
        dir_name = run_dir.name  # e.g. "run_003"
        try:
            run_id = int(dir_name.split("_")[-1])
        except ValueError:
            run_id = 0

        # Look for event file
        hepmc_file = None
        for pattern in ["events.hepmc", "events.hepmc.gz", "*.hepmc", "*.hepmc.gz"]:
            candidates = list(run_dir.glob(pattern))
            if candidates:
                hepmc_file = candidates[0]
                break

        if hepmc_file is None:
            print(f"[run {run_id:04d}] SKIP: no HepMC file in {run_dir}")
            continue

        print(f"[run {run_id:04d}] {hepmc_file}")
        events = load_events(hepmc_file, max_events=max_events, beam_photons=beam_photons)
        if not events:
            print(f"  WARNING: no events found")
            continue

        print(f"  {len(events)} events loaded")
        efp_array, shape_rows = process_events(events, efpset, hemisphere=hemisphere)
        save_results(efp_array, shape_rows, output_dir, run_id, col_names)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute hemisphere EFPs and event shapes from HepMC3 files."
    )

    # Mode selection
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: process all run_* directories under --runs-dir.")

    # Paths
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a single HepMC3 event file (single-run mode).")
    parser.add_argument("--runs-dir", type=str, default="runs",
                        help="Parent directory containing run_NNN/ subdirectories (batch mode).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory for output CSV files.")
    parser.add_argument("--run-id", type=int, default=0,
                        help="Integer run identifier (single-run mode).")

    # Physics options
    parser.add_argument("--measure", choices=["ee", "hadr"], default="ee",
                        help="EFP angular measure (default: ee).")
    parser.add_argument("--normed", action="store_true", default=True,
                        help="Normalise EFPs by total energy (default: True).")
    parser.add_argument("--no-normed", dest="normed", action="store_false",
                        help="Disable EFP normalisation.")
    parser.add_argument("--hemisphere", choices=["heavy", "light"], default="heavy",
                        help="Which hemisphere to evaluate EFPs on (default: heavy).")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Cap on number of events per file.")
    parser.add_argument("--beam-photons", choices=["remove", "keep"], default="remove",
                        help="remove (the paper): leave the photons radiated by the beams out of "
                             "every shape, EFP and multiplicity; keep: count them, as the shape "
                             "files of an earlier draft did (default: remove).")

    args = parser.parse_args()

    if args.batch:
        run_batch(
            args.runs_dir, args.output_dir,
            measure=args.measure, normed=args.normed,
            hemisphere=args.hemisphere, max_events=args.max_events,
            beam_photons=args.beam_photons,
        )
    else:
        if args.input is None:
            parser.error("--input is required in single-run mode (or use --batch).")
        run_single(
            args.input, args.output_dir, args.run_id,
            measure=args.measure, normed=args.normed,
            hemisphere=args.hemisphere, max_events=args.max_events,
            beam_photons=args.beam_photons,
        )


if __name__ == "__main__":
    main()
