#!/usr/bin/env python3
"""Build the deposit of the MIX17 reference sample and of the calculation the paper imposes on it.

The released MIX17 head gives a weight to every event of one fixed reference sample, the pooled
union of its 720 training runs, 1597 events from each, 1,149,840 in all. Without those events the
weights have nothing to act on. This script writes them, with the per-event observables of the
paper, as one numpy file, reference_MIX17.npz. Nothing in it is recomputed. Every array is copied
from the files the paper's numbers were made from, and the script proves that before it writes.

What the weights need. A head reads an event only through its event-side vector a(Phi), of shape
(ENS, Nev, K) = (4, 1149840, 48). There are two ways to get it. One is to ship the per-particle
features and have the user run gentune.Trunk.embed on them. The other is to ship the vectors
themselves, and that is what this file does. They are the exact array of the export
(MIXGEOV3S9_cond.npz, key AE) that every number in the paper was computed from. The trunk
reproduces them only to float32 rounding, and takes about 1.7e13 multiply-adds over the whole
sample. The features route is still used below, as the check of the event order.

The paper mixes the two generators linearly. The head is evaluated only at the fraction 0, where
it is the log ratio of Sherpa, and at 1, where it is that of Herwig. Each generator's weights are
normalized over all reference events, the calculation is imposed on each generator alone, and a
prediction at the fraction xi is (1 - xi) <O>_S + xi <O>_H. Every distribution is the fraction of
all events in a bin over its width, and every mean is over all events. The observables leave out
the photons radiated by the beams (beam_photons.py).

The checks, all of which must pass before anything is moved into place:
  provenance   the released head file is bitwise the export apart from the a(Phi) it leaves out,
               the observables are in the training order of meta.json, run after run, and three
               runs replayed from their particle files give the stored reference events bitwise
  order        the released trunk on the stored particles of 8 events of every run (5760 events)
               reproduces the export's a(Phi) to float32 precision, and the same events shifted by
               one do not; the charged multiplicity and the kaon fraction recomputed from the
               particles equal the stored observables on every event
  logits       the released head on the shipped a(Phi) reproduces the export's checksum logits and
               both sets of logits in MIX17_selftest.npz at its reference_index events
  paper        at the paper's central point, numpy alone on the reference file and the targets
               file reproduces the stored means and DELPHI histograms of
               output/three_bands_MIXGEOV3S9_lin, of each generator and of the mixture, before
               and after the calculation is imposed. The maximum-entropy step is written out
               below in numpy, step for step as directlib.Model.tilt_on. The stored bands must be
               the paper's: 2000 points of the seed-11 Latin hypercube over the whole box, the
               default tunes at the fraction 1/2, and DELPHI as the data.
After writing, the file is read back, every array is checked against its sha256 in the metadata,
and the paper check is run again on what was read.

The deposit of the paper (on Symmetry, symmetry/release/refbuild_lmx.sbatch runs this, as
build_reference.py, in /mnt/beegfs/bassi/reweighting/lmx_runs/deposit_lin):

    BEAM_PHOTONS=remove MAXENT_DIR=<the rerun's folder> T_DIR=<its output/temperature, absolute> \\
        python -B build_reference.py --gentune <folder of gentune> --models <folder of MIX17_*.npz> \\
        --targets-file targets_thrust_nlo.npz

Every source then comes from the rerun's folder. Its observables are those of the refreshed bundle,
listed in output/bundle_rule.json with the beam_photons.py in use and checked by score_obs.py against
the published bundle of the same events and against the particles, and the export must carry the
refitted temperature of T_DIR. The calculation did not change, so the deposited
targets_thrust_nlo.npz stays as it is: it is checked to hold, bitwise, the calculation of this
build, and the paper check runs again on it. Only reference_MIX17.npz and build_reference.json are
written.

With BEAM_PHOTONS unset the script builds the deposit of an earlier draft as it was published
(the geometric mixture at the central point, the observables with the beam photons, histograms
normalized over the fitted bins), and also writes targets_thrust_nlo.npz. REFERENCE.md says which
revision of which script made which file.
"""
import os, sys, json, time, hashlib, zipfile, argparse, datetime, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/mnt/beegfs/bassi/reweighting/maxent_code'
# BEAM_PHOTONS=remove builds the deposit of the rerun, whose observables leave out the beam photons
# (beam_photons.py): every source is taken from the rerun's folder, the refreshed bundle's observables are
# checked by score_obs.py there against the published bundle of the same events and against the
# particles, and the export must carry the refitted temperature of its model (T_DIR, an absolute path).
# The paper check follows the linear mixture. Unset, everything is as published.
REMOVE = os.environ.get('BEAM_PHOTONS') == 'remove'
SLIM_PUBLISHED = f'{SRC}/output/models/MIXGEOV3S9_ref_v2_slim.npz'
if REMOVE:
    SRC = '/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code'
    sys.path.append(SRC)
    import score_obs as SO
    import bundle_rule as BR
EXPORT = f'{SRC}/output/models/MIXGEOV3S9_cond.npz'       # the trained export: a(Phi) and the head
REF = f'{SRC}/output/models/MIXGEOV3S9_ref.npz'           # the reference events' particles
SLIM = f'{SRC}/output/models/MIXGEOV3S9_ref_v2_slim.npz'  # their observables, exact thrust axis
DATA = f'{SRC}/data_stagePURE17_v3'                        # the training runs: meta.json, particles
TARGETS = f'{SRC}/output/thrust_targets_nlo_point.npz'    # the calculation, targets_nlo_point.py
BANDS = f'{SRC}/output/three_bands_MIXGEOV3S9' + ('_lin' if REMOVE else '')   # the paper's bands, .json and .npz

ap = argparse.ArgumentParser()
ap.add_argument('--gentune', default=f'{HERE}/check', help='directory that holds the gentune package')
ap.add_argument('--models', default=f'{HERE}/check/models', help='directory with the released MIX17_*.npz')
ap.add_argument('--out', default=HERE)
ap.add_argument('--version', default=None, help='a version string for the metadata: none by default under '
                                                 'BEAM_PHOTONS=remove, the published one when unset')
ap.add_argument('--targets-file', default=None, help='BEAM_PHOTONS=remove: the deposited targets_thrust_nlo.npz, '
                                                      'checked and kept as it is (no targets file is written)')
a = ap.parse_args()
VERSION = a.version if a.version is not None else (None if REMOVE else '1.0')
sys.path.insert(0, a.gentune)
from gentune import load_head, Trunk
from gentune.features import trunk_features

# The per-event observables, under the names the paper's code reads them by (directlib.Model), so
# the file also serves that code as a reference bundle unchanged.
OBS = ('1_minus_thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon', 'mult_total', 'strange',
       'tau_parton', 'n_parton')
SHERPA8 = ('alphas', 'pt_max', 'alpha_l', 'gamma_l', 'strange_fraction', 'baryon_fraction',
           'alpha_g', 'beta_l')
HERWIG8 = ('alpha_fsr', 'ptmin', 'clmax', 'clpow', 'psplit', 'pwtsquark', 'pwtdiquark', 'clsmr')
THETA_NAMES = tuple('S:' + n for n in SHERPA8) + tuple('H:' + n for n in HERWIG8) + ('fraction',)
GENERATORS = ('Sherpa 3.0.4', 'Herwig 7.3.0p1')
# The linear mixture: each generator's weights come from the head at its own fraction, 0 for Sherpa
# and 1 for Herwig, under the names three_bands.py stores them by.
GENS = (('sherpa', 0.0), ('herwig', 1.0))
# The moments of the calculation are <t^a (ln t)^b> inside the window, (a, b) as in
# targets_nlo_point.py.
POWERS = {'logt': (0, 1), 'tau': (1, 0), 'tlogt': (1, 1), 'log2t': (0, 2), 'tau2': (2, 0),
          'tlog2t': (1, 2), 't2logt': (2, 1), 'log3t': (0, 3), 'tau3': (3, 0), 'tlog3t': (1, 3),
          't3logt': (3, 1), 't2log2t': (2, 2), 'log4t': (0, 4), 'tau4': (4, 0)}
FLOOR_REL = 0.002       # the paper's diagonal floor of Sigma, Model(floor_rel=0.002) in three_bands.py
PER_RUN = 8             # events per run in the order check, 5760 in all
NODES = 2000            # the points of the paper's generator band (three_bands.py, linear)
SEED = 11               # the seed of their Latin hypercube (three_bands.sbatch)
TEST_DATA = 'delphi'    # the data the bands are compared with (TEST_DATA of three_bands.py)
# the paper's central point: the default tunes of Sherpa and Herwig (gentune.axes.DEFAULT) and the fraction 1/2,
# the CENTRAL of three_bands.sbatch
CENTRAL = (0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18,
           0.102337, 0.654714, 3.528693, 1.849375, 0.914156, 0.374094, 0.33107, 0.78, 0.5)
TOL_A = 1e-3            # a(Phi) from the trunk against the export: float32 rounding in the pooling stays well below
TOL_LOGIT = 1e-3        # the self-test tolerance of gentune.selftest
TOL_PAPER = 1e-5        # relative, float32 head in the paper's torch code against float64 here
DOC = {
    'AE': 'a(Phi) of every event and ensemble member, (4, N, 48) float32: pass to head.weights or '
          'head.logit of models/MIX17_head.npz. Bitwise the AE of the export the paper used.',
    '1_minus_thrust': 'tau = 1 - T, exact thrust axis, all final-state particles but neutrinos',
    'B_total': 'total jet broadening B_W + B_N, hemispheres split by the exact thrust axis',
    'rho_heavy': 'heavy hemisphere mass M_H^2 / E_vis^2, exact thrust axis',
    'nch': 'number of charged particles among the stored particles (at most 100, the hardest)',
    'nbaryon': 'number of baryons plus antibaryons in the full event',
    'mult_total': 'number of final-state particles in the full event, neutrinos excluded',
    'strange': 'fraction of the stored particles with 10^-0.36 < m/GeV < 10^-0.20 (the kaons)',
    'tau_parton': '1 - T of the end-of-shower partons, NaN for the few events without them',
    'n_parton': 'number of end-of-shower partons',
    'run_id': 'the training run each event comes from',
    'generator': 'the generator of each event, an index into generator_names',
    'generator_names': 'the two generators',
    'run_ids': 'the 720 training runs in reference order, nper events each, event i in run i // nper',
    'run_theta': 'the 17 parameters of each run, physical units, columns theta_names',
    'run_generator': 'the generator of each run, an index into generator_names',
    'theta_names': 'the parameter names, as gentune.axes.NAMES["MIX17"]',
    'nper': 'events per run',
    'meta': 'this metadata, a JSON string',
}
if REMOVE:
    for k in ('1_minus_thrust', 'B_total', 'rho_heavy', 'mult_total'):
        DOC[k] += ', the beam photons left out'
    DOC['strange'] = DOC['strange'].replace('of the stored particles', 'of the stored particles other than the beam photons')
    DOC['AE'] = ('a(Phi) of every event and ensemble member, (4, N, 48) float32: pass to head.logit or head.weights '
                 'of models/MIX17_head.npz at the fraction 0 (Sherpa) or 1 (Herwig). Bitwise the AE of the export '
                 'the paper used.')
LINEAR = ('the paper mixes the generators linearly: the weights of Sherpa are those of the head at the fraction 0 '
          'and the weights of Herwig those at 1, each normalized over all events of this file, a calculation is '
          'imposed on each generator alone, and a prediction at the fraction xi is (1 - xi) <O>_S + xi <O>_H')
t0 = time.time()


def log(s):
    print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)


def need(ok, msg):
    if not ok:
        raise SystemExit(f'CHECK FAILED: {msg}')


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def sha_array(x):
    """sha256 of the array's bytes in C order, which is what np.load gives back."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def sha_array_typed(v):
    """sha256 over the dtype, the shape and the bytes, in that order: the convention of the metadata of
    targets_thrust_nlo.npz (sha_array in tools_build_targets.py)."""
    x = np.asarray(v)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def npy_shape(path, member):
    """The shape of one array inside an npz, read off its header without loading it."""
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        v = np.lib.format.read_magic(f)
        hdr = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
        return hdr(f)[0]


def lse(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def at_fraction(theta, xi):
    """theta with the mixing fraction set to xi: 0 is Sherpa at theta_S, 1 is Herwig at theta_H."""
    t = np.array(theta, np.float64)
    t[-1] = xi
    return t


def lhs_nodes(norm, n=NODES, seed=SEED):
    """The points of the paper's generator band: the Latin hypercube of three_bands.py over the whole box
    (BOX=full), as gentune.bands.lhs_nodes draws it. norm holds the (centre, half width) of each parameter."""
    rng = np.random.default_rng(seed)
    U = (np.argsort(rng.random((n, len(norm))), axis=0) + rng.random((n, len(norm))))/n*2 - 1
    return norm[:, 0] + norm[:, 1]*U


def moment_functions(t, names, powers):
    """f_k(t) = t^a (ln t)^b per event, t clipped at 1e-6 as in directlib.Model.funs."""
    t = np.clip(np.asarray(t, np.float64), 1e-6, None)
    L = np.log(t)
    return np.column_stack([t**a*L**b for a, b in (powers[i] for i in range(len(names)))])


def tilt(lw0, inwin, Fm, c, Sig, itmax=60, tol=1e-10):
    """The maximum-entropy step of the paper in numpy, directlib.Model.newton and tilt_on.

    lw0 are the log weights of the head, normalized over all events, inwin marks the window
    events and Fm holds their moment functions. The tilted weights are w0 exp(F lam)/Z on the
    window and w0/Z outside it, F = Fm - c, with lam solving E_w[F] + Pw^2 Sigma lam = 0 and Pw the
    window fraction under w0. Newton with step halving, then one more Newton step at the converged
    multipliers as tilt_on takes it. Returns (log w, lam, Pw, iterations)."""
    lw = lw0[inwin]
    Pw = float(np.exp(lse(lw)))
    lPout = np.log1p(-Pw)
    F = Fm - c[None, :]
    Sc = Pw**2*Sig

    def state(l):
        zw = lw + F @ l
        lZ = np.logaddexp(lPout, lse(zw))
        ww = np.exp(zw - lZ)
        m = ww @ F
        return ww, m, m + Sc @ l

    lam = np.zeros(len(c))
    ww, m, g = state(lam)
    it = 0
    for it in range(itmax):
        if np.linalg.norm(g) < tol:
            break
        J = (F*ww[:, None]).T @ F - np.outer(m, m) + Sc
        step = np.linalg.solve(J, g)
        g0, s = np.linalg.norm(g), 1.0
        for _ in range(30):
            lt = lam - s*step
            wt, mt, gt = state(lt)
            if np.linalg.norm(gt) < g0:
                break
            s *= 0.5
        lam, ww, m, g = lt, wt, mt, gt
    J = (F*ww[:, None]).T @ F - np.outer(m, m) + Sc
    lam1 = lam - np.linalg.solve(J, g)
    z = lw0.copy()
    z[inwin] += F @ lam1
    return z - lse(z), lam1, Pw, it


def predict(lw, ev, inwin, edges, use):
    """The seven means of three_bands.py and the DELPHI histogram normalized over the fitted bins."""
    w = np.exp(lw)
    thr = ev['1_minus_thrust'].astype(np.float64)
    pw = float(w[inwin].sum())
    o = dict(tau_win=float(w[inwin] @ thr[inwin])/pw, pwin=pw, thrust=float(w @ thr),
             **{k: float(w @ ev[k].astype(np.float64)) for k in ('B_total', 'rho_heavy', 'nch', 'nbaryon')})
    nb = len(edges) - 1
    idx = np.searchsorted(edges, thr, side='right') - 1
    ins = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ins], weights=w[ins], minlength=nb)
    return o, cnt/(np.diff(edges)*cnt[use].sum())


def predict_lin(lw, ev, inwin, edges):
    """One generator under the linear mixture: the seven means of three_bands.py, each over all events
    (the window mean over the window events of this generator), and the DELPHI histogram as the fraction
    of all events in each bin over its width, with no renormalization over a range of bins."""
    w = np.exp(lw)
    thr = ev['1_minus_thrust'].astype(np.float64)
    pw = float(w[inwin].sum())
    o = dict(tau_win=float(w[inwin] @ thr[inwin])/pw, pwin=pw, thrust=float(w @ thr),
             **{k: float(w @ ev[k].astype(np.float64)) for k in ('B_total', 'rho_heavy', 'nch', 'nbaryon')})
    nb = len(edges) - 1
    idx = np.searchsorted(edges, thr, side='right') - 1
    ins = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ins], weights=w[ins], minlength=nb)
    return o, cnt/np.diff(edges)


def rel_hist(h, hs):
    """The largest relative difference between two histograms; a bin that is empty in both is equal."""
    h, hs = np.asarray(h, np.float64), np.asarray(hs, np.float64)
    need(h.shape == hs.shape, f'histograms of {h.shape} and {hs.shape} bins')
    d = np.abs(h - hs)
    safe = np.where(hs != 0, np.abs(hs), 1.0)
    return float(np.max(np.where(hs != 0, d/safe, np.where(d == 0, 0.0, np.inf))))


def paper_check(ev, AE, T, head, bands, B):
    """Reproduce the paper's central point from a reference file and a targets file alone."""
    central = np.asarray(bands['central']['theta'], np.float64)
    f = head.logit(AE, central)/head.T
    lw0 = f - lse(f)
    names = [str(k) for k in T['moment_names']]
    powers = {i: tuple(int(v) for v in T['moment_powers'][i]) for i in range(len(names))}
    thr = ev['1_minus_thrust'].astype(np.float64)
    lo, hi = T['window']
    inwin = (thr >= lo) & (thr < hi)
    c = np.asarray(T['central'], np.float64)
    Sig = np.asarray(T['sigma_pert'], np.float64) + np.diag((float(T['floor_rel'])*np.abs(c))**2)
    lw, lam, Pw, its = tilt(lw0, inwin, moment_functions(thr[inwin], names, powers), c, Sig)
    edges = np.r_[B['lo_low'], B['lo'], B['hi'][-1]]
    need(np.allclose(np.r_[B['hi_low'], B['hi']][:-1], edges[1:-1], rtol=0, atol=1e-12), 'DELPHI bins not contiguous')
    use = np.arange(len(B['lo_low']), len(edges) - 1)
    out = {}
    for tag, l, ref, href in (('before', lw0, bands['central']['generator_only'], 'h_central_before'),
                              ('after', lw, bands['central']['imposed'], 'h_central')):
        o, h = predict(l, ev, inwin, edges, use)
        hs = np.r_[B[href + '_low'], B[href]]
        out[tag] = dict(max_rel_means=max(abs(o[k]/ref[k] - 1) for k in ref),
                        max_rel_hist=float(np.max(np.abs(h/hs - 1))),
                        thrust_mean=o['thrust'], thrust_mean_paper=ref['thrust'])
    out.update(newton_iterations=its, window_fraction=Pw, events_in_window=int(inwin.sum()),
               lam=lam.tolist())
    return out


def paper_check_lin(ev, AE, T, head, bands, B):
    """The paper's central point under the linear mixture, from a reference file and a targets file alone.

    For each generator, the head at its own fraction (0 for Sherpa, 1 for Herwig) gives log weights
    normalized over all events; the calculation is imposed on them alone; the means and the DELPHI
    histogram before and after are compared with that generator's in the stored bands. The two are then
    mixed with (1 - xi, xi) at the xi of the central point and compared with the stored mixture."""
    central = np.asarray(bands['central']['theta'], np.float64)
    xi = float(central[-1])
    names = [str(k) for k in T['moment_names']]
    powers = {i: tuple(int(v) for v in T['moment_powers'][i]) for i in range(len(names))}
    thr = ev['1_minus_thrust'].astype(np.float64)
    lo, hi = T['window']
    inwin = (thr >= lo) & (thr < hi)
    c = np.asarray(T['central'], np.float64)
    Sig = np.asarray(T['sigma_pert'], np.float64) + np.diag((float(T['floor_rel'])*np.abs(c))**2)
    Fm = moment_functions(thr[inwin], names, powers)
    edges = np.r_[B['lo_low'], B['lo'], B['hi'][-1]]
    need(np.allclose(np.r_[B['hi_low'], B['hi']][:-1], edges[1:-1], rtol=0, atol=1e-12), 'DELPHI bins not contiguous')
    stored = lambda key: np.r_[B[key + '_low'], B[key]]
    out = dict(xi=xi, events_in_window=int(inwin.sum()), generators={}, mixed={})
    mix = {}
    for nm, xg in GENS:
        share = 1 - xi if xg == 0.0 else xi
        f = head.logit(AE, at_fraction(central, xg))/head.T
        lw0 = f - lse(f)
        lw, lam, Pw, its = tilt(lw0, inwin, Fm, c, Sig)
        ref = bands['central']['generators'][nm]
        rec = dict(fraction=xg, newton_iterations=its, window_fraction=Pw,
                   max_rel_window_fraction=abs(Pw/float(ref['window_fraction_before']) - 1), lam=lam.tolist())
        for tag, l, refo, href in (('before', lw0, ref['generator_only'], f'h_central_before_{nm}'),
                                   ('after', lw, ref['imposed'], f'h_central_{nm}')):
            o, h = predict_lin(l, ev, inwin, edges)
            rec[tag] = dict(max_rel_means=max(abs(o[k]/refo[k] - 1) for k in refo),
                            max_rel_hist=rel_hist(h, stored(href)),
                            thrust_mean=o['thrust'], thrust_mean_paper=refo['thrust'])
            if share:
                mo, mh = mix.get(tag, ({k: 0.0 for k in o}, 0.0))
                mix[tag] = ({k: mo[k] + share*o[k] for k in o}, mh + share*h)
        out['generators'][nm] = rec
    for tag, refo, href in (('before', bands['central']['generator_only'], 'h_central_before'),
                            ('after', bands['central']['imposed'], 'h_central')):
        o, h = mix[tag]
        out['mixed'][tag] = dict(max_rel_means=max(abs(o[k]/refo[k] - 1) for k in refo),
                                 max_rel_hist=rel_hist(h, stored(href)),
                                 thrust_mean=o['thrust'], thrust_mean_paper=refo['thrust'])
    return out


def paper_worst(pc):
    """The largest relative difference of a paper check."""
    if 'mixed' in pc:
        blocks = [pc['mixed'][t] for t in ('before', 'after')] + \
                 [g[t] for g in pc['generators'].values() for t in ('before', 'after')]
        extra = [g['max_rel_window_fraction'] for g in pc['generators'].values()]
    else:
        blocks, extra = [pc[t] for t in ('before', 'after')], []
    return max([b[y] for b in blocks for y in ('max_rel_means', 'max_rel_hist')] + extra)


def paper_line(pc):
    if 'mixed' not in pc:
        return (f'before {pc["before"]["max_rel_means"]:.1e} (means) {pc["before"]["max_rel_hist"]:.1e} (histogram), '
                f'after {pc["after"]["max_rel_means"]:.1e} {pc["after"]["max_rel_hist"]:.1e}, '
                f'{pc["newton_iterations"]} Newton steps')
    part = lambda r: (f'before {r["before"]["max_rel_means"]:.1e} (means) {r["before"]["max_rel_hist"]:.1e} (histogram), '
                      f'after {r["after"]["max_rel_means"]:.1e} {r["after"]["max_rel_hist"]:.1e}')
    gens = '; '.join(f'{nm} {part(r)}, {r["newton_iterations"]} Newton steps' for nm, r in pc['generators'].items())
    return f'{gens}; mixed at xi = {pc["xi"]:g} {part(pc["mixed"])}; {pc["events_in_window"]} window events'


# ---- the rerun: where every source comes from ---------------------------------------------------
if REMOVE:
    need(SO.REMOVE, 'score_obs.py does not see BEAM_PHOTONS=remove')
    for mod in (SO, BR):
        need(os.path.dirname(os.path.realpath(mod.__file__)) == os.path.realpath(SRC),
             f'{mod.__name__} is imported from {mod.__file__}, not from {SRC}')
    need(os.path.realpath(BR.run_dir()) == os.path.realpath(SRC),
         f'MAXENT_DIR is {BR.run_dir()}: output/bundle_rule.json would not be read from {SRC}')
    need(bool(SO.T_DIR) and os.path.isabs(SO.T_DIR),
         'BEAM_PHOTONS=remove needs T_DIR, the absolute path of the refitted temperatures (D11)')
    need(a.targets_file is not None and os.path.isfile(a.targets_file),
         '--targets-file: give the deposited targets_thrust_nlo.npz, which this build checks and keeps')
    for p in (SLIM, REF):    # listed in output/bundle_rule.json with the beam_photons.py in use (I1, I2)
        try:
            BR.check(p)
        except RuntimeError as e:
            need(False, str(e))
    need(os.path.dirname(os.path.realpath(BR.rule_md5()[0])) == os.path.realpath(SRC),
         f'beam_photons.py is {BR.rule_md5()[0]}, not that of {SRC}')

# ---- sources ---------------------------------------------------------------------------------
src_sha = {p: sha_file(p) for p in (EXPORT, REF, SLIM, f'{DATA}/meta.json', TARGETS,
                                    BANDS + '.json', BANDS + '.npz')}
rel = {f: f'{a.models}/{f}' for f in ('MIX17_head.npz', 'MIX17_trunk.npz', 'MIX17_selftest.npz')}
rel_sha = {f: sha_file(p) for f, p in rel.items()}
log('hashed the sources')
checks = {}

# ---- provenance: the released head is the export's ---------------------------------------------
c = np.load(EXPORT)
if REMOVE:
    SO.check_export_T(EXPORT, c['temperature'])
AE = c['AE']
ENS, N, K = AE.shape
h = np.load(rel['MIX17_head.npz'])
common = [k for k in h.files if k in c.files]
differ = [k for k in common if not np.array_equal(h[k], c[k])]
need(not differ, f'the released head differs from the export in {differ}')
need(int(h['n_reference_events']) == N, f'head says {int(h["n_reference_events"])} reference events, export has {N}')
head = load_head(rel['MIX17_head.npz'])
need(head.ENS == ENS and head.K == K, 'head and a(Phi) shapes disagree')
checks['release_head_equals_export'] = f'{len(common)} arrays bitwise equal, {len(h.files) - len(common)} only in the release'
log(f'a(Phi) {AE.shape} {AE.dtype}; the released head equals the export in all {len(common)} shared arrays')

# ---- the order of the runs ---------------------------------------------------------------------
s = np.load(SLIM)
meta = json.load(open(f'{DATA}/meta.json'))
run_ids = np.asarray(s['order'], np.int64)
nper = int(s['nper'])
need([int(t['rid']) for t in meta['train']] == run_ids.tolist(), 'slim order is not meta.json train order')
need(len(run_ids)*nper == N, f'{len(run_ids)} runs x {nper} is not {N}')
need(np.array_equal(np.sort(run_ids), c['train_rids']), 'run ids differ from the export train_rids')
need(tuple(meta['s_axes']) == SHERPA8 and tuple(meta['h_axes']) == HERWIG8, 'parameter names differ')
run_theta = np.array([t['theta'] for t in meta['train']], np.float64)
fitted = {int(r): th for r, th in zip(c['train_rids'], c['fitted_theta'])}
need(all(np.array_equal(fitted[int(r)], run_theta[i]) for i, r in enumerate(run_ids)), 'run theta differs from the export')
gen = np.array([t['n_herwig'] > 0 for t in meta['train']], np.int8)
need(all((t['n_sherpa'] > 0) != (t['n_herwig'] > 0) for t in meta['train']), 'a training run mixes generators')
need(np.array_equal(run_theta[:, -1], gen.astype(np.float64)), 'fraction is not 0 for Sherpa and 1 for Herwig')
n_run = np.array([npy_shape(f'{DATA}/particles_full_{r:04d}.npz', 'mask.npy')[0] for r in run_ids])
checks['runs'] = dict(n=len(run_ids), sherpa=int((gen == 0).sum()), herwig=int((gen == 1).sum()), nper=nper,
                      events_per_run=sorted(set(n_run.tolist())))
log(f'{len(run_ids)} runs ({checks["runs"]["sherpa"]} Sherpa, {checks["runs"]["herwig"]} Herwig) x {nper}, '
    f'run sizes {checks["runs"]["events_per_run"]}')

# ---- replay: the reference events of a run are a fixed draw of its events ----------------------
R = np.load(REF)
P, Mk = R['particles'], R['mask']
need(len(Mk) == N, 'particle and a(Phi) counts differ')
replayed = []
for k in (0, int(np.argmax(gen == 1)), len(run_ids) - 1):
    rid = int(run_ids[k])
    z = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
    idx = np.random.default_rng(50000 + rid).choice(len(z['mask']), min(nper, len(z['mask'])), replace=False)
    sl = slice(k*nper, (k + 1)*nper)
    need(np.array_equal(z['particles'][idx], P[sl]) and np.array_equal(z['mask'][idx], Mk[sl]),
         f'run {rid} does not replay')
    replayed.append(rid)
checks['replayed_runs_bitwise'] = replayed
log(f'runs {replayed} replay bitwise from their particle files')

# ---- order: the released trunk on the stored particles gives the export's a(Phi) ----------------
trunk = Trunk(rel['MIX17_trunk.npz'])
head.check_trunk(trunk)
sel = (np.arange(len(run_ids))[:, None]*nper + np.linspace(0, nper - 1, PER_RUN).astype(int)[None, :]).ravel()
A_sel = trunk.embed(trunk_features(P[sel]), Mk[sel])
dA = float(np.max(np.abs(A_sel.astype(np.float64) - AE[:, sel])))
dA_shift = float(np.max(np.abs(A_sel.astype(np.float64) - AE[:, (sel + 1) % N])))
th_c = np.array(json.load(open(BANDS + '.json'))['central']['theta'])
# the logits at the central point; under the linear mixture the head is evaluated at the fractions 0 and 1 only
pts = [at_fraction(th_c, x) for _, x in GENS] if REMOVE else [th_c]
dlogit = max(float(np.max(np.abs(head.logit(A_sel, t) - head.logit(AE[:, sel], t)))) for t in pts)
need(dA < TOL_A, f'trunk on the stored particles misses a(Phi) by {dA:.2e}')
need(dA_shift > 100*TOL_A, f'a shift by one event is not seen ({dA_shift:.2e})')
nch = ((P[..., 5] != 0) & Mk).sum(1).astype(np.int16)
need(np.array_equal(nch, s['nch']), 'nch recomputed from the particles differs')
lm = P[..., 3]
ink = (lm > -0.36) & (lm < -0.20) & Mk
nb = SO.n_beam(P, Mk) if REMOVE else 0    # under remove the beam photons leave the denominator and the count
strange = (ink.sum(1)/np.maximum(Mk.sum(1) - nb, 1)).astype(np.float32)
need(np.array_equal(strange, s['strange']), 'strange recomputed from the particles differs')
nstored = Mk.sum(1)
mt = s['mult_total']
below = nstored < 100                     # mult_total counts the full event, the particles stop at 100
checks['order'] = dict(events=len(sel), max_abs_dA=dA, max_abs_A=float(np.max(np.abs(AE[:, sel]))),
                       max_abs_dA_shifted_by_one=dA_shift, max_abs_dlogit_at_central=dlogit,
                       nch_equal_all_events=True, strange_equal_all_events=True,
                       events_truncated_at_100=int((~below).sum()),
                       mult_total_equals_stored_count_below_100=float(np.mean(mt[below] == (nstored - nb)[below])))
if REMOVE:
    checks['order']['logits_at_central_taken_at_fractions'] = [x for _, x in GENS]
    # the deposited observables are the refreshed bundle's: its shapes against the published bundle of the same
    # events, its multiplicity and kaon fraction against the particles (score_obs.reference_obs)
    robs = SO.reference_obs(SLIM, list(SO.SHAPES) + ['mult_total', 'strange'], particles=REF, orig=SLIM_PUBLISHED)
    need(np.array_equal(robs['mult_total'], np.asarray(mt, np.float64)), 'mult_total still counts the beam photons')
    checks['beam_photons'] = dict(SO.describe(), checked_against=SLIM_PUBLISHED,
                                  checked_against_sha256=sha_file(SLIM_PUBLISHED),
                                  events_with_beam_photons=int((nb > 0).sum()))
log(f'order: trunk vs export a(Phi) {dA:.2e} on {len(sel)} events (shifted by one {dA_shift:.2f}), '
    f'logit {dlogit:.2e}; nch and strange equal on all {N} events')
del P, Mk, R, lm, ink

# ---- logits: the released head on the shipped a(Phi) ------------------------------------------
sb = np.load(rel['MIX17_selftest.npz'])
ri = sb['reference_index']
need(int(sb['n_reference_events']) == N, 'self-test bundle is for another reference')
dAE = float(np.max(np.abs(sb['AE'] - AE[:, ri])))
f = np.stack([head.logit(AE[:, ri], t) for t in sb['theta_probe']])
d_exp = float(np.max(np.abs(f - sb['logit_probe_from_export'])))
d_probe = float(np.max(np.abs(f - sb['logit_probe'])))
d_chk = float(np.max(np.abs(head.logit(AE[:, :256], c['f_checksum_theta'], check_box=False) - c['f_checksum'])))
need(dAE == 0.0, f'self-test a(Phi) differs from the export at reference_index by {dAE:.2e}')
need(max(d_exp, d_probe, d_chk) < TOL_LOGIT, f'logits off: {d_exp:.2e} {d_probe:.2e} {d_chk:.2e}')
checks['logits'] = dict(selftest_events=len(ri), selftest_points=len(sb['theta_probe']),
                        max_abs_dAE_selftest=dAE, max_abs_dlogit_vs_export=d_exp,
                        max_abs_dlogit_vs_selftest=d_probe, max_abs_dlogit_checksum=d_chk)
log(f'logits: self-test a(Phi) equal, vs export logits {d_exp:.2e}, vs shipped logits {d_probe:.2e}, '
    f'checksum {d_chk:.2e}')

# ---- the two output files, in memory -----------------------------------------------------------
ev = {k: s[k] for k in OBS}
ref_arrays = dict(AE=AE, **ev,
                  run_id=np.repeat(run_ids, nper).astype(np.int16),
                  generator=np.repeat(gen, nper),
                  generator_names=np.array(GENERATORS),
                  run_ids=run_ids, run_theta=run_theta, run_generator=gen,
                  theta_names=np.array(THETA_NAMES), nper=np.int64(nper))
G = np.load(TARGETS)
names = [str(k) for k in G['keys']]
need(set(names) == set(POWERS), 'moment names differ from targets_nlo_point.py')
need(len(G['alphas']) == 1 and len(G['alpha0']) == 1, 'not a single-point calculation')
tgt_arrays = dict(
    moment_names=G['keys'], moment_powers=np.array([POWERS[k] for k in names], np.int64),
    central=G['central'][0, 0], sigma_pert=G['sigma_pert'][0, 0], window=G['window'],
    floor_rel=np.float64(FLOOR_REL), alpha_s=G['alphas'][0], alpha_0=G['alpha0'][0], pwin=G['pwin'][0, 0],
    scale_variations=G['scale'][0, 0], scale_labels=G['scale_labels'], scale_pwin=G['pwin_scale'][0, 0],
    np_variations=G['np_vars'][0, 0], np_labels=G['np_labels'],
    milan=G['milan'], sub_order_central=G['sub_order_central'], order=G['order'],
    calc_point=G['calc_point'])
same = {'central': G['central'][0, 0], 'sigma_pert': G['sigma_pert'][0, 0], 'window': G['window'],
        'scale_variations': G['scale'][0, 0], 'np_variations': G['np_vars'][0, 0]}
need(all(np.array_equal(tgt_arrays[k], v) for k, v in same.items()), 'a number of the calculation changed')
t = np.geomspace(1e-7, 0.5, 1001)
L = np.log(np.clip(t, 1e-6, None))
dm = {'logt': L, 'tau': np.clip(t, 1e-6, None), 'tlogt': np.clip(t, 1e-6, None)*L}
for k, v in dm.items():                   # the powers give directlib's functions bitwise
    need(np.array_equal(moment_functions(t, [k], {0: POWERS[k]})[:, 0], v), f'moment function {k}')
labels = [str(v) for v in G['scale_labels']]
pairs = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
D = [np.sqrt(0.5)*(G['scale'][0, 0, labels.index(l)] - G['central'][0, 0]) for p in pairs for l in p]
D.append(G['scale'][0, 0, labels.index("(1.0, 1.0, 'modR', 1.0)")] - G['central'][0, 0])
D = np.stack(D, 1)
sig_rec = float(np.max(np.abs(D @ D.T - G['sigma_pert'][0, 0]))/np.max(np.abs(G['sigma_pert'][0, 0])))
need(sig_rec < 1e-12, f'sigma_pert is not the documented sum over the variations ({sig_rec:.1e})')
checks['targets'] = dict(numbers_bitwise_equal_to_source=sorted(same), sigma_pert_rebuilt_rel=sig_rec)
if REMOVE:
    # the deposited targets file stays: it must hold this calculation bitwise, name it as its source, and carry the
    # sha256 of every array in its own metadata
    TK = np.load(a.targets_file, allow_pickle=False)
    tmeta = json.loads(str(TK['meta']))
    need(set(TK.files) == set(tgt_arrays) | {'meta'}, f'the deposited targets file holds {sorted(TK.files)}')
    need(all(TK[k].dtype == np.asarray(v).dtype and TK[k].shape == np.shape(v)
             and TK[k].tobytes() == np.asarray(v).tobytes() for k, v in tgt_arrays.items()),
         'the deposited targets file does not hold the calculation of this rerun bitwise')
    need(tmeta.get('source') == {os.path.basename(TARGETS): src_sha[TARGETS]},
         f'the deposited targets file names another source: {tmeta.get("source")}')
    need(set(tmeta.get('arrays', {})) == set(tgt_arrays)
         and all(sha_array_typed(TK[k]) == d['sha256'] for k, d in tmeta['arrays'].items()),
         'an array of the deposited targets file does not have the sha256 of its metadata')
    tgt_kept = dict(file=os.path.basename(a.targets_file), bytes=os.path.getsize(a.targets_file),
                    sha256=sha_file(a.targets_file), built=tmeta.get('built'),
                    build_script_sha256=tmeta.get('build_script_sha256'))
    checks['targets'].update(deposited_file_kept=tgt_kept, deposited_arrays_bitwise_equal=sorted(tgt_arrays),
                             deposited_source_is_this_calculation=True)
    log(f'targets: {tgt_kept["file"]} holds this calculation bitwise (sha256 {tgt_kept["sha256"]}); kept as it is')

# ---- the paper, from the in-memory arrays --------------------------------------------------------
bands = json.load(open(BANDS + '.json'))
need(bands['export'].endswith('MIXGEOV3S9_cond.npz') and bands['grid'].endswith('thrust_targets_nlo_point.npz')
     and not bands['wifi_central'], 'the stored bands are not of this export, calculation and central network')
if REMOVE:
    need(bands.get('mix') == 'linear' and bands.get('box') == 'full' and bands.get('nodes') == NODES
         and 'hypercube_size' not in bands,
         f'the stored bands are not those of the paper: the linear mixture, {NODES} points over the full box')
    need(bands['central']['theta'] == list(CENTRAL) and set(bands['central'].get('generators', {})) == {n for n, _ in GENS},
         f"the stored central point is not the paper's, the default tunes at xi = 1/2 with each generator stored: "
         f'{bands["central"]["theta"]}')
    need(bands.get('test') == TEST_DATA, f'the stored bands are compared with {bands.get("test")}, not {TEST_DATA}')
B = np.load(BANDS + '.npz')
if REMOVE:
    norm = np.asarray(c['norm'], np.float64)
    nodes = np.asarray(B['nodes'], np.float64)
    need(nodes.shape == (NODES, len(norm)), f'the stored bands hold points of shape {nodes.shape}')
    dnode = float(np.max(np.abs(nodes - lhs_nodes(norm))/norm[:, 1]))
    need(dnode < 1e-12, f'the stored points are not the Latin hypercube of seed {SEED} over the box of the export '
                        f'(they differ by {dnode:.2e} of a half width)')
    checks['bands'] = dict(mix='linear', box='full', nodes=NODES, seed=SEED, test=TEST_DATA, central=list(CENTRAL),
                           max_node_difference_half_widths=dnode)
check_paper = paper_check_lin if REMOVE else paper_check
pc = check_paper(ev, AE, tgt_arrays, head, bands, B)
worst = paper_worst(pc)
need(worst < TOL_PAPER, f'the paper is not reproduced: worst relative difference {worst:.2e}')
checks['paper'] = pc
log(f'paper: {paper_line(pc)}')

# ---- write, read back, check again --------------------------------------------------------------
build_sha = sha_file(os.path.abspath(__file__))
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
src_info = {os.path.relpath(p, SRC): sha for p, sha in src_sha.items()}
features_bytes = sum(i.compress_size for i in zipfile.ZipFile(REF).infolist() if i.filename in ('particles.npy', 'mask.npy'))
ref_meta = dict(
    name='reference_MIX17', **({} if VERSION is None else dict(version=VERSION)), built=stamp, build_script_sha256=build_sha,
    paper='B. Assi and J. Thaler, Combined Uncertainties from Event Generators and Precision QCD',
    N=N, ens=ENS, K=K, process='e+e- -> hadrons at 91.2 GeV',
    order=(f'run after run in run_ids, the training order of {os.path.relpath(DATA, SRC)}/meta.json, '
           f'{nper} events per run. Within run r the events are '
           f'numpy.random.default_rng(50000 + r).choice(n_r, {nper}, replace=False) of its n_r = 80000 '
           f'training events, in the order drawn. Event i of every array is event i of AE[:, i].'),
    pairs_with=dict(head='models/MIX17_head.npz', head_sha256=rel_sha['MIX17_head.npz'],
                    trunk='models/MIX17_trunk.npz', trunk_sha256=rel_sha['MIX17_trunk.npz'],
                    export='MIXGEOV3S9'),
    **({'mixture': LINEAR} if REMOVE else {}),
    why_AE=(f'a(Phi) is shipped instead of the per-particle features ({features_bytes/1e9:.2f} GB compressed): '
            f'it is the array every number of the paper was computed from, and needs no trunk pass. '
            f'The released trunk on the features reproduces it to {dA:.1e}.'),
    source=src_info, keys=DOC,
    arrays={k: dict(dtype=str(v.dtype), shape=list(np.shape(v)), sha256=sha_array(v)) for k, v in ref_arrays.items()},
    checks=checks)
ref_arrays['meta'] = np.array(json.dumps(ref_meta, indent=1))
paths = {'reference_MIX17.npz': np.savez_compressed}
docs = {'reference_MIX17.npz': ref_meta}
if not REMOVE:
    tgt_doc = dict(
        name='targets_thrust_nlo', version=VERSION, built=stamp, build_script_sha256=build_sha,
        source={os.path.relpath(TARGETS, SRC): src_sha[TARGETS]},
        calculation=('NNLL+NLO thrust with a dispersive power correction, at the published pair (alpha_s, alpha_0) of Gehrmann, Luisoni '
                     'and Monni, Eur. Phys. J. C 73 (2013) 2265, log-R, with their subtraction to O(alpha_s^3) (output/calc_point.json, targets_nlo_point.py)'),
        keys=dict(
            moment_names='the 14 moments, in order',
            moment_powers='(a, b) of each moment: f(tau) = tau^a (ln tau)^b, tau clipped at 1e-6',
            central='c_k = <f_k(tau)> over window[0] <= tau < window[1], normalized inside the window',
            sigma_pert=('Sigma_c of the calculation, sum over d d^T with d the scale variation minus central: the two '
                        'log-R pairs (muR x2 and x1/2, x_V x2 and x1/2) at weight 1/2 each, the mod-R scheme at '
                        'central scales at weight 1'),
            window='[low, high) in tau = 1 - T, the only events the calculation constrains',
            floor_rel=('the paper\'s diagonal floor: the maximum-entropy step uses '
                       'Sigma = sigma_pert + diag((floor_rel*|central|)^2)'),
            alpha_s='alpha_s(m_Z) of the point', alpha_0='alpha_0 of the dispersive model at the point',
            pwin='the calculated window fraction (not imposed by the paper)',
            scale_variations='the moments for each scale and scheme variation at the same point, rows scale_labels',
            scale_labels='(x_muR, x_V, matching scheme, x_NNLO) of each row',
            scale_pwin='the window fraction of each scale variation',
            np_variations='the moments for each variation of the dispersive model, rows np_labels',
            np_labels='Milan factor +20 percent, -20 percent, the renormalon subtraction at the other order',
            milan='the central Milan factor', sub_order_central='order of the central renormalon subtraction',
            order='perturbative order of the fixed-order part (2: NLO)',
            calc_point='the point and its source, JSON'),
        arrays={k: dict(dtype=str(np.asarray(v).dtype), shape=list(np.shape(v)), sha256=sha_array(v))
                for k, v in tgt_arrays.items()})
    tgt_arrays['meta'] = np.array(json.dumps(tgt_doc, indent=1))
    paths['targets_thrust_nlo.npz'] = np.savez
    docs['targets_thrust_nlo.npz'] = tgt_doc

tmp = {n: os.path.join(a.out, n.replace('.npz', '.partial.npz')) for n in paths}
paths['reference_MIX17.npz'](tmp['reference_MIX17.npz'], **ref_arrays)
if not REMOVE:
    paths['targets_thrust_nlo.npz'](tmp['targets_thrust_nlo.npz'], **tgt_arrays)
log('written, reading back')
del AE, ev, ref_arrays
rr = np.load(tmp['reference_MIX17.npz'])
tt = np.load(a.targets_file, allow_pickle=False) if REMOVE else np.load(tmp['targets_thrust_nlo.npz'])
for n, doc in docs.items():
    z = rr if n == 'reference_MIX17.npz' else tt
    m = json.loads(str(z['meta']))
    need(m == json.loads(json.dumps(doc)), 'metadata did not survive the round trip')
    for k, d in m['arrays'].items():
        need(sha_array(z[k]) == d['sha256'], f'{k} changed on the way to disk')
ev = {k: rr[k] for k in OBS}
pc2 = check_paper(ev, rr['AE'], tt, head, bands, B)
worst2 = paper_worst(pc2)
need(worst2 < TOL_PAPER, f'the paper is not reproduced from the files read back: {worst2:.2e}')
log(f'paper from the files read back: worst relative difference {worst2:.1e}')
for n, p in tmp.items():
    os.replace(p, os.path.join(a.out, n))
out = {n: dict(bytes=os.path.getsize(os.path.join(a.out, n)), sha256=sha_file(os.path.join(a.out, n))) for n in paths}
summary = dict(built=stamp, build_script=os.path.basename(__file__) if REMOVE else 'build_reference.py',
               build_script_sha256=build_sha, outputs=out,
               sources=src_info, release=rel_sha, numpy=np.__version__, python=platform.python_version(),
               checks=checks, paper_from_files_read_back=pc2, seconds=round(time.time() - t0))
if REMOVE:
    mark = os.path.join(SRC, 'output', '.lmx_rerun_start')
    summary['rerun'] = dict(run_dir=SRC, mark_run_id=json.load(open(mark)).get('run_id') if os.path.exists(mark) else None,
                            t_dir=SO.T_DIR, beam_photons=SO.describe(),
                            bundle_rule_sha256=sha_file(os.path.join(SRC, 'output', 'bundle_rule.json')),
                            targets_kept=tgt_kept, worst_relative_difference_paper=worst,
                            worst_relative_difference_paper_read_back=worst2)
json.dump(summary, open(os.path.join(a.out, 'build_reference.json'), 'w'), indent=1)
for n, d in out.items():
    log(f'{n}: {d["bytes"]/1e6:.1f} MB, sha256 {d["sha256"]}')
log('all checks passed')
