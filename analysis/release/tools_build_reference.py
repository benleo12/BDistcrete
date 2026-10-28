#!/usr/bin/env python3
"""Build the deposit of the MIX17 reference sample and of the calculation the paper imposes on it.

The released MIX17 head gives a weight to every event of one fixed reference sample, the pooled
union of its 720 training runs, 1597 events from each, 1,149,840 in all. Without those events the
weights have nothing to act on. This script writes them, with the per-event observables of the
paper, as one numpy file, reference_MIX17.npz, and the paper's NNLL+NLO thrust calculation as
targets_thrust_nlo.npz. Nothing in either file is recomputed. Every array is copied from the files
the paper's numbers were made from, and the script proves that before it writes.

What the weights need. A head reads an event only through its event-side vector a(Phi), of shape
(ENS, Nev, K) = (4, 1149840, 48). There are two ways to get it. One is to ship the per-particle
features (0.62 GB compressed) and have the user run gentune.Trunk.embed on them. The other is to
ship the vectors themselves (0.83 GB compressed), and that is what this file does. They are the
exact array of the export (MIXGEOV3S9_cond.npz, key AE) that every number in the paper was
computed from. The trunk reproduces them only to float32 rounding, about 2e-5, and takes about
1.7e13 multiply-adds over the whole sample. The features route is still used below, as the check
of the event order.

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
  paper        at the paper's central point, numpy alone on the two output files reproduces the
               stored means and DELPHI histogram of output/three_bands_MIXGEOV3S9, before and
               after the calculation is imposed. The maximum-entropy step is written out below in
               numpy, step for step as directlib.Model.tilt_on.
After writing, both files are read back, every array is checked against its sha256 in the
metadata, and the paper check is run again on what was read.

    python build_reference.py          # on Symmetry, in /mnt/beegfs/bassi/reweighting/release_ref
Writes reference_MIX17.npz, targets_thrust_nlo.npz and build_reference.json next to itself.
"""
import os, sys, json, time, hashlib, zipfile, argparse, datetime, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/mnt/beegfs/bassi/reweighting/maxent_code'
EXPORT = f'{SRC}/output/models/MIXGEOV3S9_cond.npz'       # the trained export: a(Phi) and the head
REF = f'{SRC}/output/models/MIXGEOV3S9_ref.npz'           # the reference events' particles
SLIM = f'{SRC}/output/models/MIXGEOV3S9_ref_v2_slim.npz'  # their observables, exact thrust axis
DATA = f'{SRC}/data_stagePURE17_v3'                        # the training runs: meta.json, particles
TARGETS = f'{SRC}/output/thrust_targets_nlo_point.npz'    # the calculation, targets_nlo_point.py
BANDS = f'{SRC}/output/three_bands_MIXGEOV3S9'            # the paper's bands, .json and .npz

ap = argparse.ArgumentParser()
ap.add_argument('--gentune', default=f'{HERE}/check', help='directory that holds the gentune package')
ap.add_argument('--models', default=f'{HERE}/check/models', help='directory with the released MIX17_*.npz')
ap.add_argument('--out', default=HERE)
ap.add_argument('--version', default='1.0')
a = ap.parse_args()
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
# The moments of the calculation are <t^a (ln t)^b> inside the window, (a, b) as in
# targets_nlo_point.py.
POWERS = {'logt': (0, 1), 'tau': (1, 0), 'tlogt': (1, 1), 'log2t': (0, 2), 'tau2': (2, 0),
          'tlog2t': (1, 2), 't2logt': (2, 1), 'log3t': (0, 3), 'tau3': (3, 0), 'tlog3t': (1, 3),
          't3logt': (3, 1), 't2log2t': (2, 2), 'log4t': (0, 4), 'tau4': (4, 0)}
FLOOR_REL = 0.002       # the paper's diagonal floor of Sigma, Model(floor_rel=0.002) in three_bands.py
PER_RUN = 8             # events per run in the order check, 5760 in all
TOL_A = 1e-3            # a(Phi) from the trunk against the export, float32 pooling sits near 2e-5
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


def npy_shape(path, member):
    """The shape of one array inside an npz, read off its header without loading it."""
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        v = np.lib.format.read_magic(f)
        hdr = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
        return hdr(f)[0]


def lse(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


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


# ---- sources ---------------------------------------------------------------------------------
src_sha = {p: sha_file(p) for p in (EXPORT, REF, SLIM, f'{DATA}/meta.json', TARGETS,
                                    BANDS + '.json', BANDS + '.npz')}
rel = {f: f'{a.models}/{f}' for f in ('MIX17_head.npz', 'MIX17_trunk.npz', 'MIX17_selftest.npz')}
rel_sha = {f: sha_file(p) for f, p in rel.items()}
log('hashed the sources')
checks = {}

# ---- provenance: the released head is the export's ---------------------------------------------
c = np.load(EXPORT)
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
dlogit = float(np.max(np.abs(head.logit(A_sel, th_c) - head.logit(AE[:, sel], th_c))))
need(dA < TOL_A, f'trunk on the stored particles misses a(Phi) by {dA:.2e}')
need(dA_shift > 100*TOL_A, f'a shift by one event is not seen ({dA_shift:.2e})')
nch = ((P[..., 5] != 0) & Mk).sum(1).astype(np.int16)
need(np.array_equal(nch, s['nch']), 'nch recomputed from the particles differs')
lm = P[..., 3]
ink = (lm > -0.36) & (lm < -0.20) & Mk
strange = (ink.sum(1)/np.maximum(Mk.sum(1), 1)).astype(np.float32)
need(np.array_equal(strange, s['strange']), 'strange recomputed from the particles differs')
nstored = Mk.sum(1)
mt = s['mult_total']
below = nstored < 100                     # mult_total counts the full event, the particles stop at 100
checks['order'] = dict(events=len(sel), max_abs_dA=dA, max_abs_A=float(np.max(np.abs(AE[:, sel]))),
                       max_abs_dA_shifted_by_one=dA_shift, max_abs_dlogit_at_central=dlogit,
                       nch_equal_all_events=True, strange_equal_all_events=True,
                       events_truncated_at_100=int((~below).sum()),
                       mult_total_equals_stored_count_below_100=float(np.mean(mt[below] == nstored[below])))
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
    chain_summary=G['chain_summary'])
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

# ---- the paper, from the in-memory arrays --------------------------------------------------------
bands = json.load(open(BANDS + '.json'))
need(bands['export'].endswith('MIXGEOV3S9_cond.npz') and bands['grid'].endswith('thrust_targets_nlo_point.npz')
     and not bands['wifi_central'], 'the stored bands are not of this export, calculation and central network')
B = np.load(BANDS + '.npz')
pc = paper_check(ev, AE, tgt_arrays, head, bands, B)
worst = max(pc[x][y] for x in ('before', 'after') for y in ('max_rel_means', 'max_rel_hist'))
need(worst < TOL_PAPER, f'the paper is not reproduced: worst relative difference {worst:.2e}')
checks['paper'] = pc
log(f'paper: before {pc["before"]["max_rel_means"]:.1e} (means) {pc["before"]["max_rel_hist"]:.1e} (histogram), '
    f'after {pc["after"]["max_rel_means"]:.1e} {pc["after"]["max_rel_hist"]:.1e}, {pc["newton_iterations"]} Newton steps')

# ---- write, read back, check again --------------------------------------------------------------
build_sha = sha_file(os.path.abspath(__file__))
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
src_info = {os.path.relpath(p, SRC): sha for p, sha in src_sha.items()}
features_bytes = sum(i.compress_size for i in zipfile.ZipFile(REF).infolist() if i.filename in ('particles.npy', 'mask.npy'))
ref_meta = dict(
    name='reference_MIX17', version=a.version, built=stamp, build_script_sha256=build_sha,
    paper='B. Assi and J. Thaler, Combined Uncertainties from Event Generators and Precision QCD',
    N=N, ens=ENS, K=K, process='e+e- -> hadrons at 91.2 GeV',
    order=(f'run after run in run_ids, the training order of {os.path.relpath(DATA, SRC)}/meta.json, '
           f'{nper} events per run. Within run r the events are '
           f'numpy.random.default_rng(50000 + r).choice(n_r, {nper}, replace=False) of its n_r = 80000 '
           f'training events, in the order drawn. Event i of every array is event i of AE[:, i].'),
    pairs_with=dict(head='models/MIX17_head.npz', head_sha256=rel_sha['MIX17_head.npz'],
                    trunk='models/MIX17_trunk.npz', trunk_sha256=rel_sha['MIX17_trunk.npz'],
                    export='MIXGEOV3S9'),
    why_AE=(f'a(Phi) is shipped instead of the per-particle features ({features_bytes/1e9:.2f} GB compressed): '
            f'it is the array every number of the paper was computed from, and needs no trunk pass. '
            f'The released trunk on the features reproduces it to {dA:.1e}.'),
    source=src_info, keys=DOC,
    arrays={k: dict(dtype=str(v.dtype), shape=list(np.shape(v)), sha256=sha_array(v)) for k, v in ref_arrays.items()},
    checks=checks)
tgt_doc = dict(
    name='targets_thrust_nlo', version=a.version, built=stamp, build_script_sha256=build_sha,
    source={os.path.relpath(TARGETS, SRC): src_sha[TARGETS]},
    calculation=('NNLL+NLO thrust with a dispersive power correction, at the pair (alpha_s, alpha_0) of its own '
                 'fit to ALEPH in 0.05 < tau < 0.33 (output/thrust_chain.py, targets_nlo_point.py)'),
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
        np_labels='Milan factor +20 percent, -20 percent, third-order renormalon subtraction',
        milan='the central Milan factor', sub_order_central='order of the central renormalon subtraction',
        order='perturbative order of the fixed-order part (2: NLO)',
        chain_summary='the fit of the chain that fixed the point, JSON'),
    arrays={k: dict(dtype=str(np.asarray(v).dtype), shape=list(np.shape(v)), sha256=sha_array(v))
            for k, v in tgt_arrays.items()})
ref_arrays['meta'] = np.array(json.dumps(ref_meta, indent=1))
tgt_arrays['meta'] = np.array(json.dumps(tgt_doc, indent=1))

paths = {'reference_MIX17.npz': np.savez_compressed, 'targets_thrust_nlo.npz': np.savez}
tmp = {n: os.path.join(a.out, n.replace('.npz', '.partial.npz')) for n in paths}
paths['reference_MIX17.npz'](tmp['reference_MIX17.npz'], **ref_arrays)
paths['targets_thrust_nlo.npz'](tmp['targets_thrust_nlo.npz'], **tgt_arrays)
log('written, reading back')
del AE, ev, ref_arrays
rr = np.load(tmp['reference_MIX17.npz'])
tt = np.load(tmp['targets_thrust_nlo.npz'])
for z, doc in ((rr, ref_meta), (tt, tgt_doc)):
    m = json.loads(str(z['meta']))
    need(m == json.loads(json.dumps(doc)), 'metadata did not survive the round trip')
    for k, d in m['arrays'].items():
        need(sha_array(z[k]) == d['sha256'], f'{k} changed on the way to disk')
ev = {k: rr[k] for k in OBS}
pc2 = paper_check(ev, rr['AE'], tt, head, bands, B)
worst2 = max(pc2[x][y] for x in ('before', 'after') for y in ('max_rel_means', 'max_rel_hist'))
need(worst2 < TOL_PAPER, f'the paper is not reproduced from the files read back: {worst2:.2e}')
log(f'paper from the files read back: worst relative difference {worst2:.1e}')
for n, p in tmp.items():
    os.replace(p, os.path.join(a.out, n))
out = {n: dict(bytes=os.path.getsize(os.path.join(a.out, n)), sha256=sha_file(os.path.join(a.out, n))) for n in paths}
summary = dict(built=stamp, build_script='build_reference.py', build_script_sha256=build_sha, outputs=out,
               sources=src_info, release=rel_sha, numpy=np.__version__, python=platform.python_version(),
               checks=checks, paper_from_files_read_back=pc2, seconds=round(time.time() - t0))
json.dump(summary, open(os.path.join(a.out, 'build_reference.json'), 'w'), indent=1)
for n, d in out.items():
    log(f'{n}: {d["bytes"]/1e6:.1f} MB, sha256 {d["sha256"]}')
log('all checks passed')
