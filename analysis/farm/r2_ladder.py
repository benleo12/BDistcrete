#!/usr/bin/env python3
"""R2: the unified conditional ladder (Stages A, B, C) under the FROZEN recipe.
One code path, one recipe for every stage; produces output/ladder_final.json and exports
each stage's conditional (cached A-vectors on a reference set + numpy B-net) to
output/models/<stage>_cond.npz for R3/R4.

FROZEN RECIPE (identical across stages): full-event thrust-frame representation; 4-model
ensemble; K=24; 12000 steps; cosine LR 1e-3->1e-5; Adam.

THE RULER (rewritten after an adversarial review of the first version found the metric could
be satisfied by degenerate weights rather than by a good model):
 * width = sqrt(mean(pull^2)), the RMS about ZERO, not np.std(). A standard deviation is
   taken about the sample mean and therefore subtracts exactly the coherent bin-to-bin offset
   that a normalization error produces, hiding it.
 * both histograms are normalized to their mass INSIDE the bin range, because the reference
   and a single-theta target lose different tail fractions outside it, which would otherwise
   put a common offset in every bin.
 * errors are propagated on both sides, so a correct model has width 1 BY CONSTRUCTION. The
   null is 1. Checkpoints are therefore selected on |width - 1|, never on the smallest width:
   minimizing the width alone rewards over-dispersed weights, whose inflated sigma shrinks
   every pull, i.e. it selects for the most degenerate model available.
 * a bin is used only if BOTH sides hold at least MIN_COUNT (effective) events, since a bin
   with one event on one side and none on the other returns a pull of exactly +/-1 and drags
   the answer toward the null mechanically.
 * integer observables get integer bin edges, so the multiplicity-like distributions are not
   chopped into mostly-empty float bins.
 * two controls accompany every quoted number: N_eff = 1/sum(w^2) of the reference weights
   (a width is uninterpretable if the weights have collapsed), and the width obtained with
   UNIFORM weights (if that is no worse, the conditional is doing nothing).
 * the same-run calibration check replaces the old "floor": two disjoint halves of one held
   run compared through the identical estimator and the identical bin edges. It should also
   be 1; its deviation measures how well calibrated the ruler itself is.
Early stopping uses a stop split that is event-disjoint from the reported split.
Usage: python r2_ladder.py A|B|C   (or `all`)."""
import sys, os, json
import numpy as np, torch, torch.nn as nn, pandas as pd
DEV = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ENS, K, STEPS = 4, 24, 36000
# The rank K of the factorized head <a(Phi), b(theta)> is overridable so that a run at the
# PHYSICAL rank predicted by the score expansion (K_2 = 1 + d + d(d+1)/2 per stage) can use
# this exact code path with nothing else changed. Unset LADDER_K reproduces the production 24.
K = int(os.environ.get('LADDER_K', '24'))
# Optional suffix on the exported model filenames, so a non-production K never clobbers the
# production output/models/<stage>_cond.npz. Unset LADDER_SUFFIX reproduces the production path.
SUF = os.environ.get('LADDER_SUFFIX', '')
# Optional results file. Unset LADDER_OUT reproduces the production output/ladder_final.json.
OUT_JSON = os.environ.get('LADDER_OUT', 'output/ladder_final.json')
# Committed recipe: the standard PFN with SiLU activation (LADDER_ACT, a documented energyflow
# option) at lr 1e-3 on STANDARDIZED inputs. The story of how this was settled: at the
# published width (256 latent) with the package-default ReLU, the flavor stages plateaued near
# 1.8 with the baryon direction badly mis-weighted (per-observable 3.5), independent of lr
# (3e-4 and 1e-3 both). SiLU on the same sizes recovered it to ~1.25 and still descending,
# because ReLU dead units on the sparse flavor features (baryon number, charge) killed exactly
# the direction those stages depend on. The step budget is 36000 so the flavor stages, which
# converge more slowly than the kinematic Stage A, reach their floor rather than stopping at
# the edge of the budget.
LR = float(os.environ.get('LADDER_LR', '1e-3'))
# Unset LADDER_SEED_OFFSET reproduces the published seeds 0..ENS-1.
SEED_OFFSET = int(os.environ.get('LADDER_SEED_OFFSET', '0'))
# Unset LADDER_FEATS reproduces the published 7-feature flavor stages.
FEATS = os.environ.get('LADDER_FEATS', 'full')
STAGES_ENV = os.environ.get('LADDER_STAGES', '')
# The budget was raised from 12000 once, for every stage together, because at 12000 Stage B
# selected the FINAL checkpoint while still descending monotonically (1.535 -> 1.207). A rung
# that stops at the edge of its own budget measures the budget, not the difficulty of the
# stage, and would have made the 2-parameter rung look harder than the 3-parameter one. The
# recipe stays identical across stages; only the common budget changed.
CKPTS = [6000, 12000, 18000, 24000, 30000, 33000, 36000]
NREF_PER = 12000
MIN_COUNT = 5          # per-bin occupancy required on BOTH sides
NBIN = 26
os.makedirs('output/models', exist_ok=True)

# observables binned on their value support (with sparse-tail merging) rather than on an
# equal-width grid. Integer counts belong here, and so does the strange fraction: it is
# lattice-valued (a ratio k/n of small integers), its spikes fall essentially ON equal-width
# bin edges, and float32/float64 rounding then flips entire spikes across an edge, producing
# giant anticorrelated adjacent pulls that measure the binning rather than the physics
# (observed: the same model and weights read 0.75 or 5.3 depending on value dtype).
DISCRETE = {'mult_total', 'nbaryon', 'strange'}

# ---------- stage configuration ----------
ASV_A = [0.11000, 0.11167, 0.11333, 0.11500, 0.11667, 0.11833, 0.12000, 0.12167, 0.12333,
         0.12500, 0.12667, 0.12833, 0.13000, 0.10667, 0.13333]                       # incl margin
STAGES = {
 'A': dict(data='data_stageA_full', flavor=False, ntheta=1,
           train={6100+i: (a,) for i, a in enumerate(ASV_A[:13])} | {6113: (0.10667,), 6114: (0.13333,)},
           held={6200: (0.11250,), 6201: (0.11583,), 6202: (0.11917,), 6203: (0.12250,),
                 6204: (0.12583,), 6205: (0.12917,)},
           norm=[(0.120, 0.010)], obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'],
           flav_obs=[]),
 'B': dict(data='data_stageB_full', flavor=True, ntheta=2,
           train={6600+i*4+j: (a, b) for i, a in enumerate([0.1100, 0.1133, 0.1167, 0.1200, 0.1233, 0.1267, 0.1300])
                  for j, b in enumerate([0.05, 0.15, 0.25, 0.35])},
           held={6700: (0.1175, 0.10), 6701: (0.1225, 0.30), 6702: (0.1150, 0.20),
                 6703: (0.1280, 0.30), 6704: (0.1200, 0.15)},
           norm=[(0.120, 0.012), (0.20, 0.18)],
           obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'], flav_obs=['nbaryon']),
 'C': dict(data='data_stageC', flavor=True, ntheta=3,
           train={6800+i: t for i, t in enumerate([(a, s, k) for a in [0.112, 0.120, 0.128]
                  for s in [0.30, 0.46, 0.65] for k in [0.80, 1.21, 1.80]])},
           held={6900: (0.116, 0.38, 1.00), 6901: (0.124, 0.55, 1.50), 6902: (0.120, 0.46, 1.21),
                 6903: (0.114, 0.60, 0.95), 6904: (0.126, 0.35, 1.60)},
           norm=[(0.120, 0.008), (0.475, 0.175), (1.30, 0.50)],
           obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'], flav_obs=['nbaryon', 'strange']),
}


def build_feats(p, mask, flavor, eps=1e-6):
    cos = p[..., 1]; phi = p[..., 2]; z = p[..., 0]
    cols = [cos, phi, np.log(np.clip(z, eps, None)), np.log(np.clip(1 - cos**2 + eps, eps, None))]
    if flavor:
        # FEATS='mass' drops the two signed species tags and keeps only log10 m, testing
        # whether the mass column alone carries the species information they encode.
        cols += [p[..., 3]] if FEATS == 'mass' else [p[..., 3], p[..., 4], p[..., 5]]
    feats = np.stack(cols, -1) * mask[..., None]
    return torch.tensor(feats.astype(np.float32)), torch.tensor(mask.astype(np.float32))


def fit_feature_norm(fs, ms, max_ev=4000):
    """Per-feature mean and standard deviation over REAL particles, pooled across runs.

    The published PFN uses ReLU with he_uniform initialization, which assumes standardized
    inputs. Our raw features are not: log z reaches about -10 and log sin^2(theta) further, and
    feeding those to the published architecture trains badly (measured: closure stuck near 2.1
    instead of converging to 1). The old bespoke network used SiLU and a narrow latent, which
    tolerated the raw scale, which is why this only surfaced on adopting the standard one.

    The statistics MUST be pooled over all runs and applied identically. Normalizing each run
    against its own mean would remove exactly the between-run differences that carry the
    parameter dependence the classifier is meant to learn."""
    C = fs[0].shape[-1]
    tot = torch.zeros(C, dtype=torch.float64); tot2 = torch.zeros(C, dtype=torch.float64); n = 0.0
    for f, m in zip(fs, ms):
        sel = m[:max_ev].bool()
        v = f[:max_ev][sel].double()
        tot += v.sum(0); tot2 += (v*v).sum(0); n += v.shape[0]
    mu = tot/n; sd = torch.sqrt(torch.clamp(tot2/n - mu*mu, min=1e-12))
    return mu.float(), sd.float()


def apply_feature_norm(f, m, mu, sd):
    """Standardize, then re-zero the padded slots so pooling still ignores them."""
    return ((f - mu)/sd) * m.unsqueeze(-1)


def strange_frac(p, mask):
    lm = p[..., 3]; m = mask.astype(bool); ink = ((lm > -0.36) & (lm < -0.20)) & m
    return ink.sum(1) / np.maximum(m.sum(1), 1)


# Published Particle Flow Network hyperparameters (Komiske, Metodiev, Thaler, 1810.05165),
# as documented for the energyflow package: Phi and F layer sizes, ReLU activations on every
# layer including the latent, he_uniform initialization. The latent dimension is the last
# entry of PHI_SIZES. These are NOT tuned here; they are the published values.
PHI_SIZES = (100, 100, 256)
F_SIZES = (100, 100, 100)
# The activation is a documented argument of the energyflow PFN (Phi_acts / F_acts); ReLU is
# the package default but swish/SiLU is supported. It is selected here by closure, not invented.
ACT_NAME = os.environ.get('LADDER_ACT', 'relu')
_ACTS = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}

# ---------------------------------------------------------------------------------------
# Options that address the event-side collapse of a(Phi). EVERY default below reproduces the
# published behaviour bit for bit, in the same style as LADDER_K / LADDER_SEED_OFFSET.
#
# LADDER_ADDITIVE=1 makes the two rank-one pieces that the score expansion says the log ratio
# carries ANYWAY explicit:  f = <a(Phi), b(theta)> + g(Phi) + c(theta).  g is the
# theta-independent event offset (the pooled mixture reference is not any single theta) and c
# is the event-independent normalization.  It is implemented by widening A and B by ONE output
# each and contracting the extra pair against a constant 1, so the bilinear part is still
# exactly K-dimensional and nothing else in the network changes.
ADDITIVE = os.environ.get('LADDER_ADDITIVE', '0') == '1'
# LADDER_PHI_NORM=ln puts a LayerNorm before every activation of the PER-PARTICLE map, so its
# pre-activations cannot drift into the region where SiLU and its derivative both vanish.
PHI_NORM = os.environ.get('LADDER_PHI_NORM', 'none')
# LADDER_HEAD_NORM=ln does the same inside the head A that turns the pooled embedding into
# a(Phi). This is the one aimed at the measured failure: pooling is a SUM over ~44 particles,
# so A's first layer receives pre-activations with a standard deviation near 20 instead of the
# unit scale he_uniform assumes, and that layer dies of SiLU saturation. It is applied to A
# only, never to B: B reads standardized theta of order one and has no such problem, and
# leaving it a plain MLP keeps the exported B weights in the Linear-only format every
# downstream reader (ab_analysis, reeval_bins, wave2_cond, ...) already understands.
HEAD_NORM = os.environ.get('LADDER_HEAD_NORM', 'none')
# LADDER_INIT_GAIN scales the he_uniform weights of the per-particle map at initialization.
INIT_GAIN = float(os.environ.get('LADDER_INIT_GAIN', '1.0'))
# LADDER_EMB_SCALE divides the POOLED embedding by a constant before the head reads it. Pooling
# is a SUM over ~44 particles, so E is ~20x larger than the unit-scale input he_uniform assumes,
# and the head's first layer starts with a large fraction of its units already in the region
# where SiLU and its derivative both vanish. This is a pure rescaling: the head's first layer is
# linear, so the constant can be absorbed into its weights and the function class is UNCHANGED.
# 'auto' measures the constant at initialization so that the head's first-layer pre-activations
# have unit standard deviation, which is the same standardization fit_feature_norm already
# applies to the raw inputs, applied one level later where sum pooling undoes it.
EMB_SCALE = os.environ.get('LADDER_EMB_SCALE', '1.0')
EMB_SCALE = EMB_SCALE if EMB_SCALE == 'auto' else float(EMB_SCALE)
# LADDER_WD > 0 switches Adam for AdamW with that decoupled weight decay. An L2 penalty on both
# factors of a bilinear form has a genuine restoring force along the scale degeneracy: at fixed
# product |a||b| it is minimized at |a| = |b|, which is exactly the direction that drifts.
WD = float(os.environ.get('LADDER_WD', '0.0'))


def _mlp(inp, sizes, out=None, norm='none'):
    """Dense stack with the configured activation after every listed layer, optional linear
    output layer. `norm='ln'` inserts a LayerNorm between each Linear and its activation."""
    Act = _ACTS[ACT_NAME]
    layers = []; d = inp
    for h in sizes:
        layers += [nn.Linear(d, h)]
        if norm == 'ln':
            layers += [nn.LayerNorm(h)]
        layers += [Act()]; d = h
    if out is not None:
        layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers), d


class CondPFN(nn.Module):
    """The standard PFN of Komiske, Metodiev and Thaler, with one stated deviation.

    Event side: a per-particle network Phi, masked SUM pooling over particles, then the
    downstream network F, at the published sizes (100,100,256) and (100,100,100) with ReLU and
    he_uniform initialization. The energyflow package implements exactly this, but its
    architectures require a TensorFlow/Keras backend that is not installed here, and more
    importantly its PFN emits a single classification output.

    Deviation: F's final layer emits K numbers which are contracted with a parameter network
    b(theta), because one network has to serve every theta rather than one network per
    parameter point. The rank of that contraction is the quantity the K-scan measures, so this
    head is the method rather than a hyperparameter choice, and no published package provides
    a theta-conditioned PFN."""
    def __init__(s, C, ntheta, K=24, phi_sizes=PHI_SIZES, f_sizes=F_SIZES,
                 additive=None, phi_norm=None, init_gain=None, head_norm=None, emb_scale=None):
        super().__init__()
        es = EMB_SCALE if emb_scale is None else emb_scale
        s.emb_auto = (es == 'auto')
        s.emb_scale = 1.0 if s.emb_auto else float(es)
        # None means "take the module-level default", which is the env flag, which defaults to
        # the published behaviour. Passing them explicitly lets a sweep vary them in-process.
        s.additive = ADDITIVE if additive is None else bool(additive)
        s.phi_norm = PHI_NORM if phi_norm is None else phi_norm
        s.head_norm = HEAD_NORM if head_norm is None else head_norm
        gain = INIT_GAIN if init_gain is None else float(init_gain)
        s.K = K
        s.phi, L = _mlp(C, phi_sizes, norm=s.phi_norm)
        nout = K + 1 if s.additive else K
        s.A, _ = _mlp(L, f_sizes, nout, norm=s.head_norm)
        s.B, _ = _mlp(ntheta, f_sizes, nout)
        s.latent = L
        for mod in s.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')   # he_uniform
                nn.init.zeros_(mod.bias)
        if gain != 1.0:
            with torch.no_grad():
                for mod in s.phi.modules():
                    if isinstance(mod, nn.Linear):
                        mod.weight.mul_(gain)
    def emb(s, f, m):
        # Lazy autoscale: satellite trainers (rank scan, wide box, showers, disjoint halves)
        # build CondPFN directly and never call autoscale(), so under LADDER_EMB_SCALE=auto the
        # first batch fixes the constant here. Deterministic given the batch stream, a no-op
        # once set, and identical in effect to the explicit call in run_stage.
        if s.emb_auto and s.emb_scale == 1.0:
            s.autoscale(f, m)
        B, P, _ = f.shape
        E = (s.phi(f.reshape(B*P, -1)).reshape(B, P, -1) * m.unsqueeze(-1)).sum(1)
        return E if s.emb_scale == 1.0 else E/s.emb_scale
    @torch.no_grad()
    def autoscale(s, f, m):
        """Fix emb_scale ONCE, at initialization, so the head's first layer sees pre-activations
        of unit standard deviation. Computes the RAW pooled embedding inline rather than through
        emb(), which under lazy triggering would recurse. Idempotent: emb_auto drops after the
        first call, so later calls (or the explicit one in run_stage) return the stored scale."""
        if not s.emb_auto: return s.emb_scale
        B, P, _ = f.shape
        E = (s.phi(f.reshape(B*P, -1)).reshape(B, P, -1) * m.unsqueeze(-1)).sum(1)
        lin0 = [l for l in s.A if isinstance(l, nn.Linear)][0]
        s.emb_scale = float(lin0(E).std())
        print(f'emb_scale=auto -> {s.emb_scale:.2f}', flush=True)
        s.emb_auto = False
        return s.emb_scale
    def f_from(s, E, th):
        a = s.A(E); b = s.B(th)
        if s.additive:
            # <a, b> + g(Phi) + c(theta), i.e. the augmented vectors [a, g, 1] and [b, 1, c]
            return (a[..., :s.K]*b[..., :s.K]).sum(-1) + a[..., s.K] + b[..., s.K]
        return (a*b).sum(-1)


EMB_CHUNK = 10000


def emb_all(m, f, mask, chunk=EMB_CHUNK):
    """Pool a large reference in chunks.

    The per-particle activations are (events x particles x latent), so at the published latent
    of 256 a 180k-event reference needs ~18 GB in one shot and exhausts the GPU. Chunking over
    events changes nothing numerically because the pooling sum is per event."""
    out = []
    with torch.no_grad():
        for i in range(0, len(f), chunk):
            out.append(m.emb(f[i:i+chunk], mask[i:i+chunk]))
    return torch.cat(out)


MIN_BIN_FRAC = 0.002        # a bin must hold this fraction of the combined sample


def make_bins(vals_all, obs):
    """Bin edges shared by the closure, the controls and the calibration check.

    Discrete observables are binned on their ACTUAL support rather than on a unit-spaced
    integer grid. The baryon count, for instance, only takes even values because baryons are
    produced in pairs, so a unit grid leaves every odd bin empty by construction and silently
    throws away half the resolution of the test. Sparse values in the tail are merged into
    their neighbour so that every bin can meet the occupancy requirement instead of being
    dropped, which would discard those events entirely."""
    v = np.asarray(vals_all)
    if obs in DISCRETE:
        # canonicalize the value lattice before taking the support: mixed float32 (stored
        # reference) and float64 (freshly computed target) values of the SAME lattice point
        # differ at the 1e-8 level and would otherwise become distinct "support values" with
        # an edge between them, splitting one physical spike across two bins. Real lattice
        # spacings here are > 1e-3, so rounding at 1e-5 merges dtype jitter only.
        u, c = np.unique(np.round(v, 5), return_counts=True)
        if len(u) == 1:
            return np.array([u[0]-0.5, u[0]+0.5])
        # merge sparse values (from both tails inward) until each group is populated enough
        frac = c/c.sum(); groups = [[i] for i in range(len(u))]
        gf = list(frac)
        while len(groups) > 2:
            i = int(np.argmin(gf))
            if gf[i] >= MIN_BIN_FRAC: break
            j = i+1 if i == 0 else (i-1 if i == len(groups)-1 else
                                    (i-1 if gf[i-1] <= gf[i+1] else i+1))
            lo_, hi_ = min(i, j), max(i, j)
            groups[lo_] = groups[lo_] + groups[hi_]; gf[lo_] = gf[lo_] + gf[hi_]
            del groups[hi_]; del gf[hi_]
        # edges at the midpoints between the outermost members of adjacent groups
        edges = [u[groups[0][0]] - 0.5]
        for a, b in zip(groups[:-1], groups[1:]):
            edges.append((u[a[-1]] + u[b[0]]) / 2.0)
        edges.append(u[groups[-1][-1]] + 0.5)
        return np.array(edges, dtype=float)
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo: hi = lo + 1e-6
    return np.linspace(lo, hi, NBIN)


def hist_dens(o, w, bins):
    """In-range-normalized density, the effective count per bin, and the effective total.
    Weighted case uses Kish effective counts so that a weighted sample is described by the
    statistics it actually carries rather than by its raw length."""
    bw = np.diff(bins)
    if w is None:
        n, _ = np.histogram(o, bins); N = float(n.sum())
        if N <= 0: return np.zeros(len(bw)), np.zeros(len(bw)), 0.0
        return n/(N*bw), n.astype(float), N
    h, _ = np.histogram(o, bins, weights=w); h2, _ = np.histogram(o, bins, weights=w**2)
    W = float(h.sum()); W2 = float(h2.sum())
    if W <= 0 or W2 <= 0: return np.zeros(len(bw)), np.zeros(len(bw)), 0.0
    neff = np.where(h2 > 0, h**2/np.where(h2 > 0, h2, 1.0), 0.0)     # effective count per bin
    return h/(W*bw), neff, W*W/W2                                    # in-range effective total


def pulls(oref, w, otgt, bins):
    """Per-bin pulls with errors from BOTH sides. A correct model gives RMS 1.

    The bin variance uses the POOLED density estimate rather than each side's own observed
    count. Taking sigma from the observed count makes an upward-fluctuating bin carry a
    correspondingly larger error, which compresses the pull and biases the whole width low
    (measured: 0.96 instead of 1.00 on identical samples)."""
    bw = np.diff(bins)
    pa, na, Na = hist_dens(oref, w, bins)
    pb, nb, Nb = hist_dens(otgt, None, bins)
    if Na <= 0 or Nb <= 0: return np.array([]), 0
    # Each side contributes its own per-bin variance, carrying the multinomial (1-q) factor.
    # The bin occupancies are constrained to sum to the sample size, so treating them as free
    # Poisson draws inflates sigma and biases the width low. Using each side's own effective
    # per-bin count (rather than a global effective size) matters once the weights correlate
    # with the bin, which is exactly what reweighting does.
    # The pooled bin probability must come from the DENSITIES, not from the effective counts.
    # Once the weights correlate with the bin (which is precisely what reweighting does), the
    # ratio na/Na is no longer the bin probability, and using it inflates sigma and drags the
    # width below its null.
    qa = pa*bw; qb = pb*bw
    qhat = (Na*qa + Nb*qb) / (Na + Nb)
    safe_a = np.where(na > 0, na, 1.0); safe_b = np.where(nb > 0, nb, 1.0)
    var = qhat**2 * (1.0 - qhat) * (1.0/safe_a + 1.0/safe_b) / bw**2
    sig = np.sqrt(np.maximum(var, 0.0))
    keep = (sig > 0) & (na >= MIN_COUNT) & (nb >= MIN_COUNT)
    if keep.sum() == 0: return np.array([]), 0
    return (pa[keep]-pb[keep])/sig[keep], int(keep.sum())


def width_of(p, nbins=None):
    """sqrt(chi^2/ndf) about ZERO, with ndf = nbins - 1.

    One degree of freedom is subtracted because both densities are normalized over the same
    bins, so the pulls are not free: that constraint alone pulls a naive RMS below 1. Squaring
    first and dividing by ndf also avoids the Jensen bias of averaging square roots. With
    multinomial errors and this ndf the null value is 1."""
    if not len(p): return float('nan')
    ndf = max((nbins if nbins is not None else len(p)) - 1, 1)
    return float(np.sqrt(np.sum(p**2)/ndf))


def run_stage(tag):
    cfg = STAGES[tag]; DATA = cfg['data']; OBS = cfg['obs'] + cfg['flav_obs']
    norm = np.array(cfg['norm'])
    def tn(theta): return ((np.asarray(theta) - norm[:, 0]) / norm[:, 1]).astype(np.float32)
    # load
    P = {}; SH = {}; NB = {}; STR = {}
    for rid in list(cfg['train']) + list(cfg['held']):
        d = np.load(f'{DATA}/particles_full_{rid:04d}.npz')
        P[rid] = (d['particles'], d['mask']); SH[rid] = pd.read_csv(f'{DATA}/shapes_run_{rid:04d}.csv')
        # the CSV and the npz are indexed by the same event order; a mismatch would silently
        # pair one event's shape with another event's particles
        assert len(SH[rid]) == len(d['mask']), f'{rid}: csv {len(SH[rid])} vs npz {len(d["mask"])}'
        if 'nbaryon' in cfg['flav_obs']: NB[rid] = d['nbaryon'].astype(np.float32)
        if 'strange' in cfg['flav_obs']: STR[rid] = strange_frac(d['particles'], d['mask']).astype(np.float32)
    def obsvals(rid, o, idx=None):
        v = (SH[rid][o].values if o in cfg['obs'] else (NB[rid] if o == 'nbaryon' else STR[rid]))
        return v if idx is None else v[idx]
    # features (train on CPU, reference on GPU)
    tf = {}; tm = {}
    for rid in cfg['train']:
        tf[rid], tm[rid] = build_feats(*P[rid], cfg['flavor'])
    # standardize with statistics pooled over all training runs (see fit_feature_norm)
    FMU, FSD = fit_feature_norm([tf[r] for r in cfg['train']], [tm[r] for r in cfg['train']])
    for rid in cfg['train']:
        tf[rid] = apply_feature_norm(tf[rid], tm[rid], FMU, FSD)
    print(f'[{tag}] feature norm mu={np.round(FMU.numpy(),2)} sd={np.round(FSD.numpy(),2)}')
    rng = np.random.default_rng(0)
    rf, rm, robs, rpart, rpmask = [], [], {o: [] for o in OBS}, [], []
    for rid in cfg['train']:
        n = len(tf[rid]); idx = rng.choice(n, min(NREF_PER, n), replace=False)
        rf.append(tf[rid][idx]); rm.append(tm[rid][idx])
        rpart.append(P[rid][0][idx]); rpmask.append(P[rid][1][idx])
        for o in OBS: robs[o].append(obsvals(rid, o, idx))
    ref_particles = np.concatenate(rpart); ref_pmask = np.concatenate(rpmask)
    ref_f = torch.cat(rf).to(DEV); ref_m = torch.cat(rm).to(DEV)
    robs = {o: np.concatenate(v) for o, v in robs.items()}; Nref = len(ref_f)
    Cdim = ref_f.shape[-1]
    for rid in cfg['train']:                       # particles are no longer needed once features exist
        P[rid] = None
    # disjoint stop / report split of each held run
    split = {rid: np.random.default_rng(1000+rid).permutation(len(SH[rid])) for rid in cfg['held']}
    stop_idx = {rid: split[rid][:len(split[rid])//2] for rid in cfg['held']}
    rep_idx = {rid: split[rid][len(split[rid])//2:] for rid in cfg['held']}
    # bins are fixed ONCE per (held run, observable) from reference+report, then reused by the
    # closure, the uniform control and the calibration check so all three are commensurate
    BINS = {(rid, o): make_bins(np.r_[robs[o], obsvals(rid, o, rep_idx[rid])], o)
            for rid in cfg['held'] for o in OBS}
    print(f'[{tag}] ref {Nref}, C={Cdim}, {len(cfg["train"])} train, {len(cfg["held"])} held')

    models, opts, schs, gens = [], [], [], []
    for sd0 in range(ENS):
        sd = sd0 + SEED_OFFSET
        torch.manual_seed(sd); m = CondPFN(Cdim, cfg['ntheta'], K=K).to(DEV); models.append(m)
        if m.emb_auto:
            print(f'[{tag}] seed {sd} emb_scale=auto -> {m.autoscale(ref_f[:4096], ref_m[:4096]):.2f}')
        opts.append(torch.optim.Adam(m.parameters(), LR) if WD == 0.0 else
                    torch.optim.AdamW(m.parameters(), LR, weight_decay=WD))
        schs.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[-1], STEPS, eta_min=LR/100))
        gens.append(torch.Generator().manual_seed(sd))
    bce = nn.BCEWithLogitsLoss(); lab = torch.cat([torch.zeros(1024), torch.ones(1024)]).to(DEV)
    rids = list(cfg['train'])

    def embeddings():
        """Reference embeddings do not depend on theta, so they are computed once per pass."""
        Es = []
        for m in models:
            m.eval(); Es.append(emb_all(m, ref_f, ref_m)); m.train()
        return Es

    def logits_at(Es, theta):
        th = torch.tensor(tn(theta), device=DEV).float().expand(Nref, cfg['ntheta'])
        with torch.no_grad():
            # the ensemble is combined as the MEAN OF LOGITS (a geometric mean of density
            # ratios); downstream reloads must use the same rule
            return np.mean([m.f_from(E, th).cpu().numpy() for m, E in zip(models, Es)], 0)

    def weights_from(f, T=1.0):
        w = np.exp((f - f.max())/T); w /= w.sum()
        return w, float(1.0/np.sum(w**2))               # weights and their effective count

    def closure(which, T=1.0, want_controls=False, Es=None):
        idxs = stop_idx if which == 'stop' else rep_idx
        if Es is None: Es = embeddings()
        out = {}; ctrl = {}; uni = np.full(Nref, 1.0/Nref)
        for rid, theta in cfg['held'].items():
            w, neff = weights_from(logits_at(Es, theta), T); po = {}; cu = {}
            for o in OBS:
                ot = obsvals(rid, o, idxs[rid]); b = BINS[(rid, o)]
                p, nb = pulls(robs[o], w, ot, b)
                po[o] = width_of(p)
                if want_controls:
                    pu, _ = pulls(robs[o], uni, ot, b)
                    cu[o] = dict(width_uniform=width_of(pu), nbins=nb,
                                 pull_mean=float(np.mean(p)) if len(p) else float('nan'))
            out[rid] = po
            if want_controls: ctrl[rid] = dict(N_eff=neff, per_obs=cu)
        return (out, ctrl) if want_controls else out

    def master(cl): return float(np.mean([np.mean(list(po.values())) for po in cl.values()]))

    def fit_temperature(Es):
        """Post-hoc temperature calibration (Guo et al. 2017): rescale the log-ratio by T so
        the softmax weights exp(f/T) are correctly sharp. T is chosen on the STOP split by
        minimizing |width - 1| and applied to the disjoint REPORT split, so it can never see
        the reported number. T=1 is the identity, so this only helps a mis-scaled ratio."""
        grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
        best_T, best_d = 1.0, None
        for T in grid:
            d = abs(master(closure('stop', T=T, Es=Es)) - 1.0)
            if best_d is None or d < best_d: best_d, best_T = d, T
        return best_T

    step = 0; best = None; traj = []
    for ck in CKPTS:
        for mi, m in enumerate(models):
            opt, sch, g = opts[mi], schs[mi], gens[mi]
            for _ in range(ck - step):
                j = rids[int(torch.randint(len(rids), (1,), generator=g))]
                ir = torch.randint(Nref, (1024,), generator=g); it = torch.randint(len(tf[j]), (1024,), generator=g)
                fb = torch.cat([ref_f[ir], tf[j][it].to(DEV)]); mb = torch.cat([ref_m[ir], tm[j][it].to(DEV)])
                th = torch.tensor(tn(cfg['train'][j]), device=DEV).float().expand(2048, cfg['ntheta'])
                loss = bce(m.f_from(m.emb(fb, mb), th), lab); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        step = ck
        ms = master(closure('stop'))
        traj.append((ck, ms))
        # select on |width - 1|: the null is 1, and minimizing the width itself would reward
        # over-dispersed weights whose inflated errors shrink every pull
        score = abs(ms - 1.0)
        if best is None or score < best[0]:
            best = (score, ck, ms, [{kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}
                                    for m in models])
        print(f'[{tag}] step {ck}: stop-width={ms:.3f} (|w-1|={score:.3f})')
    for m, st in zip(models, best[3]): m.load_state_dict(st)
    print(f'[{tag}] selected step {best[1]} with stop-width {best[2]:.3f}; reporting on disjoint split')

    Es_final = embeddings()
    Traw = master(closure('report', T=1.0, Es=Es_final))
    T = fit_temperature(Es_final)
    print(f'[{tag}] temperature T={T} (fit on stop split); raw report width={Traw:.3f}')
    rep, ctrl = closure('report', T=T, want_controls=True, Es=Es_final)
    # calibration check: two disjoint halves of the SAME held run, identical estimator and
    # identical bins. This should also be 1; its deviation is the ruler's own miscalibration.
    calib = {o: [] for o in OBS}
    for o in OBS:
        for rid in cfg['held']:
            ot = obsvals(rid, o, rep_idx[rid]); b = BINS[(rid, o)]
            for r in range(3):
                pr = np.random.default_rng(1_000_000 + 1000*r + rid).permutation(len(ot))
                h = len(ot)//2
                o1, o2 = ot[pr[:h]], ot[pr[h:]]
                p, _ = pulls(o1, np.full(len(o1), 1.0/len(o1)), o2, b)
                calib[o].append(width_of(p))
    calib_med = {o: float(np.nanmedian(calib[o])) for o in OBS}
    calib_sd = {o: float(np.nanstd(calib[o])) for o in OBS}
    per_obs = {o: float(np.mean([rep[rid][o] for rid in cfg['held']])) for o in OBS}
    uni_obs = {o: float(np.mean([ctrl[rid]['per_obs'][o]['width_uniform'] for rid in cfg['held']]))
               for o in OBS}
    result = dict(master=master(rep), master_raw=Traw, temperature=T,
                  rank_K=K, best_step=best[1], stop_trajectory=traj,
                  per_obs=per_obs, width_uniform=uni_obs,
                  gain_over_uniform={o: round(uni_obs[o]/per_obs[o], 2) for o in OBS},
                  calibration_check=calib_med, calibration_spread=calib_sd,
                  N_eff={str(cfg['held'][rid]): ctrl[rid]['N_eff'] for rid in cfg['held']},
                  N_ref=Nref, min_count=MIN_COUNT,
                  nbins_used={o: int(np.mean([ctrl[rid]['per_obs'][o]['nbins'] for rid in cfg['held']]))
                              for o in OBS},
                  pull_mean={o: float(np.mean([ctrl[rid]['per_obs'][o]['pull_mean'] for rid in cfg['held']]))
                             for o in OBS},
                  per_point={str(cfg['held'][rid]): rep[rid] for rid in cfg['held']})
    # export: cached A-vectors + B nets + the conventions needed to rebuild f, plus a checksum
    Es = embeddings()
    with torch.no_grad():
        AE = np.stack([np.concatenate([m.A(E[i:i+EMB_CHUNK]).cpu().numpy()
                                       for i in range(0, len(E), EMB_CHUNK)])
                       for m, E in zip(models, Es)])                           # (ENS, Nref, K)
    if ADDITIVE:
        # Export the AUGMENTED vectors [a, g, 1] and [b, 1, c] so that every downstream reader,
        # which forms AE @ b(theta) and knows nothing about the offsets, still reconstructs
        # f = <a,b> + g + c exactly. K is the CONTRACTION width; K_bilinear is the rank.
        AE = np.concatenate([AE, np.ones((AE.shape[0], AE.shape[1], 1), AE.dtype)], -1)
    exp = dict(AE=AE, norm=norm, ntheta=cfg['ntheta'], ens=ENS, K=AE.shape[-1], K_bilinear=K,
               additive=int(ADDITIVE), phi_norm=PHI_NORM, head_norm=HEAD_NORM,
               init_gain=INIT_GAIN, weight_decay=WD, emb_scale=str(EMB_SCALE),
               emb_scale_used=np.array([m.emb_scale for m in models]),
               act=ACT_NAME, nlayers=len(F_SIZES)+1, combine='mean_logits', temperature=T,
               feat_mu=FMU.numpy(), feat_sd=FSD.numpy(),
               phi_sizes=np.array(PHI_SIZES), f_sizes=np.array(F_SIZES))
    for mi, m in enumerate(models):
        Ws = [l.weight.detach().cpu().numpy() for l in m.B if isinstance(l, nn.Linear)]
        bs = [l.bias.detach().cpu().numpy() for l in m.B if isinstance(l, nn.Linear)]
        if ADDITIVE:
            # insert the constant-1 output between b and c, matching the extra column above
            Ws[-1] = np.concatenate([Ws[-1][:K], np.zeros((1, Ws[-1].shape[1]), Ws[-1].dtype),
                                     Ws[-1][K:K+1]], 0)
            bs[-1] = np.concatenate([bs[-1][:K], np.ones(1, bs[-1].dtype), bs[-1][K:K+1]])
        for li, (W, b) in enumerate(zip(Ws, bs)):
            exp[f'B{mi}_W{li}'] = W; exp[f'B{mi}_b{li}'] = b
    chk_theta = list(cfg['held'].values())[0]
    th = torch.tensor(tn(chk_theta), device=DEV).float().expand(Nref, cfg['ntheta'])
    with torch.no_grad():
        exp['f_checksum'] = np.mean([m.f_from(E, th).cpu().numpy() for m, E in zip(models, Es)], 0)[:256]
    exp['f_checksum_theta'] = np.array(chk_theta, dtype=float)
    # Replay the export the way every downstream reader does (AE @ b(theta), numpy, no torch)
    # and require it to agree with the model. This is what makes the additive offsets safe:
    # they are folded into the exported vectors, so a reader that knows nothing about them is
    # still exact, and if that ever stops being true the run fails here rather than silently.
    _relu = lambda z: np.maximum(z, 0.0)
    _npact = {'relu': _relu, 'silu': lambda z: z/(1+np.exp(-z)),
              'gelu': lambda z: 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))}[ACT_NAME]
    _nl = sum(1 for kk in exp if kk.startswith('B0_W'))
    _fr = np.zeros(256)
    for mi in range(ENS):
        x = tn(chk_theta).reshape(1, -1).astype(np.float64)
        for li in range(_nl):
            x = x @ exp[f'B{mi}_W{li}'].T + exp[f'B{mi}_b{li}']
            if li < _nl-1: x = _npact(x)
        _fr += exp['AE'][mi, :256] @ x.ravel()
    _fr /= ENS
    _err = float(np.max(np.abs(_fr - exp['f_checksum'])))
    print(f'[{tag}] export replay max|df| = {_err:.2e}')
    assert _err < 1e-3, f'{tag}: exported AE @ b(theta) does not reproduce f (max err {_err:.2e})'
    # A member whose logit is constant is a dead network. Averaging it into the ensemble
    # divides the live signal by ENS/(live members), which the temperature then silently
    # absorbs. Stage B shipped with one such member once; never again unnoticed. The sds are
    # always recorded, and the artifact is saved before we raise, so nothing is lost.
    with torch.no_grad():
        _th = torch.tensor(tn(chk_theta), device=DEV).float().expand(Nref, cfg['ntheta'])
        _sds = [float(_m.f_from(_E, _th).cpu().numpy().std()) for _m, _E in zip(models, Es)]
    exp['member_logit_sd'] = np.array(_sds)
    for _mi, _sd in enumerate(_sds):
        print(f'[{tag}] member {_mi} logit sd = {_sd:.3e}')
    np.savez_compressed(f'output/models/{tag}{SUF}_cond.npz', **exp)
    _dead = [i for i, v in enumerate(_sds) if v < 1e-4]
    if _dead:
        raise RuntimeError(
            f'{tag} members {_dead} are DEAD (logit sd < 1e-4). The export was saved for '
            'inspection, but do not use it: retrain those seeds with LADDER_SEED_OFFSET.')
    np.savez_compressed(f'output/models/{tag}{SUF}_ref.npz',
                        particles=ref_particles, mask=ref_pmask,
                        **{o: robs[o] for o in OBS})
    print(f'[{tag}] RESULT width={result["master"]:.3f} (null 1.0)  calib={ {o: round(calib_med[o],2) for o in OBS} }')
    print(f'[{tag}]   per_obs={ {o: round(per_obs[o],2) for o in OBS} }')
    print(f'[{tag}]   uniform-weight control={ {o: round(uni_obs[o],2) for o in OBS} }  (>1 means the model helps)')
    print(f'[{tag}]   N_eff={ {k: int(v) for k, v in result["N_eff"].items()} } of {Nref}')
    return result


def main():
    which = STAGES_ENV or (sys.argv[1] if len(sys.argv) > 1 else 'all')
    stages = ['A', 'B', 'C'] if which == 'all' else list(which)
    out = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for s in stages:
        out[s] = run_stage(s)
        json.dump(out, open(OUT_JSON, 'w'), indent=1)
    print('R2 LADDER DONE:', {s: round(out[s]['master'], 2) for s in stages})


if __name__ == '__main__':
    main()
