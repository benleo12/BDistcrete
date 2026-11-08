#!/usr/bin/env python3
"""Cost of the generator weights at a new parameter point (Sec. 3.6, Fig. 7) for the networks of the paper,
on the 1.15M-event reference sample of the seventeen-parameter mixture.

BENCH_MIX=linear (the default) times the linear mixture of the paper, (1 - xi) <O>_S + xi <O>_H: a point needs the
weights of both generators, l_S at theta_S and l_H at theta_H (the head at xi = 0 and at xi = 1), each normalized
and, in a fit step, tilted on its own; the derivatives of one prediction are one backward pass from the cotangents
of both (the fraction enters only through the mixing of the two predictions, not through a network). The
factorized network then makes two inner products per event instead of one; the other two designs already computed
both generators' outputs. Writes output/bench_final_<device>_<threads>_lin.json. BENCH_MIX=geometric is the earlier
draft's multiplicative mixture, timed as before (output/bench_final_<device>_<threads>.json), kept for comparison.
In both, each design is an ensemble of four networks, and in the geometric mixture of Eq. (geomix)
log w = (1 - xi) l_S + xi l_H:

  factorized     l = <a(Phi), b(theta)>: a(Phi) computed once and stored, so the mixture is one inner
                 product per event with (1 - xi) b_S + xi b_H (the paper's model, read from its export
                 and evaluated by directlib as in the fit)
  concatenation  r2_ladder.ConcatGeoMixture, the network of Table tab:accuracy: one network g sees each
                 generator's own parameters, zero-padded to the larger block, with a generator label
                 appended to every particle, so every new theta needs the whole network on every
                 particle, once per generator
  late           concat_baseline.LatePFN in the same mixture form: theta joins after the sum over
                 particles, so the pooled features are computed once and stored, and every new theta
                 needs the downstream network on every event, once per generator

Timed as the median over BENCH_REPS repeats after one warm-up:
  (c) the one-time pass that fills each design's store, and the memory each design keeps;
  (a) the weights of every reference event at a new theta;
  (n) the network part of a fit step: the logits and their gradient for a given cotangent;
  (g) a gradient check of each design against central finite differences, on float64 copies of the
      networks, with the concatenation gradient missing one network as a control the check must
      reject (on by default on the GPU);
  (b) on the CPU, one fit step through directlib: the chi^2 to the thrust distributions of ALEPH,
      DELPHI and OPAL above tau = 0.05 and the L3 multiplicity, and its gradient in the seventeen
      parameters, with the windowed thrust moments of Sec. 5 imposed again (linear: on each generator,
      directlib.Model.chi2_lin_at); and the part of it that is the same for every design (the
      maximum-entropy weights and the chi^2). Linear: the concatenation network's fit step is never
      timed (BENCH_SKIP=b_early in effect), since it is its network step (n) plus that same part.
The concatenation network is timed in its fastest form: particles packed (padded slots skipped), and
the pooled features of the forward pass kept for the gradient, which then needs one more pass of the
per-particle network with its derivative. Its repeats and warm-up are set apart (BENCH_EARLY_REPS,
BENCH_EARLY_WARMUP), and BENCH_SKIP=b_early leaves out its full fit step, which is its network step (n)
plus the design-independent part. The concatenation and late networks have untrained weights, drawn
with a fixed seed (the cost does not depend on them). Writes output/bench_final_<device>_<threads>.json
after every section.

    BENCH_DEVICE=cpu BENCH_THREADS=4 BENCH_EARLY_REPS=1 BENCH_EARLY_WARMUP=0 BENCH_SKIP=b_early python bench_cost_final.py
    BENCH_DEVICE=cuda BENCH_THREADS=12 python bench_cost_final.py
BENCH_REF (the reference bundle with the observables) and BENCH_PARTICLES (the bundle with the particles) override
the files of BENCH_TAG; BENCH_NEV (tests only) keeps the first BENCH_NEV reference events.
"""
import os, sys, json, time, copy
os.environ.setdefault('LADDER_ACT', 'silu')            # the activation the networks are trained with
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model, at_fraction               # sets the default dtype to float64, as in the fit
from r2_ladder import build_feats, CondPFN, ConcatPFN
from concat_baseline import LatePFN

MIX = os.environ.get('BENCH_MIX', 'linear'); assert MIX in ('linear', 'geometric'), MIX
LINEAR = MIX == 'linear'
DEV = os.environ.get('BENCH_DEVICE', 'cpu'); THREADS = int(os.environ.get('BENCH_THREADS', '4'))
REPS = int(os.environ.get('BENCH_REPS', '5')); CHUNK = int(os.environ.get('BENCH_CHUNK', '2000000'))
SKIP = set(os.environ.get('BENCH_SKIP', '').split(',')) | ({'b_early'} if LINEAR else set())
EARLY_REPS = int(os.environ.get('BENCH_EARLY_REPS', str(max(1, REPS - 2) if DEV == 'cpu' else REPS)))
EARLY_WARM = os.environ.get('BENCH_EARLY_WARMUP', '1') == '1'
GRADCHECK = os.environ.get('BENCH_GRADCHECK', '1' if DEV == 'cuda' else '0') == '1'
TAG = os.environ.get('BENCH_TAG', 'MIXGEOV3S9')
GRID = os.environ.get('TARGETS_GRID', 'output/thrust_targets_nlo_point.npz')
torch.set_num_threads(THREADS)
torch.manual_seed(int(os.environ.get('BENCH_SEED', '0')))      # the untrained weights, for a reproducible record
f32 = torch.float32
OUT = os.environ.get('BENCH_OUT', f'output/bench_final_{DEV}_{THREADS}' + ('_lin' if LINEAR else '') + '.json')
NEV = int(os.environ.get('BENCH_NEV', '0'))
res = dict(mix=MIX, device=DEV, threads=THREADS, reps=REPS, early_reps=EARLY_REPS, early_warmup=EARLY_WARM,
           host=os.uname().nodename, model=TAG, grid=GRID, skip=sorted(x for x in SKIP if x),
           torch=torch.__version__)
if DEV == 'cuda':
    res['gpu'] = torch.cuda.get_device_name(0)
else:
    try:
        res['cpu'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
    except Exception:
        pass
def save():
    json.dump(res, open(OUT, 'w'), indent=1)

def timed(fn, reps=REPS, warm=True):
    """median wall time of fn() over reps calls, after one warm-up call unless warm is False"""
    sync = DEV == 'cuda'
    if warm:
        fn()
    if sync: torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn()
        if sync: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(x) for x in ts]

# ---- the reference sample, the fit model and the networks -----------------------------------------
M = Model('MIX', export=f'output/models/{TAG}_cond.npz', ref=os.environ.get('BENCH_REF', f'output/models/{TAG}_ref_v2_slim.npz'),
          first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=GRID)
assert M.kind == 'mixture' and M.mix_form == 'geometric', (M.kind, M.mix_form)
if NEV:                                         # tests only: the first NEV events, every per-event array cut the same way
    M.AE = M.AE[:, :NEV].contiguous(); M.N = NEV
    M.thr, M.nch = M.thr[:NEV], M.nch[:NEV]; M.extra = {k: v[:NEV] for k, v in M.extra.items()}
    M.inwin = M.inwin[:NEV]; M.IW = torch.from_numpy(np.where(M.inwin)[0]); M.MW = torch.from_numpy(M.funs(M.thr[M.inwin]))
    M.NCHT = torch.from_numpy(M.nch); M.THRT = torch.from_numpy(M.thr)
    for x_ in M.exp:
        ins = x_['ins'].numpy(); k_ = ins < NEV
        x_['ins'] = torch.from_numpy(ins[k_]); x_['idx'] = x_['idx'][torch.from_numpy(k_)]
    res['n_events_test'] = NEV
assert np.allclose(M.NORM[-1], [0.5, 0.5]), M.NORM[-1]          # the fraction xi = 0.5 + 0.5 u
N, NT, ENS, T, NS, NH = M.N, M.NT, M.ENS, M.T, M.NS, M.NH
c = np.load(f'output/models/{TAG}_cond.npz')
mu, sd = torch.tensor(c['feat_mu'], dtype=f32), torch.tensor(c['feat_sd'], dtype=f32)
r = np.load(os.environ.get('BENCH_PARTICLES', f'output/models/{TAG}_ref.npz'))
t0 = time.perf_counter()
feats, mask = build_feats(r['particles'][:NEV or None], r['mask'][:NEV or None], True)     # (N, 100, 7), (N, 100)
assert len(mask) == N, (len(mask), N)
real = mask.bool(); EV = torch.nonzero(real, as_tuple=True)[0]            # event index of every real particle
FP = ((feats[real] - mu)/sd).to(f32)                                      # packed, standardized particles
del feats, r
res['n_events'], res['n_particles'] = int(N), int(len(FP)); res['prep_seconds'] = time.perf_counter() - t0
print(f'{N} events, {len(FP)} particles ({len(FP)/N:.1f} per event), d={NT} ({NS}+{NH}+1), K={M.K}, '
      f'device {DEV}, {THREADS} threads', flush=True)
FPd, EVd = FP.to(DEV), EV.to(DEV)
NPAD = max(NS, NH); NARG = NPAD + 2                     # a generator's parameters, zero-padded, and its label
def nets(cls):
    out = [cls(FP.shape[1], NARG).to(f32) for _ in range(ENS)]
    for m in out:
        m.eval(); m.emb_auto = False; m.emb_scale = 1.0     # a fixed constant, as after training
        for p in m.parameters():
            p.requires_grad_(False)
    return [m.to(DEV) for m in out]

def args(u):
    """the inputs of the mixture at standardized parameters u: each generator's own block, zero-padded,
    with its label (as ConcatGeoMixture._arg), and the mixing fraction"""
    zS = u.new_zeros(NPAD - NS); zH = u.new_zeros(NPAD - NH); one, nil = u.new_ones(1), u.new_zeros(1)
    aS = torch.cat([u[:NS], zS, one, nil]); aH = torch.cat([u[NS:NS + NH], zH, nil, one])
    return aS, aH, (0.5 + 0.5*u[-1]).clamp(0.0, 1.0)

# ---- the concatenation network: the whole network on every particle, once per generator ----------
class Concat:
    """The concatenation mixture on the packed particles X. The logits at u keep the pooled features of
    every network, and the gradient starts from them: the head and the fraction directly, then one pass
    of the per-particle network with its derivative, chunked over particles so that no graph over all
    particles is kept."""
    def __init__(s, members, X):
        s.members, s.X, s.keep, s.u = members, X, {}, None

    def pooled(s, m, a):
        E = torch.zeros(N, m.head[0].in_features, dtype=s.X.dtype, device=DEV)
        for i in range(0, len(s.X), CHUNK):
            x = s.X[i:i+CHUNK]
            E.index_add_(0, EVd[i:i+CHUNK], m.phi(torch.cat([x, a.expand(len(x), NARG)], 1)))
        return E

    def logit(s, u):
        u = u.to(s.X.dtype); aS, aH, xi = args(u)
        s.keep.clear(); out = 0
        with torch.no_grad():
            for i, m in enumerate(s.members):
                ES, EH = s.pooled(m, aS), s.pooled(m, aH)
                s.keep[i] = (ES, EH)
                out = out + (1 - xi)*m.head(ES).squeeze(-1) + xi*m.head(EH).squeeze(-1)
        s.u = u.detach().clone()
        return out/ENS/T

    def logit_lin(s, u):
        """The linear mixture: each generator's logits (l_S, l_H) from the same pooled features."""
        u = u.to(s.X.dtype); aS, aH, _ = args(u)
        s.keep.clear(); oS = oH = 0
        with torch.no_grad():
            for i, m in enumerate(s.members):
                ES, EH = s.pooled(m, aS), s.pooled(m, aH)
                s.keep[i] = (ES, EH)
                oS = oS + m.head(ES).squeeze(-1); oH = oH + m.head(EH).squeeze(-1)
        s.u = u.detach().clone()
        return oS/ENS/T, oH/ENS/T

    def vjp(s, u, g, members=None, gH=None):
        """d/du of sum_e g_e logit_e (members: the networks to include, all by default); with gH, the linear
        mixture: d/du of sum_e (g_e l_S,e + gH_e l_H,e)"""
        u, g = u.to(s.X.dtype), g.to(s.X.dtype)
        if s.u is None or not torch.equal(s.u, u):
            s.logit(u) if gH is None else s.logit_lin(u)
        with torch.enable_grad():
            aS, aH, _ = args(u)
            uu = u.detach().clone().requires_grad_(True)
            gu = torch.zeros(NT, dtype=s.X.dtype, device=DEV)
            for i in (range(len(s.members)) if members is None else members):
                m = s.members[i]; xi = (0.5 + 0.5*uu[-1]).clamp(0.0, 1.0)
                ES, EH = (E.detach().requires_grad_(True) for E in s.keep[i])
                if gH is None:
                    tot = ((1 - xi)*m.head(ES).squeeze(-1) + xi*m.head(EH).squeeze(-1)) @ g
                    gS, gH_, gx = torch.autograd.grad(tot, [ES, EH, uu])
                    gu += gx
                else:
                    tot = m.head(ES).squeeze(-1) @ g + m.head(EH).squeeze(-1) @ gH.to(s.X.dtype)
                    gS, gH_ = torch.autograd.grad(tot, [ES, EH])
                for a, ge, off, k in ((aS, gS, 0, NS), (aH, gH_, NS, NH)):
                    ga = torch.zeros(NARG, dtype=s.X.dtype, device=DEV)
                    for j in range(0, len(s.X), CHUNK):
                        x = s.X[j:j+CHUNK]; aa = a.detach().clone().requires_grad_(True)
                        h = m.phi(torch.cat([x, aa.expand(len(x), NARG)], 1))
                        ga += torch.autograd.grad((h*ge[EVd[j:j+CHUNK]]).sum(), [aa])[0]
                    gu[off:off + k] += ga[:k]
            return gu/ENS/T

early = Concat(nets(ConcatPFN), FPd)
late = nets(LatePFN)
res['weights_per_network'] = dict(concatenation=sum(p.numel() for p in early.members[0].parameters()),
                                  late=sum(p.numel() for p in late[0].parameters()))

class EarlyLogits(torch.autograd.Function):
    """the concatenation network's logits as a function of the physical parameters, for directlib"""
    @staticmethod
    def forward(ctx, theta):
        u = ((theta - M.CEN)/M.HW).to(f32).to(DEV); ctx.save_for_backward(u)
        return early.logit(u).to('cpu', torch.float64)
    @staticmethod
    def backward(ctx, g):
        u, = ctx.saved_tensors
        return early.vjp(u, g.to(f32).to(DEV)).to('cpu', torch.float64)/M.HW

# ---- the late network: the pooled features once, the downstream network on every event ----------
def pooled_once(m):
    E = torch.zeros(N, m.head[0].in_features - NARG, dtype=f32, device=DEV)
    with torch.no_grad():
        for i in range(0, len(FPd), CHUNK):
            E.index_add_(0, EVd[i:i+CHUNK], m.phi(FPd[i:i+CHUNK]))
    return E

def late_logit_t(u, P, members=None):
    aS, aH, xi = args(u)
    return sum((1 - xi)*m.head(torch.cat([p, aS.expand(N, NARG)], 1)).squeeze(-1)
               + xi*m.head(torch.cat([p, aH.expand(N, NARG)], 1)).squeeze(-1)
               for m, p in zip(late if members is None else members, P))/ENS/T

def late_logit_lin(u, P, members=None, only=None):
    """The linear mixture: (l_S, l_H), or the one generator named by only ('S' or 'H')."""
    aS, aH, _ = args(u); nets = late if members is None else members
    one = lambda a: sum(m.head(torch.cat([p, a.expand(N, NARG)], 1)).squeeze(-1) for m, p in zip(nets, P))/ENS/T
    return one(aS) if only == 'S' else one(aH) if only == 'H' else (one(aS), one(aH))

# ---- the factorized network on the device, the arithmetic of directlib.Model.logit ----------------
AEd = M.AE.to(DEV) if DEV != 'cpu' else M.AE
def fact_logit_t(theta, AE=None):
    AE = AEd if AE is None else AE
    tn = (theta - M.CEN.to(theta))/M.HW.to(theta); xi = theta[-1]; acc = 0
    for m in range(ENS):
        bS = M._mlp([(W.to(theta), b.to(theta)) for W, b in M.NET_S[m]], tn[:NS])
        bH = M._mlp([(W.to(theta), b.to(theta)) for W, b in M.NET_H[m]], tn[NS:NS + NH])
        acc = acc + (AE[m] @ ((1 - xi)*bS + xi*bH).to(AE.dtype)).double()
    return acc/ENS/T

def fact_logit_lin(theta, AE=None):
    """The linear mixture: (l_S, l_H), one inner product per event for each generator."""
    AE = AEd if AE is None else AE
    tn = (theta - M.CEN.to(theta))/M.HW.to(theta); aS = aH = 0
    for m in range(ENS):
        bS = M._mlp([(W.to(theta), b.to(theta)) for W, b in M.NET_S[m]], tn[:NS])
        bH = M._mlp([(W.to(theta), b.to(theta)) for W, b in M.NET_H[m]], tn[NS:NS + NH])
        aS = aS + (AE[m] @ bS.to(AE.dtype)).double(); aH = aH + (AE[m] @ bH.to(AE.dtype)).double()
    return aS/ENS/T, aH/ENS/T

# ---- (c) one-time passes and memory ---------------------------------------------------------------
gb = lambda x: x.numel()*x.element_size()/1e9
store = {}
if 'c' not in SKIP:
    trunk = [CondPFN(FP.shape[1], NT, K=M.K).to(f32).to(DEV).eval() for _ in range(ENS)]
    def fill_factorized():
        with torch.no_grad():
            for m in trunk:
                E = torch.zeros(N, m.latent, dtype=f32, device=DEV)
                for i in range(0, len(FPd), CHUNK):
                    E.index_add_(0, EVd[i:i+CHUNK], m.phi(FPd[i:i+CHUNK]))
                m.A(E)
    t_fact, _ = timed(fill_factorized, reps=1)
    del trunk
    t_late, _ = timed(lambda: store.__setitem__('P', [pooled_once(m) for m in late]), reps=1)
    res['one_time_seconds'] = dict(factorized=t_fact, late=t_late, concatenation=0.0)
P = store.get('P') or [pooled_once(m) for m in late]
res['store_GB'] = dict(factorized=gb(M.AE), late=sum(gb(p) for p in P), concatenation_particles=gb(FP) + gb(EV),
                       concatenation_kept_features=2*ENS*N*early.members[0].head[0].in_features*4/1e9)
print('one-time [s] and memory [GB]:', res.get('one_time_seconds'), res['store_GB'], flush=True)
save()

# ---- (a) the weights at a new parameter point ------------------------------------------------------
rng = np.random.default_rng(1)
u_np = rng.uniform(-0.8, 0.8, NT)
th = torch.tensor(M.CEN.numpy() + M.HW.numpy()*u_np); u = torch.tensor(u_np, dtype=f32, device=DEV)
a = {}
# on the CPU the fit's own code (directlib.Model.logit), on the GPU the same arithmetic on the device
if LINEAR:                                                        # both generators' weights at the point
    a['factorized'] = (timed(lambda: (M.logit(at_fraction(th, 0.0)), M.logit(at_fraction(th, 1.0)))) if DEV == 'cpu'
                       else timed(lambda: fact_logit_lin(th.to(DEV))))
    with torch.no_grad():
        a['late'] = timed(lambda: late_logit_lin(u, P))
    if 'early' not in SKIP:
        a['concatenation'] = timed(lambda: early.logit_lin(u), reps=EARLY_REPS, warm=EARLY_WARM)
else:
    a['factorized'] = timed(lambda: M.logit(th)) if DEV == 'cpu' else timed(lambda: fact_logit_t(th.to(DEV)))
    with torch.no_grad():
        a['late'] = timed(lambda: late_logit_t(u, P))
    if 'early' not in SKIP:
        a['concatenation'] = timed(lambda: early.logit(u), reps=EARLY_REPS, warm=EARLY_WARM)
res['a_weights_seconds'] = {k: v[0] for k, v in a.items()}; res['a_all'] = {k: v[1] for k, v in a.items()}
print('(a) weights at a new point [s]:', res['a_weights_seconds'], flush=True)
save()

# ---- (n) the network part of a fit step: the logits and their gradient for a given cotangent ------
g = torch.tensor(rng.normal(size=N), dtype=f32, device=DEV)
gH = torch.tensor(rng.normal(size=N), dtype=f32, device=DEV) if LINEAR else None   # linear: the cotangent of l_H
def n_fact():
    t = th.to(DEV).clone().requires_grad_(True)
    if LINEAR:
        lS, lH = fact_logit_lin(t); (lS.to(f32) @ g + lH.to(f32) @ gH).backward()
    else:
        (fact_logit_t(t).to(f32) @ g).backward()
def n_late():
    uu = u.clone().requires_grad_(True)
    if LINEAR:
        lS, lH = late_logit_lin(uu, P); (lS @ g + lH @ gH).backward()
    else:
        (late_logit_t(uu, P) @ g).backward()
def n_early():
    if LINEAR:
        early.logit_lin(u); early.vjp(u, g, gH=gH)
    else:
        early.logit(u); early.vjp(u, g)
n = {'factorized': timed(n_fact), 'late': timed(n_late)}
if 'early' not in SKIP:
    n['concatenation'] = timed(n_early, reps=EARLY_REPS, warm=EARLY_WARM)
res['n_network_step_seconds'] = {k: v[0] for k, v in n.items()}; res['n_all'] = {k: v[1] for k, v in n.items()}
print('(n) logits and gradient [s]:', res['n_network_step_seconds'], flush=True)
save()

# ---- (g) gradient check against central finite differences ----------------------------------------
def gradcheck():
    """each design's gradient code against a finite difference along a random direction, on float64
    copies of the networks and stores: the logits are large against their dependence on theta (the
    pooled features enter unscaled), so in float32 the finite difference is dominated by rounding and
    by the order of the atomic sums over particles. The difference is the Richardson combination of
    central differences at steps EPS and EPS/2, which removes their EPS^2 error (large for the
    untrained concatenation network, whose head is sharply curved in theta)"""
    v = torch.tensor(rng.normal(size=NT), dtype=torch.float64, device=DEV); v /= v.norm(); EPS = 1e-3
    gd = g.double(); u64 = u.double(); gHd = None if gH is None else gH.double()
    def fd(F):
        with torch.no_grad():
            D = lambda h: float((F(u64 + h*v) - F(u64 - h*v))/(2*h))
            return (4*D(EPS/2) - D(EPS))/3
    chk = {}
    AE64 = AEd.double(); CEN, HW = M.CEN.to(DEV), M.HW.to(DEV)
    th_of = lambda x: CEN + HW*x
    dot = (lambda l: l[0] @ gd + l[1] @ gHd) if LINEAR else (lambda l: l @ gd)    # the cotangents of the logits
    flog = fact_logit_lin if LINEAR else fact_logit_t
    t = th_of(u64).clone().requires_grad_(True); dot(flog(t, AE64)).backward()
    chk['factorized'] = (float(t.grad @ (HW*v)), fd(lambda x: dot(flog(th_of(x), AE64))))
    del AE64
    late64 = [copy.deepcopy(m).double() for m in late]; P64 = [p.double() for p in P]
    llog = late_logit_lin if LINEAR else late_logit_t
    uu = u64.clone().requires_grad_(True); dot(llog(uu, P64, late64)).backward()
    chk['late'] = (float(uu.grad @ v), fd(lambda x: dot(llog(x, P64, late64))))
    del late64, P64
    if 'early' not in SKIP:
        cat64 = Concat([copy.deepcopy(m).double() for m in early.members], FPd.double())
        fe = fd(lambda x: dot(cat64.logit_lin(x) if LINEAR else cat64.logit(x)))
        chk['concatenation'] = (float(cat64.vjp(u64, gd, gH=gHd) @ v), fe)
        chk['control_concatenation_one_network_left_out'] = (float(cat64.vjp(u64, gd, members=range(ENS - 1), gH=gHd) @ v), fe)
        del cat64
    res['gradcheck'] = {k: dict(gradient=x, finite_difference=y, rel_diff=abs(x - y)/max(abs(y), 1e-12),
                                passes=abs(x - y)/max(abs(y), 1e-12) < 1e-3) for k, (x, y) in chk.items()}
    ctrl = res['gradcheck'].get('control_concatenation_one_network_left_out')
    res['gradcheck_ok'] = (all(x['passes'] for k, x in res['gradcheck'].items() if not k.startswith('control'))
                           and (ctrl is None or not ctrl['passes']))
    print('(g) gradient check (float64):', {k: f"{x['rel_diff']:.1e}" for k, x in res['gradcheck'].items()},
          'OK' if res['gradcheck_ok'] else 'FAILED', flush=True)
if GRADCHECK:
    try:                                       # a failure here must not cost the fit-step timing below
        gradcheck()
    except Exception as e:
        res['gradcheck_error'] = repr(e); print('(g) gradient check raised', repr(e), flush=True)
    save()

# ---- (b) one fit step through directlib, CPU only: the chi^2 and its gradient, moments imposed ----
if DEV == 'cpu' and 'b' not in SKIP:
    u0 = rng.uniform(-0.5, 0.5, NT)
    step = M.chi2_lin_at if LINEAR else M.chi2_at                  # linear: both generators tilted, the mixture's chi^2
    b = {'factorized': timed(lambda: step(u0, 0, 0, None, grad=True))}
    def with_logit(fn, early_design=False):
        orig = M.logit; M.logit = fn
        try:
            return timed(lambda: step(u0, 0, 0, None, grad=True),
                         reps=EARLY_REPS if early_design else REPS, warm=EARLY_WARM if early_design else True)
        finally:
            M.logit = orig
    gen_of = lambda theta: 'S' if float(theta[-1]) < 0.5 else 'H'   # the head at xi = 0 (Sherpa) or 1 (Herwig)
    b['late'] = with_logit((lambda theta: late_logit_lin(((theta - M.CEN)/M.HW).to(f32), P, only=gen_of(theta)).double()) if LINEAR
                           else (lambda theta: late_logit_t(((theta - M.CEN)/M.HW).to(f32), P).double()))
    if 'early' not in SKIP and 'b_early' not in SKIP:
        b['concatenation'] = with_logit(EarlyLogits.apply, early_design=True)
    res['b_fit_step_seconds'] = {k: v[0] for k, v in b.items()}; res['b_all'] = {k: v[1] for k, v in b.items()}
    th0 = torch.tensor(M.CEN.numpy() + M.HW.numpy()*u0)
    if LINEAR:
        lgS, lgH = (x_.detach() for x_ in fact_logit_lin(th0))
    else:
        lg0 = fact_logit_t(th0).detach()
    def rest():
        orig = M.logit
        if LINEAR:                                                 # no network, gradient defined
            M.logit = lambda theta: (lgS if gen_of(theta) == 'S' else lgH).clone() + 0*theta.sum()
        else:
            L = lg0.clone(); M.logit = lambda theta: L + 0*theta.sum()
        try:
            step(u0, 0, 0, None, grad=True)
        finally:
            M.logit = orig
    res['b_tilt_and_chi2_seconds'] = timed(rest)[0]
    print('(b) fit step [s]:', res['b_fit_step_seconds'], ' tilt and chi2 alone:', res['b_tilt_and_chi2_seconds'], flush=True)

save(); print('wrote', OUT)
