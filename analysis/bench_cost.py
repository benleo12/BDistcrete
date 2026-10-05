#!/usr/bin/env python3
"""Cost of the generator weights in the fit of Sec. 6, for three designs of the conditional
classifier on the 1.15M-event reference sample of the seventeen-parameter mixture, each a
four-network ensemble of the same size:

  factorized  f = <a(Phi), b(theta)>: a(Phi) computed once and stored, one inner product per event
              (the paper's network, read from its export as in the fit)
  early       theta appended to every particle, the concatenation network of Sec. 5.8 (DCTR):
              nothing can be stored, every new theta needs the whole network on every particle
  late        theta appended after the sum over particles: the pooled features are computed once
              and stored, every new theta needs the downstream network on every event

Timed, as the median over BENCH_REPS repeats after one warm-up:
  (a) the weights of every reference event at a new theta;
  (b) one step of the fit: the chi^2 and its gradient in the seventeen nuisance parameters, with
      the tilt solved again, through the same directlib code the fit uses (CPU);
  (n) the network part of (b) alone: the logits and their gradient for a given cotangent, which is
      what changes between the designs (also on the GPU, BENCH_DEVICE=cuda);
  (c) the one-time pass that fills each design's cache, and the memory each design keeps.
  (g) a gradient check: for each design, the directional derivative of g . logits from its timed
      gradient code against central finite differences, and as a positive control the concatenation
      gradient with one network left out, which the check must reject (BENCH_GRADCHECK, on by
      default on the GPU and at 32 CPU threads).
For the early design the particles are packed (padded slots skipped), its fastest form; the network
weights are the trained ones where available (output/models/concat_MIX.pt, concat_MIX_late.pt), and
cost does not depend on them. Writes output/bench_cost_<device>_<threads>.json after every section,
so a run that hits its time limit keeps what it measured.

    BENCH_DEVICE=cpu BENCH_THREADS=4 python bench_cost.py output/profile_MIX17ext_central.json
"""
import os, sys, json, time
# time the networks with the activation they were trained with (r2_ladder defaults to ReLU)
os.environ.setdefault('LADDER_ACT', 'silu')
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model                       # sets the default dtype to float64, as in the fit
from r2_ladder import build_feats, CondPFN
from concat_baseline import ConcatPFN, LatePFN

DEV = os.environ.get('BENCH_DEVICE', 'cpu'); THREADS = int(os.environ.get('BENCH_THREADS', '4'))
REPS = int(os.environ.get('BENCH_REPS', '5')); CHUNK = int(os.environ.get('BENCH_CHUNK', '2000000'))
SKIP = set(os.environ.get('BENCH_SKIP', '').split(','))
torch.set_num_threads(THREADS)
f32 = torch.float32
src = sys.argv[1]; fit = json.load(open(src))
OUT = os.environ.get('BENCH_OUT', f'output/bench_cost_{DEV}_{THREADS}.json')
GRADCHECK = os.environ.get('BENCH_GRADCHECK', '1' if (DEV == 'cuda' or THREADS == 32) else '0') == '1'
res = dict(device=DEV, threads=THREADS, reps=REPS, host=os.uname().nodename,
           early_reps=None, early_warmup=None, skip=sorted(x for x in SKIP if x))
if DEV == 'cuda':
    res['gpu'] = torch.cuda.get_device_name(0)
def save():
    json.dump(res, open(OUT, 'w'), indent=1)

# The concatenation network takes minutes to hours per call on a CPU node, so its repeats and its
# warm-up call can be set separately (BENCH_EARLY_REPS, BENCH_EARLY_WARMUP=0), and BENCH_SKIP=b_early
# leaves out its full fit step, which equals its network step (n) plus the design-independent tilt
# and chi^2 time measured below.
EARLY_REPS = int(os.environ.get('BENCH_EARLY_REPS', str(max(1, REPS - 2) if DEV == 'cpu' else REPS)))
EARLY_WARM = os.environ.get('BENCH_EARLY_WARMUP', '1') == '1'
res.update(early_reps=EARLY_REPS, early_warmup=EARLY_WARM)

def timed(fn, reps=REPS, sync=DEV == 'cuda', warm=True):
    """median wall time of fn() over reps calls, after one warm-up call unless warm is False"""
    if warm:
        fn()
    if sync: torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn()
        if sync: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(x) for x in ts]

# ---- the reference sample, the fit model and the networks ----------------------------------------
M = Model(fit['tag'], export=fit['export'], ref=fit['ref'], first_bin=fit['first_bin'], floor_rel=fit['floor_rel'],
          nch_data=tuple(fit['nch_data']), ae_dtype='float32', grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid_ext.npz'),
          last_bin=fit.get('last_bin'), use_nch=fit.get('use_nch', True), mix_form=fit.get('mix_form', 'additive'))
N, NT, ENS, T = M.N, M.NT, M.ENS, M.T
c = np.load(fit['export'])
mu, sd = torch.tensor(c['feat_mu'], dtype=f32), torch.tensor(c['feat_sd'], dtype=f32)
r = np.load(os.environ.get('BENCH_PARTICLES', 'output/models/MIX17aug_ref.npz'))
t0 = time.perf_counter()
feats, mask = build_feats(r['particles'], r['mask'], True)               # (N, 100, 7), (N, 100)
real = mask.bool(); EV = torch.nonzero(real, as_tuple=True)[0]          # event index of every real particle
FP = ((feats[real] - mu)/sd).to(f32)                                    # packed, standardized particles
del feats
res['n_events'], res['n_particles'] = int(N), int(len(FP)); res['prep_seconds'] = time.perf_counter() - t0
assert len(mask) == N
print(f'{N} events, {len(FP)} particles ({len(FP)/N:.1f} per event), device {DEV}, {THREADS} threads', flush=True)

def load_nets(cls, path):
    nets = [cls(FP.shape[1], NT).to(f32) for _ in range(ENS)]
    if os.path.exists(path):
        for m, st in zip(nets, torch.load(path, map_location='cpu')):
            m.load_state_dict(st)
        print(f'{cls.__name__}: trained weights from {path}', flush=True)
    else:
        print(f'{cls.__name__}: {path} not found, untrained weights (cost is the same)', flush=True)
    for m in nets:
        m.eval(); m.emb_auto = False; m.emb_scale = 1.0           # a fixed constant, as after training
    return [m.to(DEV) for m in nets]
early = load_nets(ConcatPFN, 'output/models/concat_MIX.pt')
late = load_nets(LatePFN, 'output/models/concat_MIX_late.pt')
FPd, EVd = FP.to(DEV), EV.to(DEV)
res['weights_per_network'] = dict(early=sum(p.numel() for p in early[0].parameters()),
                                  late=sum(p.numel() for p in late[0].parameters()))

# ---- logits of each design at standardized parameters u (the networks take u = (theta - centre)/half-width)
def early_pooled(m, u):
    E = torch.zeros(N, m.head[0].in_features, dtype=f32, device=DEV)
    for i in range(0, len(FPd), CHUNK):
        x = FPd[i:i+CHUNK]
        E.index_add_(0, EVd[i:i+CHUNK], m.phi(torch.cat([x, u.expand(len(x), NT)], 1)))
    return E

def early_logit(u):
    with torch.no_grad():
        return sum(m.head(early_pooled(m, u)).squeeze(-1) for m in early)/ENS/T

def early_vjp(u, g, nets=None):
    """d/du of sum_e g_e logit_e, chunked over particles so no graph over all particles is kept"""
    gu = torch.zeros(NT, dtype=f32, device=DEV)
    for m in (early if nets is None else nets):
        with torch.no_grad():
            E = early_pooled(m, u)
        E.requires_grad_(True)
        (m.head(E).squeeze(-1) @ g).backward(); gE = E.grad/ENS/T
        for i in range(0, len(FPd), CHUNK):
            x = FPd[i:i+CHUNK]; uu = u.detach().clone().requires_grad_(True)
            h = m.phi(torch.cat([x, uu.expand(len(x), NT)], 1))
            (h*gE[EVd[i:i+CHUNK]]).sum().backward(); gu += uu.grad
    return gu

def pooled_once(m):
    E = torch.zeros(N, m.head[0].in_features - NT, dtype=f32, device=DEV)
    with torch.no_grad():
        for i in range(0, len(FPd), CHUNK):
            E.index_add_(0, EVd[i:i+CHUNK], m.phi(FPd[i:i+CHUNK]))
    return E

def late_logit_t(u, P):
    return sum(m.head(torch.cat([p, u.expand(N, NT)], 1)).squeeze(-1) for m, p in zip(late, P))/ENS/T

class EarlyLogits(torch.autograd.Function):
    """the early design's logits as a function of the physical parameters for directlib's autograd"""
    @staticmethod
    def forward(ctx, theta):
        u = ((theta - M.CEN)/M.HW).to(f32).to(DEV); ctx.save_for_backward(u)
        return early_logit(u).to('cpu', torch.float64)
    @staticmethod
    def backward(ctx, g):
        u, = ctx.saved_tensors
        return (early_vjp(u, g.to(f32).to(DEV)).to('cpu', torch.float64)/M.HW)

# factorized logits on the device: a(Phi) stored, b(theta) from the small parameter networks
AEd = M.AE.to(DEV) if DEV != 'cpu' else M.AE
def fact_logit_t(theta):
    tn = (theta - M.CEN.to(theta))/M.HW.to(theta); fr = theta[-1]; acc = 0
    for m in range(ENS):
        bS = M._mlp([(W.to(theta), b.to(theta)) for W, b in M.NET_S[m]], tn[:M.NS])
        bH = M._mlp([(W.to(theta), b.to(theta)) for W, b in M.NET_H[m]], tn[M.NS:M.NS + M.NH])
        lS, lH = AEd[m] @ bS.to(AEd.dtype), AEd[m] @ bH.to(AEd.dtype)
        big = torch.maximum(lS, lH)
        acc = acc + big + torch.log(torch.clamp((1 - fr)*torch.exp(lS - big) + fr*torch.exp(lH - big), min=1e-30))
    return acc/ENS/T

# ---- (c) one-time passes and memory ------------------------------------------------------------
cache = {}
gb = lambda x: x.numel()*x.element_size()/1e9
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
    t_late, _ = timed(lambda: cache.__setitem__('P', [pooled_once(m) for m in late]), reps=1)
    res['one_time_seconds'] = dict(factorized=t_fact, late=t_late, early=0.0)
P = cache.get('P') or [pooled_once(m) for m in late]
res['cache_GB'] = dict(factorized=gb(M.AE), late=sum(gb(p) for p in P),
                       early_particles=gb(FP) + gb(EV), padded_particle_array=N*100*7*4/1e9)
print('one-time and memory:', res.get('one_time_seconds'), res['cache_GB'], flush=True)
save()

# ---- (a) the weights at a new parameter point --------------------------------------------------
rng = np.random.default_rng(1)
def new_point():
    u = rng.uniform(-0.8, 0.8, NT); th = M.CEN.numpy() + M.HW.numpy()*u
    return torch.tensor(th), torch.tensor(u, dtype=f32, device=DEV)
th, u = new_point()
a = {}
# on the CPU the fit's own code path (directlib.Model.logit), on the GPU the same arithmetic on the device
a['factorized'] = timed(lambda: M.logit(th)) if DEV == 'cpu' else timed(lambda: fact_logit_t(th.to(DEV)))
with torch.no_grad():
    a['late'] = timed(lambda: late_logit_t(u, P))
if 'early' not in SKIP:
    a['early'] = timed(lambda: early_logit(u), reps=EARLY_REPS, warm=EARLY_WARM)
res['a_weights_seconds'] = {k: v[0] for k, v in a.items()}; res['a_all'] = {k: v[1] for k, v in a.items()}
print('(a) weights at a new point [s]:', res['a_weights_seconds'], flush=True)
save()

# ---- (n) the network part of a fit step: logits and their gradient for a given cotangent -------
g = torch.tensor(rng.normal(size=N), dtype=f32, device=DEV)
def n_fact():
    t = th.to(DEV).clone().requires_grad_(True); (fact_logit_t(t).to(f32) @ g).backward()
def n_late():
    uu = u.clone().requires_grad_(True); (late_logit_t(uu, P) @ g).backward()
def n_early():
    early_logit(u); early_vjp(u, g)
n = {'factorized': timed(n_fact), 'late': timed(n_late)}
if 'early' not in SKIP:
    n['early'] = timed(n_early, reps=EARLY_REPS, warm=EARLY_WARM)
res['n_network_step_seconds'] = {k: v[0] for k, v in n.items()}; res['n_all'] = {k: v[1] for k, v in n.items()}
print('(n) logits and gradient [s]:', res['n_network_step_seconds'], flush=True)
save()

# ---- (g) gradient check: each design's gradient code against central finite differences --------
def gradcheck():
    v = torch.tensor(rng.normal(size=NT), dtype=torch.float64); v /= v.norm(); EPS = 1e-2
    gd = g.double()
    th_of = lambda uu: M.CEN + M.HW*uu.double()
    u64 = u.double().cpu()
    def fd(F):
        with torch.no_grad():
            return float((F(u64 + EPS*v) - F(u64 - EPS*v))/(2*EPS))
    chk = {}
    F_fact = lambda uu: fact_logit_t(th_of(uu).to(DEV)).double() @ gd
    F_late = lambda uu: late_logit_t(uu.to(f32).to(DEV), P).double() @ gd
    F_early = lambda uu: early_logit(uu.to(f32).to(DEV)).double() @ gd
    t = th_of(u64).to(DEV).clone().requires_grad_(True)
    (fact_logit_t(t).double() @ gd).backward()
    chk['factorized'] = (float(t.grad.cpu() @ (M.HW*v)), fd(F_fact))
    uu = u.clone().requires_grad_(True)
    (late_logit_t(uu, P).double() @ gd).backward()
    chk['late'] = (float(uu.grad.double().cpu() @ v), fd(F_late))
    if 'early' not in SKIP:
        fd_early = fd(F_early)
        chk['early'] = (float(early_vjp(u, g).double().cpu() @ v), fd_early)
        chk['control_early_one_network_left_out'] = (float(early_vjp(u, g, nets=early[:-1]).double().cpu() @ v), fd_early)
    res['gradcheck'] = {k: dict(gradient=x, finite_difference=y, rel_diff=abs(x - y)/max(abs(y), 1e-12),
                                passes=abs(x - y)/max(abs(y), 1e-12) < 1e-3) for k, (x, y) in chk.items()}
    ctrl = res['gradcheck'].get('control_early_one_network_left_out')
    res['gradcheck_ok'] = (all(r['passes'] for k, r in res['gradcheck'].items() if not k.startswith('control'))
                           and (ctrl is None or not ctrl['passes']))
    print('(g) gradient check:', {k: round(r['rel_diff'], 6) for k, r in res['gradcheck'].items()},
          'OK' if res['gradcheck_ok'] else 'FAILED', flush=True)
    save()
if GRADCHECK:
    try:                                       # a failure here must not cost the fit-step timing below
        gradcheck()
    except Exception as e:
        res['gradcheck_error'] = repr(e); save(); print('(g) gradient check raised', repr(e), flush=True)

# ---- (b) one fit step through directlib, CPU only: chi^2 and gradient with the tilt solved again
if DEV == 'cpu' and 'b' not in SKIP:
    ia = int(np.argmin(np.abs(M.G['alphas'] - 0.120))); jj = int(np.argmin(np.abs(M.G['alpha0'] - 0.41)))
    u0 = rng.uniform(-0.5, 0.5, NT)
    b = {'factorized': timed(lambda: M.chi2_at(u0, ia, jj, None, grad=True))}
    Pc = [p for p in P]
    def with_logit(fn):
        orig = M.logit; M.logit = fn
        try:
            early = fn is EarlyLogits.apply
            return timed(lambda: M.chi2_at(u0, ia, jj, None, grad=True),
                         reps=EARLY_REPS if early else REPS, warm=EARLY_WARM if early else True)
        finally:
            M.logit = orig
    b['late'] = with_logit(lambda theta: late_logit_t(((theta - M.CEN)/M.HW).to(f32), Pc).double())
    if 'early' not in SKIP and 'b_early' not in SKIP:
        b['early'] = with_logit(EarlyLogits.apply)
    res['b_fit_step_seconds'] = {k: v[0] for k, v in b.items()}; res['b_all'] = {k: v[1] for k, v in b.items()}
    # the design-independent part: the tilt and chi^2 given the logits
    lg0 = fact_logit_t(torch.tensor(M.CEN.numpy() + M.HW.numpy()*u0)).detach().double()
    def rest():
        L = lg0.clone(); orig = M.logit; M.logit = lambda theta: L + 0*theta.sum()   # no network, gradient defined
        try:
            M.chi2_at(u0, ia, jj, None, grad=True)
        finally:
            M.logit = orig
    res['b_tilt_and_chi2_seconds'] = timed(rest)[0]
    print('(b) fit step [s]:', res['b_fit_step_seconds'], ' tilt and chi2 alone:', res['b_tilt_and_chi2_seconds'], flush=True)

save(); print('wrote', OUT)
