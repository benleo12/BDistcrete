#!/usr/bin/env python3
"""The signed displacement of every theory variation of Sec. 5, bin by bin and for the means, at the
central point of three_bands.py (the default tunes, xi = 1/2), with the calculation imposed: the four
scale variations, the change of scheme and the three variations of the dispersive model. three_bands.py
combines them into the signed theory band and keeps only the combination. Also the generator-only and
imposed predictions at xi = 0 and xi = 1 (each generator alone at its default tune). Same Model, grid
and test data as three_bands.py. The predictions are also given in the bins below the fitted ones,
with the same normalization, under keys ending in _low. Writes output/theory_variations_<tag>.json.

MIX=linear (the default): the linear mixture of three_bands.py, each generator at xi = 0 and 1 with its own
tilt, the predictions mixed with (1 - xi, xi), the distribution normalized to the total rate and the data as
published (to the total rate); every variation also gives each generator's own displacement (dh_sherpa,
do_sherpa, dh_herwig, do_herwig), and neff is that of the mixture's weights (1 - xi) w_S + xi w_H. Every prediction
is also given at the hadron level of DELPHI's thrust distribution and OPAL's <1-T> (D8, conventions.py: each
generator multiplied by its own factors after imposing and before mixing), under keys ending in _conv;
CONVENTIONS=none leaves them out. OUT (without .json) defaults to output/theory_variations_<tag>, with _lin when
linear.

    python theory_variations.py MIX output/models/MIXGEOV3S9_cond.npz output/models/MIXGEOV3S9_ref_v2_slim.npz \
        output/thrust_targets_nlo_point.npz
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from directlib import Model, at_fraction

tag, export, ref, grid = sys.argv[1:5]
MIX = os.environ.get('MIX', 'linear'); assert MIX in ('linear', 'geometric'), MIX
LINEAR = MIX == 'linear'
TEST = os.environ.get('TEST_DATA', 'delphi'); OUT = os.environ.get('OUT', f'output/theory_variations_{tag}' + ('_lin' if LINEAR else ''))
CONV = os.environ.get('CONVENTIONS', 'factors'); assert CONV in ('factors', 'none'), CONV
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
M = Model(tag, export=export, ref=ref, exps=(TEST,), first_bin=0.05, floor_rel=0.002, a0_stride=1, ae_dtype='float32', grid=grid)
G = M.G; NT = M.NT; cen = M.CEN.numpy(); hw = M.HW.numpy()
OBK = ['tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon']
x = M.exp[0]; use = x['use'].numpy(); wid = x['wid'].numpy()
low = np.where(x['lo'] < x['lo'][use].min() - 1e-12)[0]; ext = np.r_[low, use]; nlow = len(low)
GENS = (('S', 0.0, 'sherpa'), ('H', 1.0, 'herwig'))


def predict1(theta, which='central', anchored=True):
    with torch.no_grad():
        lw, lw0, lam, Pw, its = M.weights(torch.as_tensor(theta, dtype=torch.float64), 0, 0, which=which, anchored=anchored)
        w = torch.exp(lw); o = M.observables(w); o = {k: float(o[k]) for k in OBK if k in o}
        cnt = torch.zeros(x['nb']).index_add(0, x['idx'], w[x['ins']]).numpy()
        # the bins below the fitted ones first, then the fitted bins; linear: normalized to the total rate
        h = cnt[ext]/wid[ext] if LINEAR else cnt[ext]/(wid[ext]*cnt[use].sum())
        neff = float(1.0/(w**2).sum())
    return o, h, neff, w


def predict(theta, which='central', anchored=True, parts=None):
    """(o, h, neff); linear: (1 - xi) and xi times each generator's (o, h) at xi = 0 and 1, neff of the mixed
    weights, and parts (a dict) receives each generator's (o, h)."""
    if not LINEAR:
        return predict1(theta, which, anchored)[:3]
    xi = float(theta[-1]); o, h, w = None, 0.0, 0.0
    for g, xg, _ in GENS:
        a = 1 - xi if g == 'S' else xi
        if a == 0:
            continue
        og, hg, _, wg = predict1(at_fraction(theta, xg), which, anchored)
        if parts is not None:
            parts[g] = (og, hg)
        o = {k: a*v for k, v in og.items()} if o is None else {k: o[k] + a*og[k] for k in og}
        h = h + a*hg; w = w + a*wg
    return o, h, float(1.0/(w**2).sum())


central = np.array([float(v) for v in os.environ['CENTRAL'].split(',')], np.float64)
P1, P0 = {}, {}
o_c, h_c, ne_c = predict(central, parts=P1); o_c0, h_c0, ne_c0 = predict(central, anchored=False, parts=P0)
VC, CONV_REC = None, None
if LINEAR and CONV == 'factors':
    import conventions                                              # D8: the factors of each generator, (means, bins)
    assert TEST == 'delphi', 'the factors of D8 are given for DELPHI (TEST_DATA=delphi); set CONVENTIONS=none otherwise'
    VC, CONV_REC = conventions.load().vectors(OBK, x['lo'][ext], x['hi'][ext])


def conv(parts, xi):
    """(means, histogram) of sum_g a_g f_g y_g, each generator's prediction times its own factors (D8)."""
    a = dict(S=1 - xi, H=xi)
    y = sum(a[g]*VC[g]*np.r_[[o[k] for k in OBK], h] for g, (o, h) in parts.items() if a[g] != 0)
    return {k: float(y[j]) for j, k in enumerate(OBK)}, y[len(OBK):]

S = lambda a: np.asarray(a)[nlow:].tolist()     # the fitted bins, exactly as before
L = lambda a: np.asarray(a)[:nlow].tolist()     # the bins below them
dy, de = (x['y_raw'], x['e_raw']) if LINEAR else (x['y'].numpy(), x['e'].numpy())   # the data as the predictions
out = dict(tag=tag, test=TEST, central=central.tolist(), lo=x['lo'][use].tolist(), hi=x['hi'][use].tolist(),
           data=(dy[use]).tolist(), err=(de[use]).tolist(),
           h_imposed=S(h_c), h_gen_only=S(h_c0), o_imposed=o_c, o_gen_only=o_c0,
           neff_imposed=ne_c, neff_gen_only=ne_c0, n_ref=int(M.N), variations={},
           lo_low=x['lo'][low].tolist(), hi_low=x['hi'][low].tolist(), data_low=(dy[low]).tolist(),
           err_low=(de[low]).tolist(), h_imposed_low=L(h_c), h_gen_only_low=L(h_c0))
if LINEAR:
    out.update(mix=MIX, normalization='total rate')
if VC is not None:
    (oc1, hc1), (oc0, hc0) = conv(P1, central[-1]), conv(P0, central[-1])
    out.update(conventions=CONV_REC, o_imposed_conv=oc1, o_gen_only_conv=oc0, h_imposed_conv=S(hc1), h_gen_only_conv=S(hc0),
               h_imposed_conv_low=L(hc1), h_gen_only_conv_low=L(hc0))
for kind, labels in (('scale', [str(l) for l in G['scale_labels']]), ('np', [str(l) for l in G['np_labels']])):
    for v, lab in enumerate(labels):
        pp = {}; ov, hv, _ = predict(central, which=f'{kind}:{v}', parts=pp)
        out['variations'][f'{kind}:{lab}'] = dict(dh=S(hv - h_c), do={k: ov[k] - o_c[k] for k in o_c})
        for g, _, nm in GENS:
            if g in pp:
                out['variations'][f'{kind}:{lab}'].update({f'dh_{nm}': S(pp[g][1] - P1[g][1]), f'dh_{nm}_low': L(pp[g][1] - P1[g][1]),
                                                           f'do_{nm}': {k: pp[g][0][k] - P1[g][0][k] for k in o_c}})
for xi in (0.0, 1.0):
    th = central.copy(); th[-1] = xi
    q1, q0 = {}, {}
    o1, h1, n1 = predict(th, parts=q1); o0, h0, n0 = predict(th, anchored=False, parts=q0)
    out[f'xi{int(xi)}'] = dict(h_imposed=S(h1), h_gen_only=S(h0), o_imposed=o1, o_gen_only=o0,
                               neff_imposed=n1, neff_gen_only=n0, h_imposed_low=L(h1), h_gen_only_low=L(h0))
    if VC is not None:
        (oc1, hc1), (oc0, hc0) = conv(q1, xi), conv(q0, xi)
        out[f'xi{int(xi)}'].update(o_imposed_conv=oc1, o_gen_only_conv=oc0, h_imposed_conv=S(hc1), h_gen_only_conv=S(hc0),
                                   h_imposed_conv_low=L(hc1), h_gen_only_conv_low=L(hc0))
json.dump(out, open(OUT + '.json', 'w'), indent=1)
print('wrote', OUT + '.json', 'neff (imposed, gen only) at xi=1/2:', ne_c, ne_c0)
