#!/usr/bin/env python3
"""The prediction of Sec. 5 against DELPHI's thrust distribution from charged plus neutral particles
(d11-x01-y02, read by lepcov/lepdata_cov.py), in the twelve DELPHI bins 0.05 <= tau < 0.30 that lie wholly inside
the window: the NNLL+NLO calculation at the paper's fixed point (output/calc_point.json), ALEPH merged into DELPHI's
bins, and the prediction with the calculation imposed (three_bands npz), each against DELPHI.

As published (D3): every distribution is the fraction of all events in each bin over its width and the data are
those published, so nothing is renormalized: calc_over_delphi, modR_over_delphi, aleph_over_delphi,
imposed_over_delphi and imposed_over_calc. The imposed prediction is also compared at DELPHI's hadron level (D8,
conventions.py): each generator's imposed prediction multiplied by its own factors c/a, then mixed with the
(1 - xi, xi) of the central point (imposed_conv_over_delphi); imposed_over_delphi keeps the uncorrected value.
Apart, under keys ending in _win: the same comparisons with every distribution normalized over the twelve bins,
with the generator band of the imposed prediction from the draws of three_bands.py (its points normalized the same
way, gen_band_win_rel, relative to the prediction); and the shape of the calculation alone against DELPHI
(calc_shape_over_delphi_shape, modR_shape_over_delphi_shape, the same twelve-bin normalization). App. C quotes
calc_over_delphi, as published (make_numbers.py, TbCalcBelowTop, E12).
A three_bands npz of the earlier draft (geometric, normalized over the bins from 0.05) is taken to the total rate
with its bins below 0.05, which hold the rest of the events.

The calculation is binned with the functions of fig_thrust_vs_aleph.py (the head of that script, before its
plotting, is executed here so that the two use the same code). Writes <OUTDIR>/calc_vs_delphi_<tag>.json, OUTDIR
by default output/ next to this script; NPZ, CALC_POINT and OUTDIR given as relative paths are taken from the folder
the script is started in.

    NPZ=output/three_bands_MIXGEOV3S9_lin.npz python calc_vs_delphi.py MIXGEOV3S9_lin
"""
import os, sys, json, numpy as np
_abs = lambda p: None if p is None else os.path.abspath(p)        # paths given by the caller, before the chdir
NPZ, CALC_POINT, OUTDIR = (_abs(os.environ.get(k)) for k in ('NPZ', 'CALC_POINT', 'OUTDIR'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1] if len(sys.argv) > 1 else 'MIXGEOV3S9'
OUTDIR = OUTDIR or 'output'
CONV = os.environ.get('CONVENTIONS', 'factors'); assert CONV in ('factors', 'none'), CONV
os.environ['THRUST_ORDER'] = '2'; os.environ['CHAIN_SUFFIX'] = '_nlo'; os.environ.pop('FIXED_POINT', None)
head = open('fig_thrust_vs_aleph.py').read().split('lo, hi, y, e = aleph()')[0]
assert 'def binned' in head and 'def build' in head
exec(compile(head.replace('setup(usetex=True)', ''), 'fig_thrust_vs_aleph.py', 'exec'))
sys.path.insert(0, 'lepcov')
from lepdata_cov import lepdata_split
CP = json.load(open(CALC_POINT or 'output/calc_point.json')); ASC, A0C = float(CP['alpha_s']), float(CP['alpha_0'])
N.SUB_ORDER = int(CP.get('np_sub_order', 2))       # the subtraction order of the fit that sets alpha_0
la, ha, ya, sta, sya = lepdata_split('aleph'); ld, hd, yd, std, syd = lepdata_split('delphi')
npz = NPZ or f'output/three_bands_{tag}.npz'
Z = np.load(npz); fj = os.path.splitext(npz)[0] + '.json'
J = json.load(open(fj)) if os.path.exists(fj) else {}
MIX = str(Z['mix']) if 'mix' in Z.files else J.get('mix', 'geometric')
if tag.endswith('_lin'):
    assert MIX == 'linear', f'{npz}: a _lin tag needs the linear mixture, the file holds mix={MIX}'
G = np.load('output/thrust_targets_nlo_point.npz', allow_pickle=True)
assert abs(float(G['alphas'][0]) - ASC) < 1e-12 and abs(float(G['alpha0'][0]) - A0C) < 1e-12, 'targets of another point'
k12 = (ld >= 0.05 - 1e-9) & (hd <= 0.30 + 1e-9); lo, hi = ld[k12], hd[k12]; wid = hi - lo
D, De = yd[k12], np.hypot(std[k12], syd[k12])
AL, ALe = [], []
for l, h in zip(lo, hi):                       # ALEPH's bins merged into DELPHI's, systematics linear (conservative)
    k = (la >= l - 1e-9) & (ha <= h + 1e-9); w = ha[k] - la[k]; assert np.isclose(w.sum(), h - l)
    AL.append((ya[k]*w).sum()/(h - l)); ALe.append(np.hypot(np.sqrt(((sta[k]*w)**2).sum()), (sya[k]*w).sum())/(h - l))
AL, ALe = np.array(AL), np.array(ALe)   # not A, which the executed head uses for the grid
kb = (Z['lo'] >= 0.05 - 1e-9) & (Z['hi'] <= 0.30 + 1e-9)
assert np.allclose(Z['lo'][kb], lo) and np.allclose(Z['hi'][kb], hi)
n = lambda v: v/(v*wid).sum()                  # normalized over the twelve bins
if MIX == 'linear':
    # the npz holds fractions of all events and the data as published
    assert np.allclose(Z['data'][kb], D, rtol=1e-6), 'DELPHI values differ (the npz is not of d11-x01-y02 as published)'
    total = 1.0
else:
    # the earlier draft normalized over the bins from 0.05; with its bins below, which hold the other events, the
    # prediction goes back to the fraction of all events
    assert np.allclose(n(Z['data'][kb]), n(D), rtol=1e-6), 'DELPHI values differ'
    wz, wl = Z['hi'] - Z['lo'], Z['hi_low'] - Z['lo_low']
    total = float((Z['h_central']*wz).sum() + (Z['h_central_low']*wl).sum())
calc = binned(1.0, 1.0, 'logR', ASC, A0C, lo, hi); modr = binned(1.0, 1.0, 'modR', ASC, A0C, lo, hi)
imp = Z['h_central'][kb]/total
out = dict(tag=tag, mix=MIX, npz=npz, alpha_s=ASC, alpha_0=A0C, lo=lo.tolist(), hi=hi.tolist(),
           normalization='as published: every distribution the fraction of all events in each bin over its width',
           calc_over_delphi=(calc/D).tolist(), modR_over_delphi=(modr/D).tolist(),
           aleph_over_delphi=(AL/D).tolist(), aleph_rel_err=(ALe/AL).tolist(), delphi_rel_err=(De/D).tolist(),
           imposed_over_calc=(imp/calc).tolist(), imposed_over_delphi=(imp/D).tolist(),
           # apart: the same, every distribution normalized over the twelve bins
           window_bins='the twelve DELPHI bins wholly inside the window, 0.05 <= tau < 0.30',
           calc_shape_over_delphi_shape=(n(calc)/n(D)).tolist(), modR_shape_over_delphi_shape=(n(modr)/n(D)).tolist(),
           aleph_win_over_delphi_win=(n(AL)/n(D)).tolist(), imposed_win_over_delphi_win=(n(imp)/n(D)).tolist(),
           imposed_win_over_calc_win=(n(imp)/n(calc)).tolist())
if 'H_after' in Z.files and MIX == 'linear':
    # the generator band of the window-normalized prediction, from the draws of three_bands.py (its points)
    Hw = Z['H_after'][:, kb]; Hw = Hw/(Hw*wid).sum(1, keepdims=True)
    out['gen_band_win_rel'] = (Hw.std(0, ddof=1)/n(imp)).tolist(); out['gen_band_win_points'] = int(len(Hw))
if MIX == 'linear' and CONV == 'factors':
    # D8: each generator's imposed prediction times its own c/a, then mixed at the central point
    import conventions
    FAC = conventions.load(); xi = float(J['central']['theta'][-1]) if J else 0.5
    hg = {g: Z[f'h_central_{conventions.GEN_OF[g]}'][kb] for g in ('S', 'H')}
    fg = {g: FAC.hist(g, 'delphi_y02', conventions.DELPHI_KIND, lo, hi) for g in ('S', 'H')}
    impc = (1 - xi)*fg['S']*hg['S'] + xi*fg['H']*hg['H']
    if 'h_central_conv' in Z.files:
        assert np.allclose(Z['h_central_conv'][kb], impc, rtol=1e-12, atol=0), 'the corrected prediction of the npz differs'
    out.update(conventions=dict(file=FAC.path, meta=FAC.meta, delphi=f'delphi_y02 {conventions.DELPHI_KIND}', xi=xi),
               factors_sherpa=fg['S'].tolist(), factors_herwig=fg['H'].tolist(),
               imposed_conv_over_delphi=(impc/D).tolist(), imposed_conv_win_over_delphi_win=(n(impc)/n(D)).tolist())
    if 'H_after_conv' in Z.files:
        Hc = Z['H_after_conv'][:, kb]; Hc = Hc/(Hc*wid).sum(1, keepdims=True)
        out['gen_band_conv_win_rel'] = (Hc.std(0, ddof=1)/n(impc)).tolist()
os.makedirs(OUTDIR, exist_ok=True)
fo = os.path.join(OUTDIR, f'calc_vs_delphi_{tag}.json')
json.dump(out, open(fo, 'w'), indent=1)
print('as published (fractions of all events):')
print(f"{'bin':10s} {'calc/D':>7s} {'modR/D':>7s} {'ALEPH/D':>7s} {'imp/D':>7s} {'impc/D':>7s} {'imp/calc':>8s} {'D err%':>6s} {'A err%':>6s}")
for i in range(len(lo)):
    ic = out['imposed_conv_over_delphi'][i] if 'imposed_conv_over_delphi' in out else float('nan')
    print(f"{lo[i]:.2f}-{hi[i]:.2f} {out['calc_over_delphi'][i]:7.3f} {out['modR_over_delphi'][i]:7.3f} {out['aleph_over_delphi'][i]:7.3f} "
          f"{out['imposed_over_delphi'][i]:7.3f} {ic:7.3f} {out['imposed_over_calc'][i]:8.3f} {100*De[i]/D[i]:6.2f} {100*ALe[i]/AL[i]:6.2f}")
print('normalized over the twelve bins: calc/D ' + ' '.join(f'{v:.3f}' for v in out['calc_shape_over_delphi_shape'])
      + '\n                                 imp/calc ' + ' '.join(f'{v:.3f}' for v in out['imposed_win_over_calc_win']))
print('wrote', fo)
