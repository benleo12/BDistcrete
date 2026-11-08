#!/usr/bin/env python3
"""The NNLL+NLO thrust calculation at the paper's fixed point, binned in the ALEPH and DELPHI bins, with
every variation of Sec. 5 (four scale variations and the change of scheme at the same point, and the
three variations of the dispersive model), for the appendix figure of the calculation against the data.
Densities are normalized to the total rate, as the data. The same cumulants and
shift as targets_nlo_point.py, which this script checks by recomputing the fourteen windowed moments of
output/thrust_targets_nlo_point.npz. The data are those of lepcov/lepdata_cov.py (DELPHI: d11-x01-y02, charged
plus neutral particles), as published. Writes <OUTDIR>/calc_bins_nlo_point.json, OUTDIR by default output/ next to
this script; OUTDIR and CALC_POINT given as relative paths are taken from the folder the script is started in. The json
also holds power_correction: the rigid shift a_tau P of the distribution at the fixed point and central scales
(np_shift.shift with the subtraction order of calc_point.json), which App. C quotes (a_tau P = 0.0093) and
make_numbers.py writes as \\CalcShift.

    THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python calc_bins.py
"""
import os, sys, json
os.environ.setdefault('THRUST_ORDER', '2'); os.environ.setdefault('CHAIN_SUFFIX', '_nlo')
assert os.environ['THRUST_ORDER'] == '2'
import numpy as np
OUTDIR, CALC_POINT = (os.path.abspath(os.environ[k]) if os.environ.get(k) else None for k in ('OUTDIR', 'CALC_POINT'))
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, '.')
import np_shift as N, thrust_chain as TC
sys.path.insert(0, 'lepcov')
from lepdata_cov import lepdata_split

WLO, WHI = 0.05, 1.0/3.0
CP = json.load(open(CALC_POINT or 'output/calc_point.json')); asc, a0c = float(CP['alpha_s']), float(CP['alpha_0'])
N.SUB_ORDER = int(CP.get('np_sub_order', 2)); ALT = 2 if N.SUB_ORDER == 3 else 3
G = np.load('output/thrust_targets_nlo_point.npz', allow_pickle=True)
assert abs(float(G['alphas'][0]) - asc) < 1e-12 and abs(float(G['alpha0'][0]) - a0c) < 1e-12
_cache = {}


def cumulant(mu, xv, sch, asmz):
    key = (mu, xv, sch, asmz)
    if key not in _cache:
        tau, m = TC.build(mu, xv, sch, asmz); S, _ = N.sanitize(m/m[-1]); _cache[key] = (tau, S)
    return _cache[key]


def evaluate(fn, mu, xv, sch, milan=None, sub=None):
    """fn(t, S) at the fixed point, interpolated between the coupling nodes as in targets_nlo_point.py."""
    ia = int(np.clip(np.searchsorted(TC.ASGRID, asc) - 1, 0, len(TC.ASGRID) - 2))
    a1, a2 = TC.ASGRID[ia], TC.ASGRID[ia + 1]; wgt = (asc - a1)/(a2 - a1)
    old_m, old_s = N.MILAN, N.SUB_ORDER
    if milan is not None: N.MILAN = milan
    if sub is not None: N.SUB_ORDER = sub
    try:
        out = 0
        for aa, ww in ((a1, 1 - wgt), (a2, wgt)):
            tau, S = cumulant(mu, xv, sch, aa); t = tau + N.shift(mu, a0c, asmz=aa)
            out = out + ww*np.asarray(fn(t, S))
    finally:
        N.MILAN, N.SUB_ORDER = old_m, old_s
    return out


# check: the fourteen windowed moments of the stored targets
keys = [str(k) for k in G['keys']]
AB = {'logt': (0, 1), 'tau': (1, 0), 'tlogt': (1, 1), 'log2t': (0, 2), 'tau2': (2, 0), 'tlog2t': (1, 2), 't2logt': (2, 1),
      'log3t': (0, 3), 'tau3': (3, 0), 'tlog3t': (1, 3), 't3logt': (3, 1), 't2log2t': (2, 2), 'log4t': (0, 4), 'tau4': (4, 0)}
TF = np.linspace(WLO, WHI, 200001); TM = 0.5*(TF[1:] + TF[:-1])
FW = np.array([TM**AB[k][0]*np.log(TM)**AB[k][1] for k in keys])
mom = evaluate(lambda t, S: (lambda d: FW @ d/d.sum())(np.diff(N.cum_eval(TF, t, S))), 1.0, 1.0, 'logR')
assert np.allclose(mom, G['central'][0, 0], rtol=1e-6, atol=1e-12), np.max(np.abs(mom/G['central'][0, 0] - 1))

VARS = [((1.0, 1.0, 'logR'), {}, 'central'),
        ((2.0, 1.0, 'logR'), {}, "(2.0, 1.0, 'logR', 1.0)"), ((0.5, 1.0, 'logR'), {}, "(0.5, 1.0, 'logR', 1.0)"),
        ((1.0, 2.0, 'logR'), {}, "(1.0, 2.0, 'logR', 1.0)"), ((1.0, 0.5, 'logR'), {}, "(1.0, 0.5, 'logR', 1.0)"),
        ((1.0, 1.0, 'modR'), {}, "(1.0, 1.0, 'modR', 1.0)"),
        ((1.0, 1.0, 'logR'), dict(milan=1.2*N.MILAN), 'milan_up'), ((1.0, 1.0, 'logR'), dict(milan=0.8*N.MILAN), 'milan_dn'),
        ((1.0, 1.0, 'logR'), dict(sub=ALT), f'sub{ALT}')]
out = dict(alpha_s=asc, alpha_0=a0c, window=[WLO, WHI], moments_check='passed')
# App. C: the shift of the distribution by the power correction at the fixed point and central scales, a_tau P
SH = float(N.shift(1.0, a0c, asmz=asc))
out['power_correction'] = dict(shift=SH, a_tau=float(N.A_TAU), P=SH/float(N.A_TAU), milan=float(N.MILAN), mu_I=float(N.MUI),
                               np_sub_order=int(N.SUB_ORDER), mur_mult=1.0)
for ex in ('aleph', 'delphi'):
    lo, hi, y, st, sy = lepdata_split(ex)
    d = dict(lo=lo.tolist(), hi=hi.tolist(), y=y.tolist(), stat=st.tolist(), sys=sy.tolist(), calc={})
    for (mu, xv, sch), kw, lab in VARS:
        d['calc'][lab] = evaluate(lambda t, S: (N.cum_eval(hi, t, S) - N.cum_eval(lo, t, S))/(hi - lo), mu, xv, sch, **kw).tolist()
    out[ex] = d
OUTDIR = OUTDIR or 'output'; os.makedirs(OUTDIR, exist_ok=True); FOUT = os.path.join(OUTDIR, 'calc_bins_nlo_point.json')
json.dump(out, open(FOUT, 'w'), indent=1)
c = np.array(out['aleph']['calc']['central']); y = np.array(out['aleph']['y']); e = np.hypot(out['aleph']['stat'], out['aleph']['sys'])
lo = np.array(out['aleph']['lo']); hi = np.array(out['aleph']['hi']); w = (lo >= WLO - 1e-9) & (hi <= WHI + 1e-9)
print(f'moments check passed; ALEPH window chi2 (diagonal errors) {np.sum(((c - y)/e)[w]**2):.1f} for {w.sum()} bins')
print(f'power correction at the fixed point: a_tau P = {SH:.6f} (subtraction order {N.SUB_ORDER})')
print('wrote', FOUT)
