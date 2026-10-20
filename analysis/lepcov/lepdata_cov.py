#!/usr/bin/env python3
"""Statistical and systematic errors of the three LEP 91 GeV thrust tables, separately, and the
bin-by-bin data covariance under a chosen correlation model.

The Rivet reference tables read by fit_alpha0.lepdata carry only the total error, stat and sys in
quadrature. HEPData publishes the two components separately (label 'stat' and 'sys') for exactly
the tables Rivet uses:
    ALEPH  ins636645 Table 54 (d54-x01-y01, in T)
    DELPHI ins424112 Table 11 (d11-x01-y01, in 1-T)
    OPAL   ins669402 Table 1  (d01-x01-y01, first dependent variable = 91 GeV, in 1-T)
The tables are cached as YAML under HEPDATA_DIR (default: ./hepdata next to this file); they are
downloaded from https://www.hepdata.net/download/table/<ins>/Table%20<n>/yaml.

lepdata_split(name) returns (lo, hi, y, stat, sys) in the order of fit_alpha0.lepdata, after
checking bin edges and values against the Rivet table and that sqrt(stat^2 + sys^2) reproduces
the Rivet error.

data_cov(name, sel=None) returns the covariance of the selected bins under DATA_COV:
    diag      diag(stat^2 + sys^2), what the fit uses now (default)
    statonly  diag(stat^2), for reference only
    corr      diag(stat^2) + s s^T: the systematic fully correlated between all bins of one
              experiment (same sign, same fraction of its error in every bin)
    minov     diag(stat^2) + M, M_ij = min(sys_i, sys_j)^2: the minimum-overlap model
    rhoX      diag(stat^2) + diag(sys^2)(1-X) + X s s^T  (e.g. rho0.5): partial correlation
Experiments are never correlated with each other.
"""
import os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
HEPDATA_DIR = os.environ.get('HEPDATA_DIR', os.path.join(HERE, 'hepdata'))
sys.path.insert(0, os.environ.get('MAXENT_CODE', '/Users/user/maxent_code'))
from fit_alpha0 import lepdata

SRC = {'aleph': ('ins636645', 54, True), 'delphi': ('ins424112', 11, False), 'opal': ('ins669402', 1, False)}


def _read(name):
    import yaml
    ins, tab, in_T = SRC[name]
    f = os.path.join(HEPDATA_DIR, f'{ins}_{tab}.yaml')
    if not os.path.exists(f):
        import urllib.request
        os.makedirs(HEPDATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(f'https://www.hepdata.net/download/table/{ins}/Table%20{tab}/yaml', f)
    d = yaml.safe_load(open(f))
    dv = d['dependent_variables'][0]
    q = {x['name']: str(x['value']) for x in dv.get('qualifiers', [])}
    assert q.get('SQRT(S)', '').startswith('91'), (name, q)
    iv = d['independent_variables'][0]['values']
    lo = np.array([float(v['low']) for v in iv]); hi = np.array([float(v['high']) for v in iv])
    y = np.array([float(v['value']) for v in dv['values']])
    st = np.zeros(len(y)); sy = np.zeros(len(y))
    for i, v in enumerate(dv['values']):
        for e in v.get('errors', []):
            if 'symerror' in e:
                s = abs(float(str(e['symerror']).rstrip('%')))
                if str(e['symerror']).endswith('%'): s *= y[i]/100
            else:
                s = 0.5*(abs(float(e['asymerror']['plus'])) + abs(float(e['asymerror']['minus'])))
            if e['label'].lower().startswith('stat'): st[i] = np.hypot(st[i], s)
            else: sy[i] = np.hypot(sy[i], s)
    if in_T:
        lo, hi = 1 - hi, 1 - lo
    o = np.argsort(lo)
    return lo[o], hi[o], y[o], st[o], sy[o]


def lepdata_split(name):
    lo, hi, y, e = lepdata(name)
    L, H, Y, st, sy = _read(name)
    assert len(L) == len(lo), (name, len(L), len(lo))
    assert np.allclose(L, lo, atol=1e-9) and np.allclose(H, hi, atol=1e-9), name
    assert np.allclose(Y, y, rtol=1e-6, atol=1e-12), name
    tot = np.hypot(st, sy)
    ok = np.isclose(tot, e, rtol=2e-3, atol=1e-12)
    assert ok.all(), (name, np.where(~ok)[0], tot[~ok], e[~ok])
    return lo, hi, y, st, sy


def cov_model(st, sy, model=None):
    model = model or os.environ.get('DATA_COV', 'diag')
    if model == 'diag':
        return np.diag(st**2 + sy**2)
    if model == 'statonly':
        return np.diag(st**2)
    if model == 'corr':
        return np.diag(st**2) + np.outer(sy, sy)
    if model == 'minov':
        return np.diag(st**2) + np.minimum.outer(sy, sy)**2
    if model.startswith('rho'):
        r = float(model[3:])
        return np.diag(st**2 + (1 - r)*sy**2) + r*np.outer(sy, sy)
    raise ValueError(model)


def data_cov(name, sel=None, model=None):
    lo, hi, y, st, sy = lepdata_split(name)
    if sel is not None:
        st, sy = st[sel], sy[sel]
    return cov_model(st, sy, model)


def renorm_jacobian(y, wid):
    """d(y_i/S)/d y_j for S = sum_k y_k wid_k: the exact linear propagation of renormalizing the data
    over the selected bins (the fit's current convention ignores it and divides errors by S)."""
    S = float((y*wid).sum())
    return np.eye(len(y))/S - np.outer(y, wid)/S**2


if __name__ == '__main__':
    for ex in SRC:
        lo, hi, y, st, sy = lepdata_split(ex)
        for (a, b, lab) in ((0.05, 1.0, 'tau >= 0.05 (fit)'), (0.05, 0.10, 'window lower'), (0.10, 1/3, 'window upper')):
            s = (lo >= a - 1e-9) & (hi <= b + 1e-9) if b < 1 else (lo >= a - 1e-9)
            fr = sy[s]**2/(st[s]**2 + sy[s]**2)
            print(f'{ex:7s} {lab:18s} {int(s.sum()):3d} bins  sys^2/tot^2 median {np.median(fr):.2f} '
                  f'[{fr.min():.2f},{fr.max():.2f}]  sys/y median {np.median(sy[s]/y[s]):.3f}  stat/y median {np.median(st[s]/y[s]):.4f}')
