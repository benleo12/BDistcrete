#!/usr/bin/env python3
"""Does the fixed DELPHI convention factor hold for reweighted events? (review 2026-11-10, request P-F11-2)

The paper converts each generator's prediction from our hadron level (convention a) to DELPHI's (convention c,
hadrons with lifetimes below 1 ns decayed) with one factor per DELPHI bin, c_over_a of conventions/factors.json,
measured at the default tunes without the calculation, and applies it after the maximum-entropy weights. This
script tests that approximation on convention-c samples, whose per-event tables also hold the same events with
the decays undone (the 'col' columns, convention a exactly), so the exact conversion of any weighted sample is
known event by event:

  exact      fraction of all (weighted) events per DELPHI bin in convention c
  fixed      the same fraction in convention a, times the default-tune factor of factors.json

for each sample (the default-tune samples of lmx_phaseA/conventions and the samples of review_1110/conv at the
center and one edge of the Stage D/E box), without and with the calculation imposed. The calculation is imposed
with the release's gentune.maxent (the paper's targets, sigma and 0.2% floor) on the convention-a thrust of
each sample, i.e. by the maximum-entropy weight that the paper's procedure gives that generator at that point;
the scale and scheme variations of the targets can be imposed too. Residual = fixed/exact - 1, per bin and for
the fraction of all events in 0.05 <= tau < 0.30 (the twelve DELPHI bins wholly inside the window). Errors:
delete-a-group jackknife over chunks (the multipliers solved again in every replicate); at the other points
also the statistical error of the default factor, since that comes from independent events.

  python3 -B conv_weighted_check.py --sample sherpa:default:DIR ... --factors conventions/factors.json \
      --rivet rivet_ref --targets ../deposit/targets_thrust_nlo.npz --out OUT.json [--variations]
"""
import argparse, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('make_factors', os.path.join(HERE, 'conventions', 'make_factors.py'))
MF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(MF)
sys.path.insert(0, os.path.join(HERE, 'release'))
from gentune import maxent                                         # noqa: E402

NGROUP = 20
WIN = (0.05, 0.30)                                                 # the twelve DELPHI bins wholly inside the window


def load(d):
    S, chunks = MF.load(d)
    cid = np.repeat(np.arange(len(chunks)), [c['n'] for c in chunks])
    assert len(cid) == len(S['tau'])
    return S['tau'], S['tau_col'], cid, len(chunks)


def imposed_weights(tau_a, keep, T, which, lam0=None):
    """Normalized maximum-entropy weights of the events in keep (uniform before), with the calculation's c
    (or a variation's) imposed on the convention-a thrust."""
    t = tau_a[keep]
    inwin, F = T.functions(t)
    lw, lam, info = T.impose(np.zeros(len(t)), inwin, F, which=which, lam0=lam0)
    return np.exp(lw), lam, info


def fractions(tau, u, edges):
    i = MF.bin_index(tau, edges); K = len(edges) - 1; ok = i >= 0
    return np.bincount(i[ok], weights=u[ok], minlength=K)/u.sum()


def stats(tau_c, tau_a, u, edges, F0, sel):
    """res_* against the given factor F0; chg_* against the factor of the same events without weights, the change
    that the weights alone make (jackknifed together, so it has the small error of a paired comparison)."""
    fc, fa = fractions(tau_c, u, edges), fractions(tau_a, u, edges)
    one = np.ones(len(tau_c))
    fc0, fa0 = fractions(tau_c, one, edges), fractions(tau_a, one, edges)
    with np.errstate(divide='ignore', invalid='ignore'):
        res_bin = F0*fa/fc - 1
        r = fc/fa
        G = fc0/fa0
        chg_bin = G*fa/fc - 1
    Pc, Pfix = fc[sel].sum(), (F0*fa)[sel].sum()
    return dict(res_bin=res_bin, factor=r, res_P=Pfix/Pc - 1, P_exact=Pc, P_fixed=Pfix, P_a=fa[sel].sum(),
                chg_bin=chg_bin, chg_P=(G*fa)[sel].sum()/Pc - 1)


def run(tau_c, tau_a, cid, nch, edges, F0, sel, T, which):
    """Point estimates and jackknife errors, without (which=None) or with the calculation imposed."""
    grp = cid % NGROUP

    def one(keep, lam0=None):
        if which is None:
            u = np.ones(keep.sum()); lam = None; info = {}
        else:
            u, lam, info = imposed_weights(tau_a, keep, T, which, lam0)
        return stats(tau_c[keep], tau_a[keep], u, edges, F0, sel), lam, info

    full, lam, info = one(np.ones(len(tau_c), bool))
    reps = [one(grp != g, lam)[0] for g in range(NGROUP)]
    out = {}
    for k in ('res_bin', 'factor', 'res_P', 'P_exact', 'P_fixed', 'P_a', 'chg_bin', 'chg_P'):
        R = np.array([np.atleast_1d(r[k]) for r in reps])
        e = np.sqrt((NGROUP - 1)/NGROUP*((R - R.mean(0))**2).sum(0))
        out[k] = (np.atleast_1d(full[k]), e)
    if which is not None:
        out['info'] = dict(iterations=info['iterations'], converged=info['converged'], residual=info['residual'],
                           pwin_before=info['pwin_before'], pwin_after=info['pwin_after'], n_eff=info['n_eff'],
                           max_abs_pull=float(np.max(np.abs(info['pulls']))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', action='append', required=True, help='gen:point:dir')
    ap.add_argument('--factors', default=os.path.join(HERE, 'conventions', 'factors.json'))
    ap.add_argument('--rivet', default=os.path.join(HERE, 'rivet_ref'))
    ap.add_argument('--targets', default=os.path.join(HERE, '..', 'deposit', 'targets_thrust_nlo.npz'))
    ap.add_argument('--variations', action='store_true', help='also impose every variation of the targets')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    edges = MF.yoda_table(f'{a.rivet}/DELPHI_1996_S3430090.yoda', 'DELPHI_1996_S3430090', 'd11-x01-y02', False)[0]
    sel = np.zeros(len(edges) - 1, bool)
    sel[(edges[:-1] >= WIN[0] - 1e-9) & (edges[1:] <= WIN[1] + 1e-9)] = True
    assert sel.sum() == 12, sel.sum()
    fac = json.load(open(a.factors))
    T = maxent.load_targets(a.targets)
    whichs = [None, 'central'] + (sorted(T.variations) if a.variations else [])
    res = dict(meta=dict(edges=edges.tolist(), window_bins=sel.tolist(), targets=os.path.abspath(a.targets),
                         factors=os.path.abspath(a.factors), ngroup=NGROUP, floor_rel=T.floor_rel,
                         variations=sorted(T.variations)), samples={})
    for spec in a.sample:
        gen, point, d = spec.split(':', 2)
        F0 = np.array(fac[gen]['delphi_y02']['c_over_a'], float)
        F0e = np.array(fac[gen]['delphi_y02']['c_over_a_err'], float)
        assert np.allclose(fac[gen]['delphi_y02']['lo'], edges[:-1]) and np.allclose(fac[gen]['delphi_y02']['hi'], edges[1:])
        tau_c, tau_a, cid, nch = load(d)
        entry = dict(gen=gen, point=point, dir=d, n_events=int(len(tau_c)), n_chunks=int(nch), cases={})
        for which in whichs:
            o = run(tau_c, tau_a, cid, nch, edges, F0, sel, T, which)
            if point != 'default':                                 # the default factor comes from other events
                rb, eb = o['res_bin']; o['res_bin'] = (rb, np.sqrt(eb**2 + ((1 + rb)*F0e/F0)**2))
                rP, eP = o['res_P']
                fa = None
                # error of sum_b F0_b fa_b from the factor errors, relative to P_fixed
                fa_b = fractions(tau_a, np.ones(len(tau_a)), edges)
                eF = np.sqrt(((F0e*fa_b)[sel]**2).sum())/((F0*fa_b)[sel].sum())
                o['res_P'] = (rP, np.sqrt(eP**2 + ((1 + rP)*eF)**2))
            key = 'none' if which is None else which
            entry['cases'][key] = {k: (v if k == 'info' else [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()])
                                   for k, v in o.items()}
            rP, eP = o['res_P']; rb, eb = o['res_bin']
            print(f'{gen:6s} {point:7s} {key:26s} window fraction: fixed/exact - 1 = {100*rP[0]:+.3f} +- {100*eP[0]:.3f} %;'
                  f' bins 0.05-0.30 |max| {100*np.nanmax(np.abs(rb[sel])):.3f} %, bin 0.30-0.35 {100*rb[17]:+.3f} +- {100*eb[17]:.3f} %,'
                  f' bins below 0.05 {np.round(100*rb[:5], 2).tolist()}', flush=True)
        res['samples'][f'{gen}:{point}'] = entry
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
