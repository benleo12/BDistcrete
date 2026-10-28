"""gentune.maxent against the paper's torch code, directlib.Model.tilt_on, both in float64.

NOT NUMPY ONLY, and not needed to use gentune: tests/test_maxent.py is the check that needs numpy
alone. This one needs torch and scipy, and the paper's analysis code next to the release folder,
analysis/directlib.py with the modules it imports (analysis/fit_alpha0.py, analysis/np_shift.py),
so it runs in a full clone of the repository, from analysis/release, and not in a copy of the
release folder alone. It runs the same problem through both and prints the largest differences.
The cases:

  synthetic   50,000 made-up events with a thrust-like tau, imposed with the paper's calculation
              and all its variations, and with a made-up second calculation (six other moments in
              another order, another window, a rank-deficient covariance of its own)
  selftest    the 2000 reference events of models/MIX17_selftest.npz, reweighted by the MIX17
              head to the paper's central point (the default tunes, fraction one half), with the
              same calculations. The bundle carries no tau. With --reference the tau of those
              events is read from the reference sample at the bundle's reference_index, with --tau
              from a file of their 2000 values, and without either a synthetic tau is drawn.

Printed per case: dlogw, the largest difference of the normalized log weights, which is the
largest relative difference of any weight; dlam, the largest difference of the multipliers over
the largest multiplier, both taken after the final Newton step Model.tilt_on takes; dlam_J, the
size of that multiplier difference as a change of log w, sqrt(dlam . J dlam); cond, the
condition number of the Newton matrix J; dF, the largest difference of the moment functions.

    python tests/compare_directlib.py targets_thrust_nlo.npz [--reference reference_MIX17.npz]
"""
import argparse, os, sys, warnings
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                       # gentune
sys.path.insert(1, os.path.dirname(os.path.dirname(HERE)))      # analysis/, for directlib
try:
    import torch
    from directlib import Model
except ImportError as e:
    sys.exit(f'compare_directlib.py needs torch, scipy and analysis/directlib.py of a full clone '
             f'of the repository ({e}). tests/test_maxent.py is the check that needs numpy alone.')
from gentune import load_head
from gentune.maxent import Targets, load_targets, impose, logsumexp

torch.set_default_dtype(torch.float64)
CENTRAL = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18, 0.102337, 0.654714, 3.528693, 1.849375,
           0.914156, 0.374094, 0.33107, 0.78, 0.5]                # three_bands.py CENTRAL


def torch_tilt(lw0, tau, names, window, c, sigma, floor_rel):
    """Model.tilt_on on these events, called through a stand-in for the Model instance that
    carries only what tilt_on, funs and sigma read. Returns its log weights, the multipliers after
    its final Newton step, the Newton matrix there, and its moment functions."""
    ns = SimpleNamespace(keys=list(names), G={'sigma_pert': np.asarray(sigma)[None, None]},
                         floor_rel=floor_rel, newton=Model.newton)
    inwin = (tau >= window[0]) & (tau < window[1])
    IW = torch.from_numpy(np.flatnonzero(inwin))
    MW = torch.from_numpy(Model.funs(ns, tau[inwin]))
    lw0t = torch.from_numpy(lw0 - logsumexp(lw0))
    Sig = Model.sigma(ns, 0, 0, c)
    lw, lam, Pw, its = Model.tilt_on(ns, lw0t, IW, MW, c, Sig, torch.zeros(len(names)))
    # the multipliers its weights use, lam minus one Newton step, as tilt_on forms them
    Fw = MW - torch.as_tensor(c)[None, :]
    Sc = Pw**2*torch.as_tensor(Sig)
    zw = lw0t[IW] + Fw @ lam
    lZ = torch.logaddexp(torch.log1p(-Pw), torch.logsumexp(zw, 0))
    ww = torch.exp(zw - lZ); m = ww @ Fw
    J = (Fw*ww[:, None]).T @ Fw - torch.outer(m, m) + Sc
    lam1 = lam - torch.linalg.solve(J, m + Sc @ lam)
    return lw.numpy(), lam1.numpy(), J.numpy(), MW.numpy(), int(its)


def compare(label, lw0, tau, T, which='central', floor_rel=0.002):
    c = T[which]
    lw_t, lam_t, J, MW, its_t = torch_tilt(lw0, tau, T.names, T.window, c, T.sigma, floor_rel)
    inwin, F = T.functions(tau)
    lw, lam, info = impose(lw0, inwin, F, c, T.sigma, floor_rel=floor_rel)
    d = lam - lam_t
    r = dict(dlogw=float(np.max(np.abs(lw - lw_t))),
             dlam=float(np.max(np.abs(d))/np.max(np.abs(lam_t))),
             dlam_J=float(np.sqrt(max(d @ J @ d, 0.0))),
             cond=float(np.linalg.cond(J)), dF=float(np.max(np.abs(F - MW))),
             its=(info['iterations'], its_t), res=info['residual'], maxpull=float(np.max(np.abs(info['pulls']))))
    print(f'{label:<44} dlogw={r["dlogw"]:.1e} dlam={r["dlam"]:.1e} dlam_J={r["dlam_J"]:.1e} '
          f'cond={r["cond"]:.1e} dF={r["dF"]:.0e} newton={r["its"][0]}/{r["its"][1]} '
          f'residual={r["res"]:.0e} maxpull={r["maxpull"]:.2f}', flush=True)
    return r


def second_calculation(lw0, tau, rng):
    """A made-up calculation: six other moments in another order, the window [0.04, 0.30), c the
    window means of the sample shifted by a few percent, and a covariance of rank four from four
    made-up variations plus a small diagonal, with two of those variations kept as named ones."""
    names = ('tau', 'log2t', 'tlogt', 'tau3', 'log3t', 't2logt')
    probe = Targets(names, np.zeros(6), np.eye(6), (0.04, 0.30))
    inwin, F = probe.functions(tau)
    w = np.exp(lw0[inwin] - logsumexp(lw0[inwin]))
    mean = w @ F
    c = mean*(1 + 0.03*rng.standard_normal(6))
    D = 0.02*np.abs(c)[:, None]*rng.standard_normal((6, 4))
    sigma = D @ D.T + np.diag((0.005*np.abs(c))**2)
    return Targets(names, c, sigma, (0.04, 0.30), {'up': c + D[:, 0], 'down': c - D[:, 1]})


def run(label, lw0, tau, paper, rng):
    out = [compare(f'{label}, paper central', lw0, tau, paper)]
    for v in paper.variations:
        out.append(compare(f'{label}, paper {v}', lw0, tau, paper, v))
    T2 = second_calculation(lw0, tau, rng)
    for v in ('central', 'up', 'down'):
        out.append(compare(f'{label}, second calculation {v}', lw0, tau, T2, v))
    # not part of the verdict: the paper's sigma_pert has rank five, so without the floor nine
    # combinations of the fourteen moments would be imposed exactly, J has a condition number
    # near 1e21 and neither code is expected to converge
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        compare(f'{label}, paper central, NO FLOOR (illustration)', lw0, tau, paper, floor_rel=0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('targets')
    ap.add_argument('--reference', help='reference_MIX17.npz, for the tau of the self-test events')
    ap.add_argument('--tau', help='tau of the 2000 MIX17 self-test reference events (.npy)')
    ap.add_argument('--models', default=os.path.join(os.path.dirname(HERE), 'models'))
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    paper = load_targets(a.targets)
    print(f'paper targets: {paper}')
    res = []
    tau = 0.42*rng.beta(0.9, 5.0, 50000)
    lw0 = 0.3*rng.standard_normal(50000) + 2.0*tau
    res += run('synthetic 50000', lw0, tau, paper, rng)
    st = np.load(os.path.join(a.models, 'MIX17_selftest.npz'))
    head = load_head(os.path.join(a.models, 'MIX17_head.npz'))
    lw0 = head.logit(st['AE'], np.array(CENTRAL))/head.T
    if a.reference:
        tau = np.load(a.reference)['1_minus_thrust'][st['reference_index']].astype(np.float64)
        lab = 'selftest 2000, real tau'
    elif a.tau:
        tau, lab = np.load(a.tau).astype(np.float64), 'selftest 2000, real tau'
        assert tau.shape == lw0.shape, (tau.shape, lw0.shape)
    else:
        tau, lab = 0.42*rng.beta(0.9, 5.0, len(lw0)), 'selftest 2000, synthetic tau'
    res += run(lab, lw0, tau, paper, rng)
    worst = {k: max(r[k] for r in res) for k in ('dlogw', 'dlam', 'dlam_J', 'dF')}
    print('worst over all cases: ' + '  '.join(f'{k}={v:.1e}' for k, v in worst.items()))
    ok = worst['dlogw'] < 1e-10 and worst['dlam_J'] < 1e-10 and worst['dF'] == 0
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
