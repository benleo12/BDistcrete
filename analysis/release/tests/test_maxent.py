"""Checks of gentune.maxent that need nothing but numpy.

Each check states a property the solution of the module docstring must have and tests it on
made-up events: the weights are normalized and have the stated exponential form, events outside
the window keep their relative weights, the multipliers solve the stationarity condition, no
nearby multipliers do better on the primal objective, a vanishing covariance imposes the window
means exactly, the answer does not depend on the normalization of logw0 or on the starting
multipliers, and a targets file survives being written and read back. Given the paper's targets
file (the deposit targets_thrust_nlo.npz, or the training code's thrust_targets_nlo_point.npz) it
also reads that, checks the other layout of the same numbers reads the same, and imposes it.

    python tests/test_maxent.py [targets_thrust_nlo.npz]
"""
import os, sys, tempfile, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gentune.maxent import (THRUST_MOMENTS, Targets, load_targets, impose, moment_functions,
                            moment_powers, thrust_moments, logsumexp)

# The made-up targets below are random shifts of the window means, with no distribution behind
# them, and the paper's targets do not belong to the made-up events, so most of these solutions
# move weight out of the window and impose() warns that they are not a result. The checks are of
# properties every solution has, which hold there as well, so that warning is not shown here.
warnings.filterwarnings('ignore', message='impose: the window held')

FAILED = []


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')
    if not ok:
        FAILED.append(name)


def sample(n, seed):
    rng = np.random.default_rng(seed)
    tau = 0.42*rng.beta(0.9, 5.0, n)
    return tau, 0.3*rng.standard_normal(n) + 2.0*tau, rng


def made_up(tau, lw0, rng, names=('tau', 'log2t', 'tlogt', 'tau3'), window=(0.05, 1/3), shift=0.03):
    inwin, F = thrust_moments(tau, window, names)
    w = np.exp(lw0[inwin] - logsumexp(lw0[inwin]))
    c = (w @ F)*(1 + shift*rng.standard_normal(len(names)))
    D = 0.02*np.abs(c)[:, None]*rng.standard_normal((len(names), 3))
    return Targets(names, c, D @ D.T, window, {'up': c + D[:, 0]})


def dual(lw0, inwin, F, c, S, lam):
    """D(lam) and the primal objective at the weights of lam, written out independently."""
    lw0 = lw0 - logsumexp(lw0)
    P = np.exp(logsumexp(lw0[inwin]))
    t = lw0.copy(); t[inwin] += (F - c) @ lam
    lZ = logsumexp(t)
    w = np.exp(t - lZ)
    e = w[inwin] @ (F - c)
    primal = w @ (t - lZ - lw0) + 0.5*e @ np.linalg.solve(P**2*S, e)
    return lZ + 0.5*lam @ (P**2*S) @ lam, primal


def main(argv):
    # the names and the functions
    tau = np.array([0.0, 1e-7, 0.01, 0.1, 0.3])
    t = np.clip(tau, 1e-6, None); L = np.log(t)
    ref = {'logt': L, 'tau': t, 'tlogt': t*L, 'log2t': L**2, 'tau2': t**2, 'tlog2t': t*L**2,
           't2logt': t**2*L, 'log3t': L**3, 'tau3': t**3, 'tlog3t': t*L**3, 't3logt': t**3*L,
           't2log2t': t**2*L**2, 'log4t': L**4, 'tau4': t**4}
    M = moment_functions(tau, THRUST_MOMENTS)
    check('the fourteen thrust moments, bit for bit',
          all(np.array_equal(M[:, j], ref[n]) for j, n in enumerate(THRUST_MOMENTS)))
    check('names parse', [moment_powers(n) for n in ('tau', 'logt', 'tlog2t', 't3logt', 'tau5')]
          == [(1, 0), (0, 1), (1, 2), (3, 1), (5, 0)])
    try:
        moment_powers('sqrt_tau'); bad = False
    except ValueError:
        bad = True
    check('an unknown name is refused', bad)

    # the solution on made-up events
    tau, lw0, rng = sample(40000, 1)
    T = made_up(tau, lw0, rng)
    inwin, F = T.functions(tau)
    logw, lam, info = impose(lw0, inwin, F, T.c, T.sigma)
    w = np.exp(logw)
    check('normalized', abs(w.sum() - 1) < 1e-13, f'{w.sum() - 1:.1e}')
    d = logw - (lw0 - logsumexp(lw0))
    check('outside the window only the normalization moves',
          np.ptp(d[~inwin]) < 1e-13, f'spread {np.ptp(d[~inwin]):.1e}')
    lZ = -d[~inwin][0]
    form = np.max(np.abs(d[inwin] - ((F - T.c) @ lam - lZ)))
    check('w = w0 exp(Theta F.lam)/Z with the returned lam', form < 1e-12, f'{form:.1e}')
    S = info['sigma_used']; P = info['pwin_before']
    r = np.linalg.norm(w[inwin] @ (F - T.c) + P**2*S @ lam)
    check('stationarity E_w[Theta F] + P^2 Sigma lam = 0', r < 1e-10, f'{r:.1e}')
    check('converged, and the floor was added',
          info['converged'] and np.allclose(S - T.sigma, np.diag((0.002*T.c)**2), rtol=1e-12, atol=0))
    D0, p0 = dual(lw0, inwin, F, T.c, S, lam)
    worse = 0
    for _ in range(40):
        dl = rng.standard_normal(len(lam))*np.abs(lam)*1e-3
        D1, p1 = dual(lw0, inwin, F, T.c, S, lam + dl)
        worse += (D1 >= D0 - 1e-14) and (p1 >= p0 - 1e-14)
    check('no nearby multipliers lower the dual or the primal', worse == 40, f'{worse}/40')
    mom = info['moments']
    check('info moments are the window means of w', np.allclose(mom, (w[inwin] @ F)/w[inwin].sum(),
                                                                rtol=1e-13, atol=0))

    logw2, lam2, _ = impose(lw0 + 7.3, inwin, F, T.c, T.sigma)
    check('logw0 normalization does not matter', np.max(np.abs(logw2 - logw)) < 1e-12)
    logw3, _, i3 = impose(lw0, inwin, F, T.c, T.sigma, lam0=lam*1.01)
    check('a warm start lands on the same weights', np.max(np.abs(logw3 - logw)) < 1e-12,
          f'{i3["iterations"]} steps against {info["iterations"]}')
    logw4, _, _ = impose(lw0, np.flatnonzero(inwin), moment_functions(tau, T.names), T.c, T.sigma)
    check('window as indices, F over all events', np.max(np.abs(logw4 - logw)) < 1e-12)

    # Sigma -> 0 imposes the means exactly
    T0 = made_up(tau, lw0, rng, names=('tau', 'tau2', 'logt'), shift=0.01)
    inwin0, F0 = T0.functions(tau)
    _, _, i0 = impose(lw0, inwin0, F0, T0.c, 1e-16*np.eye(3), floor_rel=0.0)
    rel = np.max(np.abs(i0['moments'] - T0.c)/np.abs(T0.c))
    check('Sigma -> 0 hits c', rel < 1e-7, f'max relative miss {rel:.1e}')
    _, _, ib = impose(lw0, inwin0, F0, T0.c, 1e6*np.diag(T0.c**2), floor_rel=0.0)
    check('Sigma -> infinity leaves w0 alone', ib['kl'] < 1e-9, f'KL {ib["kl"]:.1e}')

    # every event in the window: the constraints become full-sample means
    allin = np.ones(len(tau), bool)
    Fa = moment_functions(tau, ('tau', 'logt'))
    ca = 1.01*(np.exp(lw0 - logsumexp(lw0)) @ Fa)
    _, _, ia = impose(lw0, allin, Fa, ca, np.zeros((2, 2)))
    check('a window holding every event', ia['converged'] and abs(ia['pwin_after'] - 1) < 1e-12,
          f'pulls {np.round(ia["pulls"], 3)}')

    # files
    with tempfile.TemporaryDirectory() as tmp:
        T.floor_rel = 0.003
        for ext in ('json', 'npz'):
            p = os.path.join(tmp, f't.{ext}')
            T.save(p)
            R = load_targets(p)
            same = (R.names == T.names and R.powers == T.powers and R.window == T.window
                    and R.floor_rel == T.floor_rel and np.array_equal(R.c, T.c)
                    and np.array_equal(R.sigma, T.sigma) and list(R.variations) == list(T.variations)
                    and all(np.array_equal(R.variations[k], T.variations[k]) for k in T.variations))
            check(f'targets survive a round trip through .{ext}', same)
        lw_a, _, _ = R.impose(lw0, *R.functions(tau), which='up')
        lw_b, _, _ = impose(lw0, inwin, F, T.variations['up'], T.sigma, floor_rel=0.003)
        check('Targets.impose uses the variation and the floor of the file', np.array_equal(lw_a, lw_b))
        Q = Targets(['f1', 'f2'], [0.1, 0.02], np.eye(2)*1e-4, (0.05, 0.3), powers=[(1, 0), (2, 0)])
        check('free labels with explicit powers',
              np.array_equal(Q.functions(tau)[1], thrust_moments(tau, (0.05, 0.3), ('tau', 'tau2'))[1]))

        # the paper's calculation, if given, as the deposit targets_thrust_nlo.npz or as the
        # training code's grid file thrust_targets_nlo_point.npz; it is written out in the other
        # layout and must read back to the same calculation
        if len(argv) > 1:
            G = np.load(argv[1])
            Pp = load_targets(argv[1])
            check('the paper targets read', Pp.names == list(THRUST_MOMENTS) and len(Pp.variations) == 12
                  and Pp.floor_rel == 0.002, repr(Pp))
            other = os.path.join(tmp, 'other_layout.npz')
            if 'keys' in G.files:
                np.savez(other, moment_names=G['keys'], moment_powers=np.array([moment_powers(n) for n in G['keys']]),
                         central=G['central'][0, 0], sigma_pert=G['sigma_pert'][0, 0], window=G['window'],
                         floor_rel=np.float64(0.002), alpha_s=G['alphas'][0], alpha_0=G['alpha0'][0],
                         pwin=G['pwin'][0, 0], scale_variations=G['scale'][0, 0], scale_labels=G['scale_labels'],
                         np_variations=G['np_vars'][0, 0], np_labels=G['np_labels'])
            else:
                np.savez(other, keys=G['moment_names'], central=G['central'][None, None],
                         sigma_pert=G['sigma_pert'][None, None], window=G['window'],
                         alphas=np.atleast_1d(G['alpha_s']), alpha0=np.atleast_1d(G['alpha_0']),
                         pwin=np.reshape(G['pwin'], (1, 1)), scale=G['scale_variations'][None, None],
                         scale_labels=G['scale_labels'], np_vars=G['np_variations'][None, None],
                         np_labels=G['np_labels'])
            Pd = load_targets(other)
            same = (Pd.names == Pp.names and Pd.powers == Pp.powers and np.array_equal(Pd.c, Pp.c)
                    and np.array_equal(Pd.sigma, Pp.sigma) and Pd.window == Pp.window
                    and Pd.floor_rel == Pp.floor_rel and list(Pd.variations) == list(Pp.variations)
                    and Pd.meta == Pp.meta
                    and all(np.array_equal(Pd.variations[k], Pp.variations[k]) for k in Pp.variations))
            check('the deposit and the grid layout read to the same calculation', same, str(Pd.meta))
            inwin, F = Pp.functions(tau)
            _, _, ip = Pp.impose(lw0, inwin, F)
            check('the paper targets imposed on made-up events', ip['converged'],
                  f'{ip["iterations"]} steps, max pull {np.max(np.abs(ip["pulls"])):.3f}, '
                  f'N_eff/N {ip["n_eff"]/len(tau):.3f}')

    print('\nall checks pass' if not FAILED else f'\nFAILED: {FAILED}')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
