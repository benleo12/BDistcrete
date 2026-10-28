#!/usr/bin/env python3
"""The paper's central prediction and generator band, from the deposited reference sample.

    python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz

This is the end-to-end check of the release, and it needs numpy alone. It reweights the 1,149,840
events of reference_MIX17.npz with the MIX17 head, imposes the NNLL+NLO thrust moments of
targets_thrust_nlo.npz with gentune.maxent, and compares the result with the numbers of the paper,
which analysis/three_bands.py computed with the paper's torch code.

  central point    the default tunes of both generators and a fraction of one half, which is
                   gentune.axes.default('MIX17'). The seven means of the paper before and after
                   the calculation is imposed, and the DELPHI thrust distribution, normalized over
                   the bins above 1 - T = 0.05 as in the paper's figure.
  generator band   the standard deviation of every prediction over 200 points of a Latin hypercube
                   with seed 11 and the fraction uniform on [0, 1], built as three_bands.py builds
                   them, before and after the calculation is imposed at every point.

The seven means are the mean of 1 - T inside the window of the calculation, the fraction of events
in that window, and the full means of 1 - T, B_T, rho_H, n_ch and n_baryon.

The paper's numbers are written out below (PAPER), copied from the output of three_bands.py, so no
other file is needed. --paper three_bands_MIXGEOV3S9.npz compares with that file instead, which
adds the prediction at each of the 200 points and the mean and percentiles of each band.

Every number agrees with the paper to a few 1e-8 relative or better: the means to 1e-9 and the
bins of the distribution to 7e-9 at the central point, the bands to 1.2e-8, and with --paper the
predictions at each of the 200 points to 2.2e-8, the largest in the last bin, which holds the
fewest events. What is left comes from the head. The paper's
code evaluated it in float32 and gentune evaluates it in float64, which moves each logit by up to
2e-6. With the head evaluated in float32 as well, the means at the central point agree to 4e-11,
and with the same logits gentune.maxent gives the paper's weights to 1e-13. The check fails above
--tol, 1e-6 by default.

Memory peaks at 2.5 GB. On 8 cores the 201 points take about six minutes.
"""
import argparse, json, os, sys, time
import numpy as np

# the release root, whether this file sits in examples/ or next to gentune/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, 'gentune')) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from gentune import load_head, axes
from gentune.maxent import load_targets, logsumexp

OBS = ('tau_win', 'pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')
LABEL = {'tau_win': '<1-T> in window', 'pwin': 'window fraction', 'thrust': '<1-T>',
         'B_total': '<B_T>', 'rho_heavy': '<rho_H>', 'nch': '<n_ch>', 'nbaryon': '<n_baryon>'}

# DELPHI's thrust distribution at 91.2 GeV in tau = 1 - T, the bins as the paper's code reads them
# from the Rivet reference table. They are kept as read, rounding included, so that every event
# falls in the paper's bin: an event goes in bin b when DELPHI_LO[b] <= tau < DELPHI_LO[b + 1]
# (DELPHI_HI[-1] for the last), and the width of bin b is DELPHI_HI[b] - DELPHI_LO[b].
DELPHI_LO = [0.0, 0.009999999999999998, 0.02, 0.030000000000000002, 0.04, 0.05,
             0.060000000000000005, 0.06999999999999999, 0.08, 0.09, 0.1, 0.12000000000000001,
             0.13999999999999999, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]
DELPHI_HI = [0.01, 0.02, 0.030000000000000002, 0.04, 0.049999999999999996, 0.06, 0.07, 0.08,
             0.09000000000000001, 0.1, 0.12, 0.14, 0.16, 0.18000000000000002, 0.2, 0.25,
             0.30000000000000004, 0.35000000000000003, 0.4, 0.5]
FIRST_BIN = 0.05       # the distribution is normalized over the bins from here up, as in the paper

# The paper's numbers, from analysis/three_bands.py (output/three_bands_MIXGEOV3S9.json and .npz).
# Means in the order of OBS, histograms over the twenty DELPHI bins, bands the standard deviation
# over the 200 points.
PAPER = {
    'central_before': [0.11135172803353859, 0.43323983421623, 0.0648320785991186,
                       0.10565888580274842, 0.051467082692212084, 17.814600278700894,
                       1.4643048274527113],
    'central_after': [0.11204387167561708, 0.4329771130228314, 0.06511019681108496,
                      0.10598159568512394, 0.05169562582253042, 17.83108249108656,
                      1.4660179886890963],
    'hist_before': [4.846718434919002, 30.709498954332638, 39.39653010143413, 31.42934008055114,
                    23.280682279485376, 17.35713336169944, 13.295875072611585, 10.51884003097767,
                    8.503678203378913, 7.005215908074184, 5.347762045674217, 3.8631139154295493,
                    2.935224756446528, 2.2574974111654202, 1.7575959650630537, 1.1868959523144105,
                    0.6777255075594634, 0.2921700633477417, 0.04100194907349394,
                    0.0007901874225120009],
    'hist_after': [4.851881353661874, 30.74221194351633, 39.4384968643895, 31.46281987335643,
                   23.305481795385006, 16.38339548738769, 12.849952214855795, 10.381172522386509,
                   8.545638537485454, 7.148320298823661, 5.552361064039017, 4.074806524739437,
                   3.117078684843731, 2.3940208185041123, 1.8473180981049553, 1.2084374333905272,
                   0.6366872143742723, 0.256317779638131, 0.04104562599308581,
                   0.0007910291618267275],
    'band_before': [0.0018576445457131486, 0.036586795938050524, 0.004379860383039147,
                    0.004864173585087794, 0.0027866693123081436, 0.6766301105853058,
                    0.28381812429845943],
    'band_after': [3.027173959961133e-06, 0.03663626098103742, 0.003689365202352174,
                   0.00427782863706989, 0.0022458887124694464, 0.6580623231868581,
                   0.2846889604010119],
    'mean_before': [0.11223211094445938, 0.4780934395746025, 0.0697981244877086,
                    0.1114406650591889, 0.054794881527807086, 18.318983436425906,
                    1.6227259462798276],
    'mean_after': [0.11204528993281283, 0.4777455291193817, 0.06963201422287729,
                   0.11138407486084104, 0.054688219479541764, 18.316849051552666,
                   1.6225116187987934],
    'hist_band_before': [1.359214542440052, 5.937217975143935, 5.015680016697653,
                         2.680574453823592, 1.4140623219645767, 0.8106271086556877,
                         0.42593936824457435, 0.2189134361124912, 0.10414836331839115,
                         0.05474837964346163, 0.09464535180226186, 0.10856008969764205,
                         0.10736298926463689, 0.0904718418438556, 0.08022692104589302,
                         0.05577794757342088, 0.039412559723995155, 0.025122984951075494,
                         0.009584658233227742, 0.00032815130211980877],
    'hist_band_after': [1.3623714408266816, 5.950071081535899, 5.029519496762826,
                        2.695912659828234, 1.4297661446927203, 0.01988037623459363,
                        0.022009874033985726, 0.03301158430259792, 0.025743929736519254,
                        0.02127498050374533, 0.01804082560539025, 0.009246292339496035,
                        0.0119302384627072, 0.011946281026774365, 0.007112176974974475,
                        0.003458169643956774, 0.001991352278547183, 0.006549108463206354,
                        0.009606936964490572, 0.0003290829742720369],
}


def lhs_nodes(head, n=200, seed=11):
    """The paper's points: a Latin hypercube on [-1, 1] in every parameter, mapped onto the box,
    written as in analysis/three_bands.py, so the same n and seed give the same points."""
    rng = np.random.default_rng(seed)
    U = (np.argsort(rng.random((n, head.nt)), axis=0) + rng.random((n, head.nt)))/n*2 - 1
    return head.centre + head.norm[:, 1]*U


class Reference:
    """The reference sample and everything a prediction needs from it, computed once: the event
    vectors as one matrix, the per-event observables, the window and moment functions of the
    calculation, and the DELPHI bin of every event."""

    def __init__(self, path, head, calc):
        r = np.load(path)
        # float64 once, here, so that every one of the 201 evaluations of the head is a plain
        # float64 product; the values are those of the float32 file, converted exactly
        AE = r['AE']
        self.N = AE.shape[1]
        self.A = np.empty((self.N, head.ENS*head.K))
        for m in range(head.ENS):
            self.A[:, m*head.K:(m + 1)*head.K] = AE[m]
        del AE
        self.tau = r['1_minus_thrust'].astype(np.float64)
        self.x = {'thrust': self.tau,
                  **{k: r[k].astype(np.float64) for k in ('B_total', 'rho_heavy', 'nch', 'nbaryon')}}
        self.inwin, self.F = calc.functions(self.tau)
        lo, hi = np.array(DELPHI_LO), np.array(DELPHI_HI)
        b = np.searchsorted(np.r_[lo, hi[-1]], self.tau, side='right') - 1
        self.ins = (b >= 0) & (b < len(lo))
        self.bin = b[self.ins]
        self.width = hi - lo
        self.fit = lo >= FIRST_BIN - 1e-12

    def predictions(self, w):
        """The seven means of OBS and the DELPHI histogram, normalized over the fitted bins, for
        weights w that sum to one."""
        win = w[self.inwin]
        pw = win.sum()
        means = [win @ self.tau[self.inwin]/pw, pw] + [w @ self.x[k] for k in OBS[2:]]
        cnt = np.bincount(self.bin, weights=w[self.ins], minlength=len(self.width))
        return np.array(means), cnt/(self.width*cnt[self.fit].sum())


def at_point(theta, head, ref, calc):
    """(means, histogram) before and after the calculation at theta, and the solve's info."""
    lw0 = head.logit(ref.A, theta)/head.T
    w0 = np.exp(lw0 - logsumexp(lw0))
    logw, _, info = calc.impose(lw0, ref.inwin, ref.F)
    return ref.predictions(w0), ref.predictions(np.exp(logw)), info


def rel(mine, paper):
    """Largest relative difference, |mine - paper|/|paper|."""
    mine, paper = np.asarray(mine, np.float64), np.asarray(paper, np.float64)
    return float(np.max(np.abs(mine - paper)/np.abs(paper)))


def read_paper(path):
    """The paper's numbers from the output of three_bands.py (its .npz and the .json next to it),
    in the layout of PAPER, plus the 200 points and the predictions at each."""
    z = np.load(path)
    with open(os.path.splitext(path)[0] + '.json') as f:
        j = json.load(f)
    obs = [str(o) for o in z['obs']]
    assert tuple(obs) == OBS, f'{path} holds the observables {obs}'
    c, g = j['central'], j['generator_band']
    full = lambda k: np.r_[z[k + '_low'], z[k]]
    P = {'central_before': [c['generator_only'][k] for k in OBS],
         'central_after': [c['imposed'][k] for k in OBS],
         'hist_before': full('h_central_before'), 'hist_after': full('h_central'),
         'hist_band_before': full('gen_hist_before'), 'hist_band_after': full('gen_hist')}
    for s in ('before', 'after'):
        P['band_' + s] = [g[s][k]['sd'] for k in OBS]
        for q in ('mean', 'p16', 'p84'):
            P[f'{q}_{s}'] = [g[s][k][q] for k in OBS]
    P.update(theta=np.array(c['theta']), nodes=z['nodes'], rows_before=z['rows_before'],
             rows_after=z['rows_after'], H_before=z['H_before'], H_after=z['H_after'],
             edges_lo=np.r_[z['lo_low'], z['lo']], edges_hi=np.r_[z['hi_low'], z['hi']])
    return P


def main():
    ap = argparse.ArgumentParser(description="The paper's central prediction and generator band, "
                                             "with numpy, from the deposited reference sample.")
    ap.add_argument('--reference', required=True, help='reference_MIX17.npz')
    ap.add_argument('--targets', required=True, help='targets_thrust_nlo.npz')
    ap.add_argument('--head', default=os.path.join(ROOT, 'models', 'MIX17_head.npz'))
    ap.add_argument('--paper', help="three_bands_MIXGEOV3S9.npz, the paper's output, to compare "
                                    "with instead of the numbers in this file")
    ap.add_argument('--nodes', type=int, default=200, help='points of the Latin hypercube')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--tol', type=float, default=1e-6, help='largest relative difference passed')
    ap.add_argument('--out', help='npz for every prediction this run makes')
    a = ap.parse_args()
    t0 = time.time()

    head = load_head(a.head)
    calc = load_targets(a.targets)
    ref = Reference(a.reference, head, calc)
    P = PAPER
    if a.paper:
        P = read_paper(a.paper)
        d = max(rel(PAPER[k], P[k]) for k in PAPER)
        print(f'{a.paper}: the numbers written in this file differ from it by {d:.1e}')
        assert np.array_equal(P['edges_lo'], DELPHI_LO) and np.array_equal(P['edges_hi'], DELPHI_HI), \
            'the DELPHI bins of this file are not those of the paper'
    print(f'{ref.N:,} reference events, {int(ref.inwin.sum()):,} in the window '
          f'[{calc.window[0]:.4g}, {calc.window[1]:.4g}) of {len(calc.names)} moments '
          f'[{time.time() - t0:.0f} s]')

    # -- the central point -----------------------------------------------------------------------
    theta = axes.default('MIX17')
    (m0, h0), (m1, h1), info = at_point(theta, head, ref, calc)
    print(f'\ncentral point, the default tunes and a fraction of one half: Newton converged '
          f'{info["converged"]} in {info["iterations"]} steps, N_eff {info["n_eff"]:,.0f} '
          f'[{time.time() - t0:.0f} s]')
    print(f'{"":<17}{"before":>15}{"paper":>15}{"rel. diff":>11}{"after":>15}{"paper":>15}'
          f'{"rel. diff":>11}')
    for j, k in enumerate(OBS):
        print(f'{LABEL[k]:<17}{m0[j]:>15.10g}{P["central_before"][j]:>15.10g}'
              f'{rel(m0[j], P["central_before"][j]):>11.1e}{m1[j]:>15.10g}'
              f'{P["central_after"][j]:>15.10g}{rel(m1[j], P["central_after"][j]):>11.1e}')
    print(f'\nDELPHI thrust distribution, normalized over the bins from {FIRST_BIN}')
    print(f'{"from":>6}{"to":>6}{"before":>13}{"paper":>13}{"rel. diff":>11}{"after":>13}'
          f'{"paper":>13}{"rel. diff":>11}')
    for b in range(len(h1)):
        print(f'{DELPHI_LO[b]:>6.2f}{DELPHI_HI[b]:>6.2f}{h0[b]:>13.7g}{P["hist_before"][b]:>13.7g}'
              f'{rel(h0[b], P["hist_before"][b]):>11.1e}{h1[b]:>13.7g}{P["hist_after"][b]:>13.7g}'
              f'{rel(h1[b], P["hist_after"][b]):>11.1e}')
    checks = [('central point, means before', rel(m0, P['central_before']), a.tol),
              ('central point, means after', rel(m1, P['central_after']), a.tol),
              ('central point, histogram before', rel(h0, P['hist_before']), a.tol),
              ('central point, histogram after', rel(h1, P['hist_after']), a.tol)]
    if a.paper:
        checks.append(('central point, theta', float(np.max(np.abs(theta - P['theta']))), 0.0))

    # -- the generator band ----------------------------------------------------------------------
    nodes = lhs_nodes(head, a.nodes, a.seed)
    rows0, rows1, H0, H1, its, conv = [], [], [], [], [], []
    for i, th in enumerate(nodes):
        (r0, g0), (r1, g1), inf = at_point(th, head, ref, calc)
        rows0.append(r0); rows1.append(r1); H0.append(g0); H1.append(g1)
        its.append(inf['iterations']); conv.append(inf['converged'])
        if i % 25 == 0 or i == len(nodes) - 1:
            print(f'  point {i + 1}/{len(nodes)} [{time.time() - t0:.0f} s]', flush=True)
    rows0, rows1, H0, H1 = map(np.array, (rows0, rows1, H0, H1))
    print(f'Newton converged at {sum(conv)} of {len(nodes)} points, in {min(its)} to {max(its)} steps')
    paper_points = (a.nodes, a.seed) == (200, 11)
    if not paper_points:
        print("these are not the paper's points (200 with seed 11), so the bands are not compared")
    else:
        if a.paper:
            checks.append((f'generator band, the {len(nodes)} points',
                           float(np.max(np.abs(nodes - P['nodes']))), 0.0))
        band = {s: (R.std(0, ddof=1), R.mean(0), H.std(0, ddof=1))
                for s, R, H in (('before', rows0, H0), ('after', rows1, H1))}
        print(f'\ngenerator band, the standard deviation over {len(nodes)} points')
        print(f'{"":<17}{"before":>13}{"paper":>13}{"rel. diff":>11}{"after":>13}{"paper":>13}'
              f'{"rel. diff":>11}')
        for j, k in enumerate(OBS):
            print(f'{LABEL[k]:<17}' + ''.join(
                f'{band[s][0][j]:>13.7g}{P["band_" + s][j]:>13.7g}'
                f'{rel(band[s][0][j], P["band_" + s][j]):>11.1e}' for s in ('before', 'after')))
        for s, R, H in (('before', rows0, H0), ('after', rows1, H1)):
            sd, mu, hsd = band[s]
            checks.append((f'generator band {s}, means', rel(sd, P['band_' + s]), a.tol))
            checks.append((f'generator band {s}, histogram', rel(hsd, P['hist_band_' + s]), a.tol))
            checks.append((f'generator band {s}, mean over the points', rel(mu, P['mean_' + s]),
                           a.tol))
            if a.paper:
                checks.append((f'generator band {s}, means at every point',
                               rel(R, P['rows_' + s]), a.tol))
                checks.append((f'generator band {s}, histogram at every point',
                               rel(H[:, ref.fit], P['H_' + s]), a.tol))
                for q, v in (('p16', np.percentile(R, 16, axis=0)),
                             ('p84', np.percentile(R, 84, axis=0))):
                    checks.append((f'generator band {s}, {q} over the points',
                                   rel(v, P[f'{q}_{s}']), a.tol))

    # -- verdict ---------------------------------------------------------------------------------
    print(f'\nlargest relative difference from the paper{" (" + a.paper + ")" if a.paper else ""}')
    ok = True
    for name, d, tol in checks:
        good = d <= tol
        ok &= good
        print(f'  {name:<48}{d:>10.1e}   {"ok" if good else "ABOVE " + format(tol, ".0e")}')
    if a.out:
        np.savez(a.out, theta=theta, central_before=m0, central_after=m1, hist_before=h0,
                 hist_after=h1, nodes=nodes, rows_before=rows0, rows_after=rows1, H_before=H0,
                 H_after=H1, iterations=np.array(its), converged=np.array(conv),
                 obs=np.array(OBS), lo=np.array(DELPHI_LO), hi=np.array(DELPHI_HI))
        print(f'wrote {a.out}')
    print(f'{"PASS" if ok else "FAIL"}: the paper reproduced with numpy from the two deposited '
          f'files [{time.time() - t0:.0f} s]')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
