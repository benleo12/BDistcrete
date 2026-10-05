#!/usr/bin/env python3
"""Merge the per-job toy_law_ruler outputs into output/kscan_toy_law_ruler.json and print a
before/after table against the published output/split_scan.json and output/budget_law.json.

Every law quantity is in chi2/ndf (the law is linear in chi2/ndf, not in the width). On the
paper's ruler chi2/ndf is the pooled sum of squared pulls over the pooled sum of (nbins - 1);
the width is its square root.
"""
import json, glob
import numpy as np

D = 'output/_law'
SPLITS = [(12, 5000), (60, 1000), (600, 100), (6000, 10), (60000, 1)]
N_CLEAN = [2000, 5000, 10000, 20000, 35000, 50000]
N_CTRL = [2000, 5000, 10000, 20000, 50000, 100000, 200000]
NBIN_PAPER = 25          # make_bins: NBIN=26 edges -> 25 bins per histogram
NBIN_OLD = 29            # np.linspace(lo, hi, 30) -> 30 edges -> 29 bins per histogram


def fit_Nstar(N, y):
    """budget_law.fit_Nstar verbatim: weighted LS slope of (chi2-1) vs N through the origin."""
    N = np.asarray(N, float); d = np.asarray(y, float) - 1.0
    slope = float(np.sum(N * d) / np.sum(N * N))
    ss_res = float(np.sum((d - slope * N) ** 2)); ss_tot = float(np.sum((d - d.mean()) ** 2))
    return (1.0 / slope if slope > 0 else float('nan')),\
           (1 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def load_published(path, empty):
    """The published JSON for the before/after columns, or a nan-filled stand-in of the same
    shape when it is missing. The merge itself does not depend on it."""
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        print(f'{path} not found: the published column reads nan')
        return empty


NAN = float('nan')
out = {}

# ---------------------------------------------------------------- dense coverage
pub = load_published('output/split_scan.json',
                     {f'{M}x{E}': dict(chi2=NAN, spread=NAN) for M, E in SPLITS})
out['split'] = {}
print('DENSE COVERAGE  (chi2/ndf, fixed total M*E = 60000 training events)')
print(f'{"split":>12} | {"published":>9} {"spread":>6} | {"reprod(old)":>11} | '
      f'{"PAPER ruler":>11} {"spread":>6} | {"width":>6}')
print('-' * 78)
for M, E in SPLITS:
    fs = [f'{D}/split_M{M}_E{E}_s{sd}.json' for sd in (0, 1)]
    try:
        rs = [json.load(open(f)) for f in fs]
    except FileNotFoundError:
        print(f'{M}x{E}: MISSING'); continue
    key = f'{M}x{E}'
    pa = [r['chi2_paper'] for r in rs]; ol = [r['chi2_old'] for r in rs]
    wd = [r['width_paper'] for r in rs]
    out['split'][key] = dict(M=M, E=E,
                             chi2_published=pub[key]['chi2'], spread_published=pub[key]['spread'],
                             chi2_old_reproduced=float(np.mean(ol)), spread_old=float(np.std(ol)),
                             chi2_paper=float(np.mean(pa)), spread_paper=float(np.std(pa)),
                             width_paper=float(np.mean(wd)), spread_width=float(np.std(wd)),
                             per_seed_chi2_paper=pa, per_seed_chi2_old=ol)
    print(f'{key:>12} | {pub[key]["chi2"]:9.3f} {pub[key]["spread"]:6.3f} | '
          f'{np.mean(ol):11.3f} | {np.mean(pa):11.3f} {np.std(pa):6.3f} | {np.mean(wd):6.3f}')

# ---------------------------------------------------------------- accuracy law
laws = []
for sd in (0, 1, 2):
    try:
        laws.append(json.load(open(f'{D}/law_s{sd}.json')))
    except FileNotFoundError:
        pass
if laws:
    pubL = load_published('output/budget_law.json',
                          dict(clean=dict(Nstar=NAN, R2_linearity=NAN, Nstar_sd_over_seeds=NAN,
                                          chi2=[NAN] * len(N_CLEAN)),
                               control=dict(chi2=[NAN] * len(N_CTRL),
                                            implied_Nstar_per_point=[None] * len(N_CTRL)),
                               per_seed_Nstar=[]))
    clean_paper = {N: [v for L in laws for v in L['clean'][str(N)]['chi2_paper_reps']] for N in N_CLEAN}
    clean_old = {N: [v for L in laws for v in L['clean'][str(N)]['chi2_old_reps']] for N in N_CLEAN}
    cl_p = [float(np.mean(clean_paper[N])) for N in N_CLEAN]
    cl_o = [float(np.mean(clean_old[N])) for N in N_CLEAN]
    sem_p = [float(np.std(clean_paper[N]) / np.sqrt(len(clean_paper[N]))) for N in N_CLEAN]
    Ns_p, r2_p = fit_Nstar(N_CLEAN, cl_p)
    Ns_o, r2_o = fit_Nstar(N_CLEAN, cl_o)
    per_seed_p = [fit_Nstar(N_CLEAN, [float(np.mean(L['clean'][str(N)]['chi2_paper_reps']))
                                      for N in N_CLEAN])[0] for L in laws]
    per_seed_o = [fit_Nstar(N_CLEAN, [float(np.mean(L['clean'][str(N)]['chi2_old_reps']))
                                      for N in N_CLEAN])[0] for L in laws]
    ctrl_p = [float(np.mean([L['control'][str(N)]['chi2_paper'] for L in laws])) for N in N_CTRL]
    ctrl_o = [float(np.mean([L['control'][str(N)]['chi2_old'] for L in laws])) for N in N_CTRL]

    # the null of the estimator, measured with the analytic (epsilon = 0) weights. The
    # published law forces the intercept to 1; that is only right if the estimator reads
    # exactly 1 for a perfect model, which is what these runs test.
    nulls = []
    for sd in (0, 1, 2):
        try:
            nulls.append(json.load(open(f'{D}/null_s{sd}.json')))
        except FileNotFoundError:
            pass
    null_p = [float(np.mean([v for L in nulls for v in L['clean'][str(N)]['chi2_paper_reps']]))
              for N in N_CLEAN] if nulls else None
    null_o = [float(np.mean([v for L in nulls for v in L['clean'][str(N)]['chi2_old_reps']]))
              for N in N_CLEAN] if nulls else None

    def free_fit(N, y):
        """Two-parameter fit chi2 = c + N/N*, the law with its intercept released."""
        N = np.asarray(N, float); y = np.asarray(y, float)
        A = np.vstack([N, np.ones_like(N)]).T
        (sl, c), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ [sl, c]
        r2 = 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
        return (1.0 / sl if sl > 0 else float('nan')), float(c), r2

    def implied(N, c): return [float(n / (x - 1)) if x > 1 else None for n, x in zip(N, c)]
    def spread(v):
        v = [x for x in v if x is not None and np.isfinite(x)]
        return (max(v) / min(v)) if v else float('nan')

    out['law'] = dict(
        N_clean=N_CLEAN, N_control=N_CTRL, n_seeds=len(laws), reps=3,
        chi2_clean_paper=cl_p, chi2_clean_old=cl_o, sem_clean_paper=sem_p,
        Nstar_paper=Ns_p, Nstar_old=Ns_o,
        R2_paper=r2_p, R2_old=r2_o,
        per_seed_Nstar_paper=per_seed_p, per_seed_Nstar_old=per_seed_o,
        Nstar_sd_over_seeds_paper=float(np.std(per_seed_p)),
        Nstar_sd_over_seeds_old=float(np.std(per_seed_o)),
        implied_Nstar_clean_paper=implied(N_CLEAN, cl_p),
        chi2_control_paper=ctrl_p, chi2_control_old=ctrl_o,
        implied_Nstar_control_paper=implied(N_CTRL, ctrl_p),
        implied_Nstar_control_old=implied(N_CTRL, ctrl_o),
        control_spread_factor_paper=spread(implied(N_CTRL, ctrl_p)),
        control_spread_factor_old=spread(implied(N_CTRL, ctrl_o)),
        nbins_per_histogram_paper=NBIN_PAPER, nbins_per_histogram_old=NBIN_OLD,
        epsilon_paper=float(np.sqrt(NBIN_PAPER / Ns_p)), epsilon_old=float(np.sqrt(NBIN_OLD / Ns_o)),
        null_chi2_paper=null_p, null_chi2_old=null_o,
        free_fit_paper=free_fit(N_CLEAN, cl_p), free_fit_old=free_fit(N_CLEAN, cl_o),
        free_fit_control_paper=free_fit(N_CTRL, ctrl_p), free_fit_control_old=free_fit(N_CTRL, ctrl_o),
        published=dict(Nstar=pubL['clean']['Nstar'], R2=pubL['clean']['R2_linearity'],
                       per_seed_Nstar=pubL['per_seed_Nstar'],
                       Nstar_sd=pubL['clean']['Nstar_sd_over_seeds'],
                       chi2_clean=pubL['clean']['chi2'], chi2_control=pubL['control']['chi2'],
                       implied_control=pubL['control']['implied_Nstar_per_point']))

    print('\nACCURACY LAW  chi2/ndf = 1 + N_tgt/N*   (CLEAN arm, nref = 10^6)')
    print(f'{"N_tgt":>8} | {"published":>9} | {"reprod(old)":>11} | {"PAPER ruler":>11} {"+-":>6}')
    print('-' * 60)
    for i, N in enumerate(N_CLEAN):
        print(f'{N:>8} | {pubL["clean"]["chi2"][i]:9.3f} | {cl_o[i]:11.3f} | {cl_p[i]:11.3f} '
              f'{sem_p[i]:6.3f}')
    print(f'\n  N*            published {pubL["clean"]["Nstar"]/1e3:6.0f}k | '
          f'reprod(old) {Ns_o/1e3:6.0f}k | PAPER {Ns_p/1e3:6.0f}k')
    print(f'  per-seed N*   published {[round(x/1e3) for x in pubL["per_seed_Nstar"]]}k | '
          f'reprod(old) {[round(x/1e3) for x in per_seed_o]}k | PAPER {[round(x/1e3) for x in per_seed_p]}k')
    print(f'  N* seed sd    published {pubL["clean"]["Nstar_sd_over_seeds"]/1e3:6.0f}k | '
          f'PAPER {np.std(per_seed_p)/1e3:6.0f}k')
    print(f'  R^2 linearity published {pubL["clean"]["R2_linearity"]:.3f} | '
          f'reprod(old) {r2_o:.3f} | PAPER {r2_p:.3f}')
    print(f'  epsilon       published {np.sqrt(30/pubL["clean"]["Nstar"])*100:.2f}% (30 edges/29 bins) | '
          f'PAPER {np.sqrt(NBIN_PAPER/Ns_p)*100:.2f}% ({NBIN_PAPER} bins)')
    if null_p:
        print('\n  MEASURED NULL of each estimator (analytic weights, epsilon = 0):')
        print(f'{"N_tgt":>8} | {"old":>7} | {"PAPER":>7}')
        for i, N in enumerate(N_CLEAN):
            print(f'{N:>8} | {null_o[i]:7.3f} | {null_p[i]:7.3f}')
        print(f'  mean null: old {np.mean(null_o):.3f}   PAPER {np.mean(null_p):.3f}   (the law assumes 1)')
        # N* refit against the MEASURED null instead of the assumed 1
        cp = fit_Nstar(N_CLEAN, [c - n + 1 for c, n in zip(cl_p, null_p)])
        co = fit_Nstar(N_CLEAN, [c - n + 1 for c, n in zip(cl_o, null_o)])
        out['law']['Nstar_nullcorrected_paper'] = cp[0]; out['law']['R2_nullcorrected_paper'] = cp[1]
        out['law']['Nstar_nullcorrected_old'] = co[0]; out['law']['R2_nullcorrected_old'] = co[1]
        print(f'  N* after subtracting the measured null: old {co[0]/1e3:.0f}k (R^2 {co[1]:.3f}) | '
              f'PAPER {cp[0]/1e3:.0f}k (R^2 {cp[1]:.3f})')
        print(f'  epsilon from the null-corrected N*: old {np.sqrt(NBIN_OLD/co[0])*100:.2f}% | '
              f'PAPER {np.sqrt(NBIN_PAPER/cp[0])*100:.2f}%')
    fp, fo = out['law']['free_fit_paper'], out['law']['free_fit_old']
    print(f'\n  LAW WITH THE INTERCEPT RELEASED (chi2 = c + N/N*):')
    print(f'    old   : c={fo[1]:.3f}  N*={fo[0]/1e3:.0f}k  R^2={fo[2]:.3f}')
    print(f'    PAPER : c={fp[1]:.3f}  N*={fp[0]/1e3:.0f}k  R^2={fp[2]:.3f}')
    gp, go = out['law']['free_fit_control_paper'], out['law']['free_fit_control_old']
    print(f'  same fit on the CONTROL arm (nref=2e5), where the reference error contaminates:')
    print(f'    old   : c={go[1]:.3f}  N*={go[0]/1e3:.0f}k  R^2={go[2]:.3f}')
    print(f'    PAPER : c={gp[1]:.3f}  N*={gp[0]/1e3:.0f}k  R^2={gp[2]:.3f}')
    print('\nCONTROL arm (nref = 2x10^5): implied N* per point, in thousands')
    print(f'  published   {[None if v is None else round(v/1e3) for v in pubL["control"]["implied_Nstar_per_point"]]}')
    print(f'  reprod(old) {[None if v is None else round(v/1e3) for v in implied(N_CTRL, ctrl_o)]}')
    print(f'  PAPER       {[None if v is None else round(v/1e3) for v in implied(N_CTRL, ctrl_p)]}')
    print(f'  spread factor max/min:  published {spread(pubL["control"]["implied_Nstar_per_point"]):.0f}x'
          f' | reprod(old) {spread(implied(N_CTRL, ctrl_o)):.0f}x | PAPER {spread(implied(N_CTRL, ctrl_p)):.0f}x')

json.dump(out, open('output/kscan_toy_law_ruler.json', 'w'), indent=1)
print('\n-> output/kscan_toy_law_ruler.json')
