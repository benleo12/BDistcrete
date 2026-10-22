"""Every accuracy number of the paper, written by script: phase2/numbers.json and phase2/numbers.tex
(one \\newcommand per quoted number), so no number in the text or the tables is typed by hand.

A cell is (design set, stage, network). For each cell, over its trainings (seeds 0 to 2): the
closure width on the standard test (the report half of the 80k-event held-out runs) and on the
strict test (400k events per held-out point), as mean and standard deviation, and the width of
uniform weights on both tests.

Stages use the code names of the scripts (E is Sherpa, F is Herwig, MIX the mixture). The macro
names use the generator names instead, so the paper's own stage labels never enter them:
    \\Acc<Set><Stage><Net><Test>[Sd], \\Acc<Set><Stage>Uni<Test>, \\Acc<Set><Stage><Net>N
e.g. \\AccAugBFacStrict, \\AccAugBFacStrictSd, \\AccAugBUniStd, \\AccOrigHerwigCatN.

A run that trained its full budget under the old early-stopping rule (it was never cut, so the
result is the one of a full-schedule run) stands in for its full-schedule rerun.
Usage: python make_numbers.py [results dir]   (LC_DIR: the folder of the learning-curve widths)"""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results')
FULL_BUDGET = 324000
SETS = [  # (macro name, file prefix, fallback prefix, stages)
    ('Grid', 'p2gridfull', 'p2', ['A', 'B', 'C']),
    ('Aug', 'p2augfull', 'p2aug', ['A', 'B', 'C']),
    ('Orig', 'p2origfull', None, ['E', 'F', 'MIX']),
    ('Dense', 'p2v3full', None, ['E', 'F', 'MIX']),
]
STAGE_NAME = {'A': 'A', 'B': 'B', 'C': 'C', 'E': 'Sherpa', 'F': 'Herwig', 'MIX': 'Mix'}
NET_NAME = {'fac': 'Fac', 'cat': 'Cat'}
# The factorized mixture of the 720 runs was trained by train_mixgeo.sbatch (the learning-curve runs,
# with the settings of train_p2.sbatch and seed offsets 0, 1000, 2000), so its training result is the
# standard test of that cell
ALIAS = {('p2v3full', 'MIX', 'fac', k): f'stage_mixture_{t}.json'
         for k, t in enumerate(['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2'])}


def load(p):
    d = json.load(open(p))
    return d if 'master' in d else list(d.values())[0]


def pick(pre, fallback, S, net, k, strict):
    """The result file of one training, or None. strict picks the 400k-event score of it."""
    tag = 'strict' if strict else ''
    if not strict and (pre, S, net, k) in ALIAS:
        p = f'{D}/{ALIAS[(pre, S, net, k)]}'
        return p if os.path.exists(p) else None
    p = f'{D}/{pre}{tag}_{S}_{net}_s{k}.json'
    if os.path.exists(p):
        return p
    if fallback:
        q = f'{D}/{fallback}_{S}_{net}_s{k}.json'
        if os.path.exists(q):
            tr = load(q).get('stop_trajectory') or []
            if tr and tr[-1][0] >= FULL_BUDGET:
                r = f'{D}/{fallback}{tag}_{S}_{net}_s{k}.json'
                return r if os.path.exists(r) else None
    return None


def uniform(d):
    return float(np.mean(list(d['width_uniform'].values())))


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))


numbers, macros = {}, []


def macro(name, value, digits=3):
    if value is not None:
        macros.append(f'\\newcommand{{\\{name}}}{{{value:.{digits}f}}}')


OBS_MACRO = {'1_minus_thrust': 'Thrust', 'mult_total': 'Mult', 'B_total': 'Btot', 'rho_heavy': 'RhoH',
             'nbaryon': 'Nb', 'strange': 'Strange'}
for sname, pre, fb, stages in SETS:
    for S in stages:
        cell_uni = {'std': [], 'strict': []}
        # the widths observable by observable of the factorized network (Table perobs), mean over trainings
        for test, strict in (('std', False), ('strict', True)):
            po = [load(p)['per_obs'] for p in (pick(pre, fb, S, 'fac', k, strict) for k in range(3)) if p]
            if po:
                m = {o: float(np.mean([x[o] for x in po if o in x])) for o in OBS_MACRO if any(o in x for x in po)}
                numbers.setdefault('per_obs', {}).setdefault(sname, {}).setdefault(S, {})[test] = m
                for o, v in m.items():
                    macro(f'Obs{sname}{STAGE_NAME[S]}{OBS_MACRO[o]}{"Std" if test == "std" else "Strict"}', v, 2)
        for net in ('fac', 'cat'):
            res = {}
            for test, strict in (('std', False), ('strict', True)):
                w, u = [], []
                for k in range(3):
                    p = pick(pre, fb, S, net, k, strict)
                    if p:
                        d = load(p); w.append(d['master']); u.append(uniform(d))
                res[test] = stats(w); cell_uni[test] += u
            if res['std'] or res['strict']:
                numbers.setdefault(sname, {}).setdefault(S, {})[net] = res
                base = f'Acc{sname}{STAGE_NAME[S]}{NET_NAME[net]}'
                for test, tname in (('std', 'Std'), ('strict', 'Strict')):
                    if res[test]:
                        macro(base + tname, res[test]['mean'])
                        macro(base + tname + 'Sd', res[test]['sd'])
                macros.append(f'\\newcommand{{\\{base}N}}{{{(res["std"] or res["strict"])["n"]}}}')
        for test, tname in (('std', 'Std'), ('strict', 'Strict')):
            st = stats(cell_uni[test])
            if st:
                numbers[sname][S][f'uniform_{test}'] = st['mean']
                macro(f'Acc{sname}{STAGE_NAME[S]}Uni{tname}', st['mean'], 1)

# ---- the range over the final models of the paper (Table accuracy): Stages A to C on the densified
# designs, the Sherpa and Herwig stages and the mixture on the 720 runs ----------------------------
FINAL = [('Aug', 'A'), ('Aug', 'B'), ('Aug', 'C'), ('Dense', 'E'), ('Dense', 'F'), ('Dense', 'MIX')]
rng_final = {}
for net, tname in (('fac', 'Fac'), ('cat', 'Cat')):
    for test, ttest in (('std', 'Std'), ('strict', 'Strict')):
        v = [numbers[s][S][net][test]['mean'] for s, S in FINAL
             if net in numbers.get(s, {}).get(S, {}) and numbers[s][S][net][test]]
        if len(v) == len(FINAL):
            rng_final[f'{net}_{test}'] = [min(v), max(v)]
            # two decimals from the three Table accuracy prints (1.095 gives 1.10, as a reader rounds it)
            two = lambda x: float(Decimal(f'{x:.3f}').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            macro(f'AccFinal{tname}{ttest}Min', two(min(v)), 2); macro(f'AccFinal{tname}{ttest}Max', two(max(v)), 2)
for test, ttest in (('std', 'Std'), ('strict', 'Strict')):
    v = [numbers[s][S][f'uniform_{test}'] for s, S in FINAL if f'uniform_{test}' in numbers.get(s, {}).get(S, {})]
    if len(v) == len(FINAL):
        rng_final[f'uniform_{test}'] = [min(v), max(v)]
        macro(f'AccFinalUni{ttest}Min', min(v), 1); macro(f'AccFinalUni{ttest}Max', max(v), 1)
numbers['final_range'] = rng_final

# ---- the choice of the training sample (learning curve of the mixture) --------------------------
# Every row is scored at the same sixteen held-out points of the paper (7200-7207 Sherpa, 8200-8207
# Herwig), on the standard test and on the strict test, so that rows with different run sets
# compare like with like. Rows: (macro name, runs, events per run, steps, members, source).
# A source is a list of learning-curve tags (widths_v2_<TAG>.json and widths_v2_<TAG>h400.json in
# output/learning_curve) or ('p2', prefix) for the three Phase 2 trainings of the 208-run set. The
# closure width grows with the size of the reference, so every row is scored against a reference of
# the size of the 720-run model's: '<TAG>@r415' reads the widths of rescore_v4.sbatch, which use 415 of
# the 1000 stored events of each of the 2768 runs (1 148 720 events, against 1 149 840).
LC = os.environ.get('LC_DIR', os.path.join(os.path.dirname(HERE), 'output', 'learning_curve'))
PAPER = [str(r) for r in list(range(7200, 7208)) + list(range(8200, 8208))]
META208 = os.path.join(os.path.dirname(HERE), 'data_stagePURE17', 'meta.json')
LC_ROWS = [
    ('Runs', 208, 80000, 324000, 4, ('p2', 'p2origfull')),
    ('RunsThree', 720, 80000, 324000, 4, ['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2']),
    ('RunsFour', 2768, 80000, 324000, 4, ['MIXGEOV4S9@r415']),
    ('EvTwenty', 720, 20000, 324000, 4, ['MIXGEOV3E20S9']),
    ('EvOneSixty', 720, 160000, 324000, 4, ['MIXGEOV3E160S9']),
    ('EvThreeTwenty', 720, 320000, 324000, 4, ['MIXGEOV3E320S9']),
    ('StepsThree', 720, 80000, 972000, 4, ['MIXGEOV3S27F']),
    ('StepsFour', 2768, 80000, 4320000, 4, ['MIXGEOV4L@r415']),
    ('Members', 720, 80000, 324000, 8, ['MIXGEOV3M8S9']),
]


def paper_width(per_run):
    """Mean width over the sixteen paper points and the observables, from {rid: {obs: width}}."""
    v = [w for r in PAPER if r in per_run for w in per_run[r].values()]
    return float(np.mean(v)) if len(v) == 16*len(next(iter(per_run.values()))) else None


def p2_per_run(d, held):
    """{rid: {obs: width}} from an r2_ladder result, whose points are keyed by their parameters."""
    by_theta = {str(tuple(h['theta'])): str(h['rid']) for h in held}
    return {by_theta[k]: v for k, v in d['per_point'].items() if k in by_theta}


lc = {}
held208 = json.load(open(META208))['held'] if os.path.exists(META208) else None
for name, runs, ev, steps, mem, src in LC_ROWS:
    std, strict, std48, strict48 = [], [], [], []
    if isinstance(src, tuple):
        for k in range(3):
            for test, acc in ((False, std), (True, strict)):
                p = pick(src[1], None, 'MIX', 'fac', k, test)
                if p and held208:
                    acc.append(paper_width(p2_per_run(load(p), held208)))
            # the same trainings scored at all 48 held-out points of the 720-run set (make_orig48.py)
            for test, acc48 in ((False, std48), (True, strict48)):
                p = pick(src[1] + '48', None, 'MIX', 'fac', k, test)
                if p:
                    acc48.append(load(p)['master'])
    else:
        for tag in src:
            tag, _, rs = tag.partition('@')
            for suffix, acc, acc48 in (('', std, std48), ('h400', strict, strict48)):
                f = f'{LC}/widths_v2_{tag}{suffix}{rs}.json'
                if os.path.exists(f):
                    d = json.load(open(f)); acc.append(paper_width(d['per_run'])); acc48.append(d['master'])
    row = dict(runs=runs, events=ev, steps=steps, members=mem, std=stats(std), strict=stats(strict),
               std48=stats(std48), strict48=stats(strict48))
    if name == 'RunsThree' and 'fac' in numbers.get('Dense', {}).get('MIX', {}):
        # the model of the paper: its row of Table level prints the numbers of Table accuracy, the same
        # three trainings scored by r2_ladder (the two scorings agree to 1e-5 per training, but the
        # mean of three sits on a rounding boundary, 1.15750)
        row['std48'] = numbers['Dense']['MIX']['fac']['std']; row['strict48'] = numbers['Dense']['MIX']['fac']['strict']
    if row['std'] or row['strict']:
        lc[name] = row
        for key, tname in (('std', 'Std'), ('strict', 'Strict'), ('std48', 'StdAll'), ('strict48', 'StrictAll')):
            if row[key]:
                macro(f'Lc{name}{tname}', row[key]['mean'])
                if row[key]['sd'] is not None:
                    macro(f'Lc{name}{tname}Sd', row[key]['sd'])
numbers['learning_curve'] = lc

# ---- does the learning band cover the error of the prediction? -------------------------------------
# wifi_coverage.py at the 48 held-out points of the strict test: the rms of the pulls of the five
# observable means without and with the learning band, and the size of the errors.
COV_TAGS = os.environ.get('COVERAGE_TAGS', 'MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2').split()
for tags, pre in ((COV_TAGS, 'Cov'), (['MIXGEOV3M8S9'], 'CovEight')):     # the final mixture, and eight members
    S = [json.load(open(f))['summary'] for f in (f'{LC}/wifi_coverage_{t}.json' for t in tags) if os.path.exists(f)]
    if not S:
        continue
    vals = lambda k: [v[k] for s in S for v in s.values()]
    cov = dict(tags=tags, n_trainings=len(S),
               z_without=[min(vals('rms_z_without')), max(vals('rms_z_without'))],
               z_with=[min(vals('rms_z_with')), max(vals('rms_z_with'))],
               rel_error_percent=[100*min(vals('rms_rel_error')), 100*max(vals('rms_rel_error'))])
    numbers['coverage' if pre == 'Cov' else 'coverage_eight'] = cov
    for key, tname, dg in (('z_without', 'ZWithout', 1), ('z_with', 'ZWith', 2), ('rel_error_percent', 'Err', 1)):
        macro(f'{pre}{tname}Min', cov[key][0], dg); macro(f'{pre}{tname}Max', cov[key][1], dg)

# ---- the three uncertainties of Sec. 4 (three_bands.py on the final mixture) ----------------------
# Bands to two significant figures, central values to the decimal place of the combined band's second
# significant figure. "Inside the window" means the DELPHI bins whose lower edge lies in [0.05, 1/3).
THREE_TAG = os.environ.get('THREE_TAG', 'MIXGEOV3S9')
OBS_NAME = {'tau_win': 'TauWin', 'pwin': 'Pwin', 'thrust': 'Thrust', 'B_total': 'Btot', 'rho_heavy': 'RhoH',
            'nch': 'Nch', 'nbaryon': 'Nb'}


def sig(v, n=2):
    """v to n significant figures, as text (powers of ten below 1e-4)."""
    if v == 0 or not np.isfinite(v):
        return '0'
    e = int(np.floor(np.log10(abs(v))))
    if e < -4:
        m = round(v/10**e, n - 1)
        if abs(m) >= 10:
            m, e = m/10, e + 1
        return f'\\ensuremath{{{m:.{n-1}f}\\times10^{{{e}}}}}'
    return f'{v:.{max(n - 1 - e, 0)}f}'


def textmacro(name, text):
    macros.append(f'\\newcommand{{\\{name}}}{{{text}}}')


f3 = f'{D}/three_bands_{THREE_TAG}'
if os.path.exists(f3 + '.json') and os.path.exists(f3 + '.npz'):
    d3 = json.load(open(f3 + '.json')); c3 = d3['central']; g3 = d3['generator_band']; z3 = np.load(f3 + '.npz')
    tb = dict(tag=THREE_TAG, table={})
    # The theory band is signed (three_bands.py, paper r8): the band above the prediction collects the upward
    # displacements of the variations and the band below the downward ones, with the three variations of
    # the dispersive model included. The combined band is signed through it.
    def updown(u, d):
        su, sd_ = sig(u), sig(d)
        return f'\\pm{su}' if su == sd_ else f'{{}}^{{+{su}}}_{{-{sd_}}}'
    for k, nm in OBS_NAME.items():
        comb = max(c3['combined_up'][k], c3['combined_down'][k])
        dec = max(1 - int(np.floor(np.log10(abs(comb)))), 0)
        row = dict(generator_only=c3['generator_only'][k], imposed=c3['imposed'][k], gen_before=g3['before'][k]['sd'],
                   gen_after=g3['after'][k]['sd'], theory_up=c3['theory_up'][k], theory_down=c3['theory_down'][k],
                   learning=c3['learning_band'][k], combined_up=c3['combined_up'][k], combined_down=c3['combined_down'][k],
                   theory_symmetric_no_dispersive=c3['theory_band'][k])
        tb['table'][k] = row
        textmacro(f'Tb{nm}GenOnly', f'{row["generator_only"]:.{dec}f}'); textmacro(f'Tb{nm}Imposed', f'{row["imposed"]:.{dec}f}')
        for key, col in (('gen_before', 'GenBefore'), ('gen_after', 'GenAfter'), ('learning', 'Learn')):
            textmacro(f'Tb{nm}{col}', sig(row[key]))
        textmacro(f'Tb{nm}Theory', updown(row['theory_up'], row['theory_down']))
        textmacro(f'Tb{nm}Comb', updown(row['combined_up'], row['combined_down']))
    lo = z3['lo']; win = (lo >= 0.05) & (lo < 1/3); above = lo >= 1/3; top = (lo >= 0.2 - 1e-9) & (lo < 1/3); low = win & ~top
    hc, hb, dat, err = z3['h_central'], z3['h_central_before'], z3['data'], z3['err']
    rng_pc = lambda a, h, m: (100*float(np.min(a[m]/h[m])), 100*float(np.max(a[m]/h[m])))
    ranges = dict(gen_before_win=rng_pc(z3['gen_hist_before'], hb, win), gen_after_win=rng_pc(z3['gen_hist'], hc, win),
                  gen_before_above=rng_pc(z3['gen_hist_before'], hb, above), gen_after_above=rng_pc(z3['gen_hist'], hc, above),
                  theory_win=rng_pc(z3['theory_hist'], hc, win), theory_above=rng_pc(z3['theory_hist'], hc, above),
                  learn_win=rng_pc(z3['learn_hist'], hc, win), learn_above=rng_pc(z3['learn_hist'], hc, above),
                  learn_before_win=rng_pc(z3['learn_hist_before'], hb, win),
                  theory_up_win=rng_pc(z3['theory_hist_up'], hc, win), theory_down_win=rng_pc(z3['theory_hist_down'], hc, win),
                  theory_up_top=rng_pc(z3['theory_hist_up'], hc, top), theory_down_top=rng_pc(z3['theory_hist_down'], hc, top),
                  theory_up_low=rng_pc(z3['theory_hist_up'], hc, low), theory_down_low=rng_pc(z3['theory_hist_down'], hc, low))
    tb['bin_ranges_percent'] = ranges
    for key, nm in (('gen_before_win', 'GenBeforeWin'), ('gen_after_win', 'GenAfterWin'), ('gen_before_above', 'GenBeforeAbove'),
                    ('gen_after_above', 'GenAfterAbove'), ('theory_win', 'TheoryWin'), ('theory_above', 'TheoryAbove'),
                    ('learn_win', 'LearnWin'), ('learn_above', 'LearnAbove'), ('learn_before_win', 'LearnBeforeWin'),
                    ('theory_up_win', 'TheoryUpWin'), ('theory_down_win', 'TheoryDownWin'), ('theory_up_top', 'TheoryUpTop'),
                    ('theory_down_top', 'TheoryDownTop'), ('theory_up_low', 'TheoryUpLow'), ('theory_down_low', 'TheoryDownLow')):
        textmacro(f'TbBin{nm}Min', sig(ranges[key][0])); textmacro(f'TbBin{nm}Max', sig(ranges[key][1]))
    rms = lambda m: float(np.sqrt(np.mean(m**2)))
    side = np.where(dat > hc, z3['comb_hist_up'], z3['comb_hist_down'])          # the band on the side of the data
    with_band = (hc - dat)/np.sqrt(err**2 + side**2)
    pulls_ = dict(imposed=rms(((hc - dat)/err)[win]), generators_only=rms(((hb - dat)/err)[win]), imposed_with_band=rms(with_band[win]),
                  above_max_with_band=float(np.max(np.abs(with_band[above]))),
                  top_min_with_band=float(np.min(np.abs(with_band[top]))), top_max_with_band=float(np.max(np.abs(with_band[top]))),
                  low_max_with_band=float(np.max(np.abs(with_band[low]))))
    tb['delphi_rms_pull'] = pulls_
    for key, nm in (('imposed', 'Imposed'), ('generators_only', 'GenOnly'), ('imposed_with_band', 'WithBand'), ('above_max_with_band', 'AboveMax'),
                    ('top_min_with_band', 'TopMin'), ('top_max_with_band', 'TopMax'), ('low_max_with_band', 'LowMax')):
        textmacro(f'TbPull{nm}', sig(pulls_[key]))
    deficit = 100*(1 - hc/dat)
    tb['top_deficit_percent'] = [float(deficit[top].min()), float(deficit[top].max())]
    textmacro('TbDeficitTopMin', sig(tb['top_deficit_percent'][0])); textmacro('TbDeficitTopMax', sig(tb['top_deficit_percent'][1]))
    # the bins below the window, which three_bands.py writes under _low with the same normalization: the
    # combined band there (the larger side, in percent of the prediction) and the largest pull of DELPHI
    if 'lo_low' in z3.files and len(z3['lo_low']):
        hl, dl, el = z3['h_central_low'], z3['data_low'], z3['err_low']
        band_below = 100*np.maximum(z3['comb_hist_up_low'], z3['comb_hist_down_low'])/hl
        pull_below = np.abs(hl - dl)/np.sqrt(el**2 + np.where(dl > hl, z3['comb_hist_up_low'], z3['comb_hist_down_low'])**2)
        tb['below_window'] = dict(lo=float(z3['lo_low'][0]), hi=float(z3['hi_low'][-1]),
                                  band_percent=[float(band_below.min()), float(band_below.max())],
                                  max_pull_with_band=float(pull_below.max()))
        textmacro('TbBelowBandMin', sig(band_below.min())); textmacro('TbBelowBandMax', sig(band_below.max()))
        textmacro('TbPullBelowMax', sig(pull_below.max()))
    # the displacement of the change of scheme in the top bins, and the largest upward displacement there
    vl = [str(l) for l in z3['theory_var_labels']]; vh = z3['theory_var_hist']
    scheme = -100*vh[vl.index("(1.0, 1.0, 'modR', 1.0)")][top]/hc[top]
    up_max = 100*np.max(np.maximum(vh[:, top], 0)/hc[top])
    tb['top_scheme_down_percent'] = [float(scheme.min()), float(scheme.max())]; tb['top_largest_up_percent'] = float(up_max)
    textmacro('TbSchemeTopMin', sig(scheme.min())); textmacro('TbSchemeTopMax', sig(scheme.max())); textmacro('TbUpTopMax', sig(up_max, 1))
    # each generator alone at its default tune (xi = 0, 1) and with the calculation imposed (theory_variations.py)
    ftv = f'{D}/theory_variations_MIX.json'
    if os.path.exists(ftv):
        tv = json.load(open(ftv)); assert np.allclose(np.asarray(tv['lo']), lo)
        dv_ = np.asarray(tv['data'])
        g0, g1 = np.asarray(tv['xi0']['h_gen_only'])/dv_, np.asarray(tv['xi1']['h_gen_only'])/dv_
        i0, i1 = np.asarray(tv['xi0']['h_imposed'])/dv_, np.asarray(tv['xi1']['h_imposed'])/dv_
        xs = dict(sherpa_alone_win=100*float(np.max(np.abs(g0[win] - 1))), herwig_alone_win=100*float(np.max(np.abs(g1[win] - 1))),
                  imposed_apart_win=100*float(np.max(np.abs(i0[win] - i1[win]))),
                  imposed_low=100*float(max(np.max(np.abs(i0[low] - 1)), np.max(np.abs(i1[low] - 1)))))
        tb['single_generators_percent'] = xs
        for key, nm in (('sherpa_alone_win', 'SherpaAloneWin'), ('herwig_alone_win', 'HerwigAloneWin'),
                        ('imposed_apart_win', 'ImposedApartWin'), ('imposed_low', 'ImposedLow')):
            textmacro(f'Tb{nm}', sig(xs[key], 1 if xs[key] < 9.5 else 2))
    # the calculation against DELPHI and ALEPH in the twelve bins wholly inside the window (calc_vs_delphi.py)
    fcd = f'{D}/calc_vs_delphi_{THREE_TAG}.json'
    if os.path.exists(fcd):
        cd = json.load(open(fcd)); clo = np.asarray(cd['lo']); ct = clo >= 0.2 - 1e-9
        cvd = dict(imposed_vs_calc_max=100*float(np.max(np.abs(np.asarray(cd['imposed_over_calc']) - 1))),
                   calc_below_top=[100*float(np.min(1 - np.asarray(cd['calc_over_delphi'])[ct])), 100*float(np.max(1 - np.asarray(cd['calc_over_delphi'])[ct]))],
                   aleph_below_top=[100*float(np.min(1 - np.asarray(cd['aleph_over_delphi'])[ct])), 100*float(np.max(1 - np.asarray(cd['aleph_over_delphi'])[ct]))],
                   aleph_err_top=[100*float(np.min(np.asarray(cd['aleph_rel_err'])[ct])), 100*float(np.max(np.asarray(cd['aleph_rel_err'])[ct]))])
        tb['calc_vs_delphi'] = cvd
        textmacro('TbImposedVsCalcMax', sig(cvd['imposed_vs_calc_max'], 1))
        for key, nm in (('calc_below_top', 'CalcBelowTop'), ('aleph_below_top', 'AlephBelowTop'), ('aleph_err_top', 'AlephErrTop')):
            textmacro(f'Tb{nm}Min', sig(cvd[key][0], 1 if cvd[key][0] < 9.5 else 2)); textmacro(f'Tb{nm}Max', sig(cvd[key][1], 1 if cvd[key][1] < 9.5 else 2))
    # the published comparisons of the captions: OPAL <1-T> and the L3 charged multiplicity, against the band on their side
    OPAL_THRUST, L3_NCH = 0.06671, 18.63
    cside = lambda k, x: c3['combined_up'][k] if x > c3['imposed'][k] else c3['combined_down'][k]
    tb['opal_pull'] = (OPAL_THRUST - c3['imposed']['thrust'])/cside('thrust', OPAL_THRUST)
    tb['l3_in_gen_bands'] = (L3_NCH - c3['imposed']['nch'])/g3['after']['nch']['sd']
    textmacro('TbOpalPull', sig(tb['opal_pull'])); textmacro('TbLThreeBands', sig(tb['l3_in_gen_bands']))
    tb['theory_window_mean_percent'] = [100*c3['theory_up']['tau_win']/c3['imposed']['tau_win'], 100*c3['theory_down']['tau_win']/c3['imposed']['tau_win']]
    textmacro('TbTheoryTauWinUpPercent', sig(tb['theory_window_mean_percent'][0])); textmacro('TbTheoryTauWinDownPercent', sig(tb['theory_window_mean_percent'][1]))
    textmacro('CalcAlphas', f'{d3["calc"]["alpha_s"]:.4f}'); textmacro('CalcAlphaZero', f'{d3["calc"]["alpha_0"]:.2f}')
    textmacro('TbNodes', str(d3['nodes']))
    # the statistical error of the prediction from the finite reference sample (in every combined band)
    if 'stat_hist' in z3.files:
        st = dict(win=rng_pc(z3['stat_hist'], hc, win), top=rng_pc(z3['stat_hist'], hc, top),
                  means_max=100*max(c3['statistical'][k]/abs(c3['imposed'][k]) for k in OBS_NAME),
                  means_min=100*min(c3['statistical'][k]/abs(c3['imposed'][k]) for k in OBS_NAME))
        tb['statistical_percent'] = st
        textmacro('TbBinStatWinMin', sig(st['win'][0])); textmacro('TbBinStatWinMax', sig(st['win'][1]))
        textmacro('TbStatMeansMin', sig(st['means_min'])); textmacro('TbStatMeansMax', sig(st['means_max']))
    # inside the window: in how many bins the larger side of the theory band exceeds the generator band of
    # the generators alone, and how much imposing the calculation changes the window fraction
    th_big = np.maximum(z3['theory_hist_up'], z3['theory_hist_down'])/hc; gb = z3['gen_hist_before']/hb
    tb['theory_wider_bins'] = [int(np.sum(th_big[win] > gb[win])), int(win.sum())]
    textmacro('TbTheoryWiderBins', str(tb['theory_wider_bins'][0])); textmacro('TbWinBins', str(tb['theory_wider_bins'][1]))
    tb['pwin_change_percent'] = 100*abs(c3['imposed']['pwin'] - c3['generator_only']['pwin'])/c3['generator_only']['pwin']
    textmacro('TbPwinChangePercent', sig(tb['pwin_change_percent'], 1))
    # the box mean against the prediction at the default tunes, in units of the generator band
    off = {k: (g3['after'][k]['mean'] - c3['imposed'][k])/g3['after'][k]['sd'] for k in ('pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')}
    tb['box_mean_offset_in_bands'] = off
    shp = [off[k] for k in ('pwin', 'thrust', 'B_total', 'rho_heavy')]; prt = [off[k] for k in ('nch', 'nbaryon')]
    textmacro('TbBoxOffsetShapesMin', sig(min(shp), 2)); textmacro('TbBoxOffsetShapesMax', sig(max(shp), 2))
    textmacro('TbBoxOffsetPartMin', sig(min(prt), 1)); textmacro('TbBoxOffsetPartMax', sig(max(prt), 1))
    # the spread of the three sets of members at the central point against the learning band
    tr = [f'{D}/three_bands_{THREE_TAG}{t}.json' for t in ('', 's1', 's2')]
    if all(os.path.exists(f) for f in tr):
        cs = [json.load(open(f))['central'] for f in tr]
        assert all(np.allclose(cc['theta'], c3['theta']) for cc in cs)
        ratio = {k: float(np.std([cc['imposed'][k] for cc in cs], ddof=1)/c3['learning_band'][k]) for k in OBS_NAME}
        tb['training_spread_over_learning_band'] = ratio
        textmacro('TbTrainSpreadMin', sig(min(ratio.values()), 1)); textmacro('TbTrainSpreadMax', sig(max(ratio.values()), 1))
    # band_checks.py: the part of the generator band from the mixing fraction, and the largest weight
    fbc = f'{D}/band_checks_{THREE_TAG}.json'
    if os.path.exists(fbc):
        bc = json.load(open(fbc)); assert np.allclose(bc['central'], c3['theta'])
        means = ('pwin', 'thrust', 'B_total', 'rho_heavy', 'nch', 'nbaryon')
        xo = {k: bc['xi_only']['after']['sd'][k]/g3['after'][k]['sd'] for k in means}
        xf = {k: bc['xi_fixed']['after']['sd'][k]/g3['after'][k]['sd'] for k in means}
        tb['xi_part_of_generator_band'] = dict(xi_only=xo, xi_fixed=xf)
        textmacro('TbXiOnlyMin', sig(min(xo.values()), 1)); textmacro('TbXiOnlyMax', sig(max(xo.values()), 1))
        textmacro('TbXiFixedMin', sig(min(xf.values()), 2)); textmacro('TbXiFixedMax', sig(max(xf.values()), 2))
        mw = dict(sherpa=bc['max_weight_over_mean_xi0']['imposed'], middle=bc['stat_imposed']['max_weight_over_mean'],
                  herwig=bc['max_weight_over_mean_xi1']['imposed'])
        tb['largest_weight_over_mean'] = mw
        textmacro('TbMaxWeightSherpa', sig(mw['sherpa'])); textmacro('TbMaxWeightMid', sig(mw['middle'])); textmacro('TbMaxWeightHerwig', sig(mw['herwig']))
    numbers['three_bands'] = tb

# ---- the learning band: the fitted basis weights of the three trainings of the final mixture -----
WIFI_TAGS = os.environ.get('WIFI_TAGS', 'MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2').split()
ORD = ['One', 'Two', 'Three']
wifi = {}
for i, t in enumerate(WIFI_TAGS):
    ff, fd = f'{D}/wifi_{t}_fit.json', f'{D}/wifi_{t}_diag.json'
    if not os.path.exists(ff):
        continue
    fit = json.load(open(ff)); row = dict(newton=fit['newton_iterations'], logw_median=fit['logw_sigma_median'],
                                          logw_p95=fit['logw_sigma_p95'], w=fit['w'], w_err=fit['w_err'])
    if os.path.exists(fd):
        dg = json.load(open(fd))
        row.update(loss_mean=dg['published']['held_loss'], width_mean=dg['published']['closure'],
                   loss_fit=dg['wifi']['held_loss'], width_fit=dg['wifi']['closure'])
    wifi[t] = row
    textmacro(f'WifiNewton{ORD[i]}', str(row['newton']))
    textmacro(f'WifiLogwMedian{ORD[i]}', sig(100*row['logw_median'])); textmacro(f'WifiLogwNinetyFive{ORD[i]}', sig(100*row['logw_p95']))
    for key, nm, dg_ in (('loss_mean', 'LossMean', 3), ('width_mean', 'WidthMean', 2), ('loss_fit', 'LossFit', 3), ('width_fit', 'WidthFit', 2)):
        if key in row:
            textmacro(f'Wifi{nm}{ORD[i]}', f'{row[key]:.{dg_}f}')
numbers['wifi'] = wifi

# ---- the map of Stage C (Fig. map): the final densified Stage C model at the twelve fresh runs ----
# eval_valid.py with VALID_HALF=report, the paper's closure test (it agrees with the blind test of
# the training code to the third decimal). The baryon offset is |z| of the mean baryon count; the
# runs are split by distance from the middle of the box (largest parameter offset in half-widths)
# into the five nearest and the seven farther out.
fmap = f'{D}/' + os.environ.get('MAP_FILE', 'valid_map_C_P2augfullfac0_report.json')
if os.path.exists(fmap):
    vm = json.load(open(fmap))
    pts = [r for k, r in vm.items() if k != 'summary' and r.get('kind') == 'valid']
    NONB = ['1_minus_thrust', 'B_total', 'rho_heavy', 'mult_total', 'x_p', 'strange', 'joint_thrust_mult']
    means = {o: float(np.mean([r[o] for r in pts])) for o in NONB + ['nbaryon']}
    per_run_nonb = [r[o] for r in pts for o in NONB]
    nb = [r['nbaryon'] for r in pts]
    BOX = [(0.120, 0.008), (0.475, 0.175), (1.30, 0.50)]
    dist = [max(abs((r['theta'][i] - BOX[i][0])/BOX[i][1]) for i in range(3)) for r in pts]
    z = [abs(r['nbaryon_z']) for r in pts]
    order = np.argsort(dist)
    mp = dict(file=fmap, n=len(pts), mean_per_obs=means,
              nonbaryon_mean_range=[min(means[o] for o in NONB), max(means[o] for o in NONB)],
              nonbaryon_run_range=[min(per_run_nonb), max(per_run_nonb)],
              baryon_run_range=[min(nb), max(nb)], baryon_mean=means['nbaryon'],
              baryon_z_near=float(np.mean([z[i] for i in order[:5]])), baryon_z_far=float(np.mean([z[i] for i in order[5:]])),
              baryon_z_over_two=int(sum(v > 2 for v in z)))
    numbers['map'] = mp
    macro('MapNonbMeanMin', mp['nonbaryon_mean_range'][0], 2); macro('MapNonbMeanMax', mp['nonbaryon_mean_range'][1], 2)
    macro('MapNonbRunMin', mp['nonbaryon_run_range'][0], 1); macro('MapNonbRunMax', mp['nonbaryon_run_range'][1], 1)
    macro('MapBaryonRunMin', mp['baryon_run_range'][0], 1); macro('MapBaryonRunMax', mp['baryon_run_range'][1], 1)
    macro('MapBaryonMean', mp['baryon_mean'], 2)
    macro('MapBaryonZNear', mp['baryon_z_near'], 1); macro('MapBaryonZFar', mp['baryon_z_far'], 1)
    textmacro('MapBaryonZOverTwo', str(mp['baryon_z_over_two']))

# ---- App. B: two halves of one fresh run, calibration constants, selected steps ------------------
# The calibration check (two disjoint halves of the same held-out run, no network) depends only on
# the data, so one training per stage gives it.
for S, (s, pre, fb) in {'A': ('Aug', 'p2augfull', 'p2aug'), 'B': ('Aug', 'p2augfull', 'p2aug'),
                        'C': ('Aug', 'p2augfull', 'p2aug')}.items():
    p = pick(pre, fb, S, 'fac', 0, False)
    if p:
        cc = load(p)['calibration_check']
        macro(f'Calib{S}Min', min(cc.values()), 2); macro(f'Calib{S}Max', max(cc.values()), 2)
kap, steps = [], {}
for s, S in FINAL:
    pre, fb = {'Aug': ('p2augfull', 'p2aug'), 'Dense': ('p2v3full', None)}[s]
    for net in ('fac', 'cat'):
        for k in range(3):
            p = pick(pre, fb, S, net, k, False)
            if p:
                d = load(p); kap.append(d['temperature'])
                if net == 'fac':
                    steps.setdefault(S, []).append(d['best_step'])
if kap:
    macro('KappaMin', min(kap), 3); macro('KappaMax', max(kap), 3)
if steps:
    numbers['selected_steps'] = steps
    textmacro('StepAMin', f'{min(steps["A"])/1e5:.1f}'); textmacro('StepAMax', f'{max(steps["A"])/1e5:.1f}')
    textmacro('StepOtherMin', f'{min(v for S, vv in steps.items() if S != "A" for v in vv)/1e5:.1f}')
# the mixture at its Sherpa points (xi = 0) and its Herwig points (xi = 1), mean over points and trainings
gen_split = {}
for test, strict in (('std', False), ('strict', True)):
    sh, hw = [], []
    for k in range(3):
        p = pick('p2v3full', None, 'MIX', 'fac', k, strict)
        if not p:
            continue
        for th, po in load(p)['per_point'].items():
            xi = float(th.strip('()').split(',')[-1])
            (sh if xi < 0.5 else hw).append(float(np.mean(list(po.values()))))
    if sh and hw:
        gen_split[test] = dict(sherpa=float(np.mean(sh)), herwig=float(np.mean(hw)))
        t_ = 'Std' if test == 'std' else 'Strict'
        macro(f'MixSherpaPoints{t_}', np.mean(sh), 2); macro(f'MixHerwigPoints{t_}', np.mean(hw), 2)
numbers['mixture_by_generator'] = gen_split
# where the gain of the fitted basis weights in the loss comes from (wifi_gensplit.py, training 1)
fgs = f'{D}/wifi_gensplit_{WIFI_TAGS[0]}.json'
if os.path.exists(fgs):
    gs = json.load(open(fgs)); numbers['wifi_gensplit'] = gs
    g = gs['gain']
    macro('WifiGainTotal', -g['loss'], 2); macro('WifiGainOther', -g['den_other'], 2)
    macro('WifiLossNum', g['num'], 2); macro('WifiLossOwn', g['den_same'], 2)
    macro('WifiZOther', gs['published']['mean_z_other'], 1)

# ---- single networks against the mean of members (single_members.py, Stage C, first training) ----
fsm = f'{D}/single_members_C_P2augfullfac0.json'
if os.path.exists(fsm):
    sm = json.load(open(fsm)); numbers['single_members'] = sm
    ws = [v['width'] for k, v in sm.items() if k.startswith('member_')]
    macro('SingleMin', min(ws), 2); macro('SingleMax', max(ws), 2); macro('SingleMean', sm['mean_of_members']['width'], 2)
# ---- the three trainings at the central point of Sec. 4: the spread of <1-T> with the calculation imposed
th3 = [json.load(open(f))['central']['imposed']['thrust'] for f in
       (f'{D}/three_bands_{t}.json' for t in WIFI_TAGS) if os.path.exists(f)]
if len(th3) == 3:
    numbers['seed_thrust'] = th3
    textmacro('SeedThrustSd', sig(float(np.std(th3, ddof=1))))
    # the same in units of 10^-4, beside the learning band of <1-T> in the same unit
    textmacro('SeedThrustSdEFour', f'{1e4*float(np.std(th3, ddof=1)):.1f}')
    if 'three_bands' in numbers:
        textmacro('TbThrustLearnEFour', f'{1e4*numbers["three_bands"]["table"]["thrust"]["learning"]:.1f}')
# ---- the wide coupling scan (App. B): three references of equal size, mean width per held point ---
def _pt(f):
    """{coupling: mean width over the observables} of one stage-W result"""
    return {float(k.strip('()').split(',')[0]): float(np.mean(list(v.values()))) for k, v in load(f)['per_point'].items()}
# the references hold 6.3e4 events, so the networks train for 1.8e4 steps (DESIGN_TAG wb18), which
# shows each reference event as often as the 3.24e5 steps of the mixture show its 1.15e6 events
fs, fp = f'{D}/p2wb18single_W_fac_s0.json', f'{D}/p2wb18pool_W_fac_s0.json'
fl = [f'{D}/p2wb18local{r}_W_fac_s0.json' for r in range(9100, 9106)]
if os.path.exists(fs) and os.path.exists(fp) and all(os.path.exists(f) for f in fl):
    single, pool = _pt(fs), _pt(fp)
    local = {a: w for f in fl for a, w in _pt(f).items()}
    worst = min(single)                                   # the target farthest below the run of the single reference
    five = [w for a, w in single.items() if a != worst]
    numbers['widebox'] = dict(single=single, pool=pool, local=local)
    macro('WideSingleMean', np.mean(list(single.values())), 2); macro('WidePoolMean', np.mean(list(pool.values())), 2)
    macro('WideLocalMean', np.mean(list(local.values())), 2)
    macro('WideSingleFiveMin', min(five), 2); macro('WideSingleFiveMax', max(five), 2); macro('WideSingleWorst', single[worst], 2)
    macro('WidePoolMin', min(pool.values()), 2); macro('WidePoolMax', max(pool.values()), 2)
    macro('WideLocalMin', min(local.values()), 2); macro('WideLocalMax', max(local.values()), 2)

# ---- the concatenation network truncated to its leading singular components (App. C) --------------
# r2_ladder.py with LADDER_TRUNCATE=1 on the first concatenation training of each model: the smallest
# number of components whose closure width comes within 0.02 of that of the full network
svd = {}
for S, nm in (('A', 'A'), ('B', 'B'), ('C', 'C'), ('E', 'Sherpa'), ('MIX', 'Mix')):
    f = f'{D}/p2trunc_{S}_cat_s0.json'
    if os.path.exists(f) and 'truncation' in load(f):
        tr_ = load(f)['truncation']; svd[S] = tr_
        if tr_['rank_within_0_02'] is not None:
            textmacro(f'SvdRank{nm}', str(tr_['rank_within_0_02']))
numbers['concat_truncation'] = svd

# ---- the input keeps the 100 most energetic particles: the fraction of events truncated ------------
ftr = f'{D}/truncation_fraction.json'          # written by trunc_frac.py over the final training runs
if os.path.exists(ftr):
    tr = json.load(open(ftr)); numbers['truncation'] = tr
    fr = [100*v['fraction'] for v in tr.values()]
    textmacro('TruncMinPercent', sig(min(fr))); textmacro('TruncMaxPercent', sig(max(fr)))
    textmacro('TruncMaxMult', str(max(v['max'] for v in tr.values())))

# ---- the energy-fraction spectrum: its closure width between two halves of one run (xp_null.py) ----
# its uncertainty per bin counts the events that contribute to the bin, so the expectation is below one
fxp = f'{D}/xp_null.json'
if os.path.exists(fxp):
    xp = json.load(open(fxp)); numbers['xp_null'] = xp
    macro('XpNullMean', xp['mean'], 2)

# ---- the Gaussian toy with the paper's network and training (toy_final.py, Sec. toy) ---------------
# three trainings at K=48 and one at each K of the scan; the figure and the slopes are the first training's
toys = {}
for fn in os.listdir(D):
    if fn.startswith('toy_final_K') and fn.endswith('.json'):
        t_ = json.load(open(os.path.join(D, fn))); toys[(int(t_['K']), int(t_['seed']))] = t_
if all((48, s) in toys for s in range(3)):
    w48 = [toys[(48, s)]['width'] for s in range(3)]; t0 = toys[(48, 0)]
    kap = sorted(set(toys[(48, s)]['kappa'] for s in range(3)))
    scan = {k: v['width'] for (k, s), v in toys.items() if s == 0}
    numbers['toy'] = dict(width48=w48, kappa48=kap, scan=scan, slope=t0['slope'], corr=t0['corr'],
                          dslope=t0['dslope'], dcorr=t0['dcorr'], uniform=t0['uniform'])
    macro('ToyWidthMean', float(np.mean(w48)), 3); macro('ToyWidthSd', float(np.std(w48, ddof=1)), 3)
    macro('ToyUniform', t0['uniform'], 1)
    macro('ToySlope', t0['slope'], 3); macro('ToyCorr', t0['corr'], 4)
    macro('ToyDSlope', t0['dslope'], 3); macro('ToyDCorr', t0['dcorr'], 4)
    if len(kap) == 1:
        macro('ToyKappa', kap[0], 2)
    if 1 in scan:
        macro('ToyWidthKOne', scan[1], 2); macro('ToyQShortKOne', 100*(1 - toys[(1, 0)]['coef_q_ratio_at_021']), 0)
    rest = [v for k, v in scan.items() if k >= 2]
    macro('ToyWidthScanMin', min(rest), 2); macro('ToyWidthScanMax', max(rest), 2)

# ---- the cost of a fit step (Sec. 3.6, Fig. cost): bench_cost_final.py on the final mixture ---------
# the network part of a step (the weights of the reference events and their derivatives) on one CPU
# process with four threads and on one H200, the rest of the step (the maximum-entropy weight and the
# chi^2, the same for every network) on the CPU, and a profile fit of COST_STEPS steps
COST_STEPS = 6000
fc, fg = f'{D}/bench_final_cpu_4.json', f'{D}/bench_final_cuda_12.json'
if os.path.exists(fc) and os.path.exists(fg):
    bc, bg = json.load(open(fc)), json.load(open(fg))
    nc, ng, rest = bc['n_network_step_seconds'], bg['n_network_step_seconds'], bc['b_tilt_and_chi2_seconds']
    if 'concatenation' in nc and 'concatenation' in ng:
        fh = {k: COST_STEPS*(v + rest)/3600 for k, v in nc.items()}
        numbers['cost'] = dict(cpu=nc, gpu=ng, rest=rest, steps=COST_STEPS, fit_hours=fh, cpu_name=bc.get('cpu'),
                               gpu_name=bg.get('gpu'), gradcheck_ok=bg.get('gradcheck_ok'))
        textmacro('CostFacCpu', sig(nc['factorized']) + '\\,s'); textmacro('CostRestCpu', sig(rest) + '\\,s')
        textmacro('CostLateCpu', sig(nc['late']) + '\\,s'); textmacro('CostCatCpuMin', f'{nc["concatenation"]/60:.0f}')
        textmacro('CostFacGpu', sig(1e3*ng['factorized']) + '\\,ms'); textmacro('CostLateGpu', sig(ng['late']) + '\\,s')
        textmacro('CostCatGpu', sig(ng['concatenation']) + '\\,s')
        textmacro('CostFitFacHours', sig(fh['factorized'])); textmacro('CostFitLateDays', sig(fh['late']/24))
        textmacro('CostFitFacMin', f'{60*fh["factorized"]:.0f}')
        textmacro('CostFitCatDays', f'{fh["concatenation"]/24:.0f}')

json.dump(numbers, open(os.path.join(HERE, 'numbers.json'), 'w'), indent=1)
open(os.path.join(HERE, 'numbers.tex'), 'w').write(
    '% written by phase2/make_numbers.py, do not edit\n' + '\n'.join(macros) + '\n')
print(f'{len(macros)} macros -> numbers.tex, blocks:', ', '.join(numbers))
