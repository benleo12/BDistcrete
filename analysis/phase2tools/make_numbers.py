"""Every accuracy number of the paper, written by script: numbers.json and numbers.tex next to this
script (one \\newcommand per quoted number), so no number in the text or the tables is typed by hand.

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

From analysis/, this writes the paper's numbers.tex:
    python phase2tools/make_numbers.py output
The argument, or INDIR, is the folder of the result files (default analysis/output); OUTDIR the folder of
numbers.json and numbers.tex (default: next to this script). The learning-curve widths and the coverage files are
read from the same folder, or from LC_DIR if it is set. SUFFIX=_lin reads the results of the linear mixture
(three_bands_<tag>_lin, theory_variations_MIX_lin, calc_vs_delphi_<tag>_lin, band_checks_<tag>_lin,
envelope_test_MIXGEOV3S9_<box>_lin, bench_final_{cpu_4,cuda_12}_lin), checks that each of them holds mix = linear,
and writes numbers_lin.json and numbers_lin.tex; the macros of the quadrature in xi and of tune_xifull are then
left out, and the full-size boxes of each generator alone add Env{Sherpa,Herwig}Full{Sd,BinMin,BinMax,MeanMin,MeanMax,
Corr,Shift} (D1: every prediction is linear in xi, so the full box's bending lies within each generator). The linear
run also writes, from the propagation's own results, \\CalcShift (the power correction a_tau P of App. C, from
calc_bins_nlo_point.json), \\GaussNullMean, \\GaussNullSd and \\GaussShiftMean (the Gaussian reference of the caption
of Table 1, gauss_ref_lin.json), and from crosscheck_seeds_summary.json of the PFN job \\PfnParamsEf and
\\PfnParamsOurs (the parameter counts of App. A.2) and \\PfnAucEf, \\PfnAucPlain, \\PfnAucOursRelu, \\PfnAucOurs and
\\PfnAucSdMax (its AUCs; the summary must hold torchK1silu, our network as the paper builds it). Since round B21 it
also writes the numbers of the window under D3, where every bin is the window fraction P_win, left to the generators,
times the shape inside the window, which the calculation fixes (\\TbShape*, \\TbPwinSherpa, \\TbPwinHerwig,
\\TbPwin*Percent, \\TbWinFrac*, \\TbPullTheoryTop*, \\TbDeficitLow*, \\TbCalcBelowEdge), the envelope's bound on the
correlations over both distributions (\\EnvTuneCorr), \\EnvAfterWin for each box apart (\\EnvAfterWinTune,
\\EnvAfterWinFull*), and each signed range of how much narrower the linear bands are as unsigned parts
(\\Env*Wider, \\Env*Narrower, \\Env*WiderN); and its two-figure numbers no longer gain a figure where the rounding carries
(9.95 prints 10, not 10.0). The comparisons with DELPHI,
OPAL and L3 use the predictions at the hadron level of each measurement (D8, keys *_conv) when the files hold them,
and the uncorrected ones otherwise; the pulls against OPAL and L3 (TbOpalPull, TbLThreePull) add the measurement's
error in quadrature to the band (E12).

With SUFFIX=_lin the inputs must be those of the full rerun (E11): no result made at a reduced size (a key
hypercube_size, nodes_evaluated, draws_generated or n_events_test, or fewer than the 2000 points or draws of D7, or
fewer than 1000 further draws on the full envelope box), and each benchmark file of the device and threads its name
says. Every input it reads must exist (no family of macros is lost in silence): the propagation's outputs, every
training of Table 1 and of the origfull cells, the learning-curve widths, the coverage, wifi, map, single-member,
wide-scan, truncated-network, truncation, x_p and toy files. The grid cells are not read (the rerun does not remake
them), and the CPU benchmark file is required: without the GPU file only its four macros (\\CostFacGpu, \\CostLateGpu,
\\CostCatGpu, \\CostWCatGpu) are not written, and a GPU file without the CPU one is refused. The scoring side's check
INDIR/scoring_check.json (I4, written by collect_lmx; INDIR is RUN/output) must exist, list files and belong to the
start mark INDIR/.lmx_rerun_start; every input must be listed there with the md5 it has, except the propagation's own
results in INDIR (the *_lin files, calc_bins_nlo_point.json and crosscheck_seeds_summary.json, which the first
submission's park also moves aside) and the data table data_stagePURE17/meta.json; and no input may resolve into the
published maxent_code (except that data table in its photon-free _nb folder) or into INDIR/published_copies.
At the end every macro of MACROS.md (EXPECTED_LIN below) must have been written, \\ToyKappa or its range form
\\ToyKappaMin and \\ToyKappaMax (EXPECTED_ALT); only those four \\Cost* macros may be
missing, by design, while no GPU benchmark exists (they are listed as such). Otherwise it lists every reason and
writes nothing. NUMBERS_TEST=1 (tests only) lists them and writes the macros as before, with the outputs marked as a
test, and never under lmx_runs."""
import hashlib, json, os, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get('INDIR') or (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'output'))
OUTD = os.environ.get('OUTDIR') or HERE          # where numbers.json and numbers.tex go
SFX = os.environ.get('SUFFIX', '')               # '_lin': the results of the linear mixture
# E11: what the linear run refuses, as (file, reason); NUMBERS_TEST=1 (tests only) writes the macros anyway
NUMBERS_TEST = os.environ.get('NUMBERS_TEST', '') == '1'
FULL_SIZE, FULL_NREF = 2000, 1000                # D7: points of three_bands and band_checks, draws of each box; NREF
REFUSED = []
READ = set()                                     # every input file read, for the checks of I4 (SUFFIX=_lin)


def refuse(path, why):
    if (path, why) not in REFUSED:
        REFUSED.append((path, why))


def need(path, what):
    """True if the input exists; with SUFFIX=_lin a missing one is refused (no macro family is lost in silence)."""
    if os.path.exists(path):
        return True
    if SFX == '_lin':
        refuse(path, f'missing: {what}')
    return False


def jl(path):
    """A json input, recorded for the checks of I4."""
    READ.add(path)
    return json.load(open(path))


def check_size(path, d):
    """E11 (SUFFIX=_lin): refuse the result d of path, a json or the arrays of an npz, if it was made at a reduced size,
    or a benchmark whose recorded device and threads are not those of its name, bench_final_<device>_<threads>."""
    name, keys = os.path.basename(path), set(d.files if hasattr(d, 'files') else d)
    for k in ('hypercube_size', 'nodes_evaluated', 'draws_generated', 'n_events_test'):
        if k in keys:
            refuse(path, f'made at a reduced size ({k} = {d[k]})')
    size = lambda k: len(d[k]) if hasattr(d, 'files') else int(d[k])
    if name.startswith(('three_bands_', 'band_checks_')) and 'nodes' in keys and size('nodes') < FULL_SIZE:
        refuse(path, f'{size("nodes")} points, not {FULL_SIZE}')
    if name.startswith('envelope_test_'):
        n = size('Y_joint') if hasattr(d, 'files') else d['ndraw']
        if n < FULL_SIZE:
            refuse(path, f'{n} draws, not {FULL_SIZE}')
        if not hasattr(d, 'files') and d.get('box') == 'full' and d['nref'] < FULL_NREF:
            refuse(path, f'NREF = {d["nref"]} further draws on the full box, not {FULL_NREF}')
    if 'gen_band_win_points' in keys and d['gen_band_win_points'] < FULL_SIZE:
        refuse(path, f'its band from {d["gen_band_win_points"]} points of three_bands, not {FULL_SIZE}')
    if name.startswith('bench_final_'):
        dev, thr = name.split('.')[0].split('_')[2:4]
        if (d.get('device'), str(d.get('threads'))) != (dev, thr):
            refuse(path, f'records device {d.get("device")!r} with {d.get("threads")!r} threads, its name says {dev} with {thr}')


def jload(path):
    """A result file of the propagation; with SUFFIX=_lin it must hold the linear mixture at the full size (E11)."""
    d = jl(path)
    if SFX == '_lin':
        assert d.get('mix') == 'linear', f'{path} does not hold the linear mixture (mix = {d.get("mix")!r})'
        check_size(path, d)
    return d


def zload(path):
    """An npz of the propagation; with SUFFIX=_lin it must hold the linear mixture at the full size (E11)."""
    READ.add(path)
    z = np.load(path)
    if SFX == '_lin':
        assert 'mix' in z.files and str(z['mix']) == 'linear', f'{path} does not hold the linear mixture'
        check_size(path, z)
    return z
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
    d = jl(p)
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
    if SFX == '_lin' and sname == 'Grid':        # the rerun does not remake the grid cells (no \AccGrid* in the text)
        continue
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
if SFX == '_lin':                                # every training the rerun remakes, both tests (a missing seed changes a mean)
    for sname, pre, fb, stages in SETS[1:]:
        for S in stages:
            for net in ('fac', 'cat'):
                for k in range(3):
                    for strict in (False, True):
                        if pick(pre, fb, S, net, k, strict) is None:
                            name = ALIAS.get((pre, S, net, k)) if not strict else None
                            name = name or '{}{}_{}_{}_s{}.json'.format(pre, 'strict' if strict else '', S, net, k)
                            alt = ' or its fallback {}{}_{}_{}_s{}.json'.format(fb, 'strict' if strict else '', S, net, k) \
                                if fb and not name.startswith('stage_mixture') else ''
                            refuse(f'{D}/{name}', f'missing{alt}: training {k} of the {sname} cell {STAGE_NAME[S]} {net} '
                                                  f'({"strict" if strict else "standard"} test)')

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
# A source is a list of learning-curve tags (widths_v2_<TAG>.json and widths_v2_<TAG>h400.json in the
# results folder, or in LC_DIR) or ('p2', prefix) for the three Phase 2 trainings of the 208-run set. The
# closure width grows with the size of the reference, so every row is scored against a reference of
# the size of the 720-run model's: '<TAG>@r415' reads the widths of rescore_v4.sbatch, which use 415 of
# the 1000 stored events of each of the 2768 runs (1 148 720 events, against 1 149 840).
LC = os.environ.get('LC_DIR', D)
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
need(META208, 'the held-out runs of the 208-run row of the learning curve (a data table)')
held208 = jl(META208)['held'] if os.path.exists(META208) else None
for name, runs, ev, steps, mem, src in LC_ROWS:
    std, strict, std48, strict48 = [], [], [], []
    if isinstance(src, tuple):
        for k in range(3):
            for test, acc in ((False, std), (True, strict)):
                p = pick(src[1], None, 'MIX', 'fac', k, test)
                if p and held208:
                    acc.append(paper_width(p2_per_run(load(p), held208)))
            # the same trainings scored at all 48 held-out points of the 720-run set (data of make_orig48.py)
            for test, acc48 in ((False, std48), (True, strict48)):
                p = pick(src[1] + '48', None, 'MIX', 'fac', k, test)
                if p:
                    acc48.append(load(p)['master'])
                elif SFX == '_lin':
                    refuse(f'{D}/{src[1]}48{"strict" if test else ""}_MIX_fac_s{k}.json', f'missing: the learning-curve row {name}')
    else:
        for tag in src:
            tag, _, rs = tag.partition('@')
            for suffix, acc, acc48 in (('', std, std48), ('h400', strict, strict48)):
                f = f'{LC}/widths_v2_{tag}{suffix}{rs}.json'
                if need(f, f'the learning-curve row {name}'):
                    d = jl(f); acc.append(paper_width(d['per_run'])); acc48.append(d['master'])
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
    S = [jl(f)['summary'] for f in (f'{LC}/wifi_coverage_{t}.json' for t in tags) if need(f, f'the coverage of \\{pre}*')]
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

# ---- the three uncertainties of Sec. 5 (three_bands.py on the final mixture) ----------------------
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
    if SFX == '_lin' and abs(float(f'{v:.{max(n - 1 - e, 0)}f}')) >= 10**(e + 1):
        e += 1                                   # the rounding carries into the next power of ten: 9.95 is 10, not 10.0 (B21)
    return f'{v:.{max(n - 1 - e, 0)}f}'


def textmacro(name, text):
    macros.append(f'\\newcommand{{\\{name}}}{{{text}}}')


f3 = f'{D}/three_bands_{THREE_TAG}{SFX}'
if all([need(f3 + e, 'the three bands of Sec. 5 (\\Tb*, \\Calc*)') for e in ('.json', '.npz')]):
    d3 = jload(f3 + '.json'); c3 = d3['central']; g3 = d3['generator_band']; z3 = zload(f3 + '.npz')
    # the comparisons with DELPHI at its hadron level (D8) when the file has them, else the uncorrected predictions
    CV = '_conv' if 'h_central_conv' in z3.files else ''
    cmp_ = lambda k: z3[k + CV]
    tb = dict(tag=THREE_TAG, table={})
    # The theory band is signed (three_bands.py): the band above the prediction collects the upward
    # displacements of the variations and the band below the downward ones, with the three variations of
    # the dispersive model included. The combined band is signed through it.
    def updown(u, d):
        su, sd_ = sig(u), sig(d)
        return f'\\pm{su}' if su == sd_ else f'{{}}^{{+{su}}}_{{-{sd_}}}'
    # D8: the text quotes <1-T> only against OPAL (\TbThrustImposed\TbThrustComb), so with the corrected predictions the
    # prediction and combined band of <1-T> are those at OPAL's hadron level (cnu/a); the uncorrected ones stay in numbers.json
    AT_MEAS = {'thrust'} if 'conv' in d3 else set()
    for k, nm in OBS_NAME.items():
        cc = d3['conv']['central'] if k in AT_MEAS else c3
        comb = max(cc['combined_up'][k], cc['combined_down'][k])
        dec = max(1 - int(np.floor(np.log10(abs(comb)))), 0)
        row = dict(generator_only=cc['generator_only'][k], imposed=cc['imposed'][k], gen_before=g3['before'][k]['sd'],
                   gen_after=g3['after'][k]['sd'], theory_up=cc['theory_up'][k], theory_down=cc['theory_down'][k],
                   learning=cc['learning_band'][k], combined_up=cc['combined_up'][k], combined_down=cc['combined_down'][k],
                   theory_symmetric_no_dispersive=cc['theory_band'][k])
        if k in AT_MEAS:
            row.update(hadron_level='OPAL (D8, cnu/a)', **{f'{q}_uncorrected': c3[q][k] for q in ('generator_only', 'imposed', 'combined_up', 'combined_down')})
        tb['table'][k] = row
        textmacro(f'Tb{nm}GenOnly', f'{row["generator_only"]:.{dec}f}'); textmacro(f'Tb{nm}Imposed', f'{row["imposed"]:.{dec}f}')
        for key, col in (('gen_before', 'GenBefore'), ('gen_after', 'GenAfter'), ('learning', 'Learn')):
            textmacro(f'Tb{nm}{col}', sig(row[key]))
        textmacro(f'Tb{nm}Theory', updown(row['theory_up'], row['theory_down']))
        textmacro(f'Tb{nm}Comb', updown(row['combined_up'], row['combined_down']))
    lo = z3['lo']; win = (lo >= 0.05) & (lo < 1/3); above = lo >= 1/3; top = (lo >= 0.2 - 1e-9) & (lo < 1/3); low = win & ~top
    hc, hb, dat, err = z3['h_central'], z3['h_central_before'], z3['data'], z3['err']
    hcc, hbc = cmp_('h_central'), cmp_('h_central_before')         # the predictions compared with DELPHI
    if CV:
        tb['comparison_with_delphi'] = 'at the hadron level of DELPHI (D8, the *_conv predictions)'
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
    side = np.where(dat > hcc, cmp_('comb_hist_up'), cmp_('comb_hist_down'))     # the band on the side of the data
    with_band = (hcc - dat)/np.sqrt(err**2 + side**2)
    pulls_ = dict(imposed=rms(((hcc - dat)/err)[win]), generators_only=rms(((hbc - dat)/err)[win]), imposed_with_band=rms(with_band[win]),
                  above_max_with_band=float(np.max(np.abs(with_band[above]))),
                  top_min_with_band=float(np.min(np.abs(with_band[top]))), top_max_with_band=float(np.max(np.abs(with_band[top]))),
                  low_max_with_band=float(np.max(np.abs(with_band[low]))))
    tb['delphi_rms_pull'] = pulls_
    if CV:                                                        # the same without the factors of D8, kept for reference
        su = np.where(dat > hc, z3['comb_hist_up'], z3['comb_hist_down']); wu = (hc - dat)/np.sqrt(err**2 + su**2)
        tb['delphi_rms_pull_uncorrected'] = dict(imposed=rms(((hc - dat)/err)[win]), generators_only=rms(((hb - dat)/err)[win]),
                                                 imposed_with_band=rms(wu[win]), top_max_with_band=float(np.max(np.abs(wu[top]))),
                                                 low_max_with_band=float(np.max(np.abs(wu[low]))))
        tb['top_deficit_percent_uncorrected'] = [float(np.min(100*(1 - hc/dat)[top])), float(np.max(100*(1 - hc/dat)[top]))]
    for key, nm in (('imposed', 'Imposed'), ('generators_only', 'GenOnly'), ('imposed_with_band', 'WithBand'), ('above_max_with_band', 'AboveMax'),
                    ('top_min_with_band', 'TopMin'), ('top_max_with_band', 'TopMax'), ('low_max_with_band', 'LowMax')):
        textmacro(f'TbPull{nm}', sig(pulls_[key]))
    deficit = 100*(1 - hcc/dat)
    tb['top_deficit_percent'] = [float(deficit[top].min()), float(deficit[top].max())]
    textmacro('TbDeficitTopMin', sig(tb['top_deficit_percent'][0])); textmacro('TbDeficitTopMax', sig(tb['top_deficit_percent'][1]))
    # the bins below the window, which three_bands.py writes under _low with the same normalization: the
    # combined band there (the larger side, in percent of the prediction) and the largest pull of DELPHI
    if 'lo_low' in z3.files and len(z3['lo_low']):
        hl, dl, el = z3['h_central' + CV + '_low'], z3['data_low'], z3['err_low']
        bu, bd = z3['comb_hist_up' + CV + '_low'], z3['comb_hist_down' + CV + '_low']
        band_below = 100*np.maximum(bu, bd)/hl
        pull_below = np.abs(hl - dl)/np.sqrt(el**2 + np.where(dl > hl, bu, bd)**2)
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
    ftv = f'{D}/theory_variations_MIX{SFX}.json'
    if need(ftv, 'each generator alone and imposed (theory_variations, \\Tb*AloneWin, \\TbImposed*)'):
        tv = jload(ftv); assert np.allclose(np.asarray(tv['lo']), lo)
        dv_ = np.asarray(tv['data'])
        gk = 'h_gen_only_conv' if 'h_gen_only_conv' in tv['xi0'] else 'h_gen_only'     # against DELPHI: its hadron level (D8)
        g0, g1 = np.asarray(tv['xi0'][gk])/dv_, np.asarray(tv['xi1'][gk])/dv_
        i0, i1 = np.asarray(tv['xi0']['h_imposed'])/dv_, np.asarray(tv['xi1']['h_imposed'])/dv_
        ik = 'h_imposed_conv' if 'h_imposed_conv' in tv['xi0'] else 'h_imposed'
        j0, j1 = np.asarray(tv['xi0'][ik])/dv_, np.asarray(tv['xi1'][ik])/dv_
        # Sherpa against Herwig, both imposed: an agreement between the two predictions, so without the factors
        xs = dict(sherpa_alone_win=100*float(np.max(np.abs(g0[win] - 1))), herwig_alone_win=100*float(np.max(np.abs(g1[win] - 1))),
                  imposed_apart_win=100*float(np.max(np.abs(i0[win] - i1[win]))),
                  imposed_low=100*float(max(np.max(np.abs(j0[low] - 1)), np.max(np.abs(j1[low] - 1)))))
        tb['single_generators_percent'] = xs
        for key, nm in (('sherpa_alone_win', 'SherpaAloneWin'), ('herwig_alone_win', 'HerwigAloneWin'),
                        ('imposed_apart_win', 'ImposedApartWin'), ('imposed_low', 'ImposedLow')):
            textmacro(f'Tb{nm}', sig(xs[key], 1 if xs[key] < 9.5 else 2))
    # the calculation against DELPHI and ALEPH in the twelve bins wholly inside the window (calc_vs_delphi.py)
    fcd = f'{D}/calc_vs_delphi_{THREE_TAG}{SFX}.json'
    if need(fcd, 'the calculation against DELPHI and ALEPH (calc_vs_delphi, \\TbCalcBelowTop*, \\TbAleph*)'):
        cd = jload(fcd); clo = np.asarray(cd['lo']); ct = clo >= 0.2 - 1e-9
        # the calculation, the imposed prediction and ALEPH against DELPHI as published (E12, D3: no distribution is
        # renormalized over a range of bins; the earlier draft's file normalized everything over the twelve bins)
        cs = np.asarray(cd['calc_over_delphi'])
        cvd = dict(imposed_vs_calc_max=100*float(np.max(np.abs(np.asarray(cd['imposed_over_calc']) - 1))),
                   calc_below_top=[100*float(np.min(1 - cs[ct])), 100*float(np.max(1 - cs[ct]))],
                   aleph_below_top=[100*float(np.min(1 - np.asarray(cd['aleph_over_delphi'])[ct])), 100*float(np.max(1 - np.asarray(cd['aleph_over_delphi'])[ct]))],
                   aleph_err_top=[100*float(np.min(np.asarray(cd['aleph_rel_err'])[ct])), 100*float(np.max(np.asarray(cd['aleph_rel_err'])[ct]))])
        tb['calc_vs_delphi'] = cvd
        textmacro('TbImposedVsCalcMax', sig(cvd['imposed_vs_calc_max'], 1))
        for key, nm in (('calc_below_top', 'CalcBelowTop'), ('aleph_below_top', 'AlephBelowTop'), ('aleph_err_top', 'AlephErrTop')):
            textmacro(f'Tb{nm}Min', sig(cvd[key][0], 1 if cvd[key][0] < 9.5 else 2)); textmacro(f'Tb{nm}Max', sig(cvd[key][1], 1 if cvd[key][1] < 9.5 else 2))
    # the published comparisons of the captions: OPAL <1-T> and the L3 charged multiplicity, against the band on their side
    OPAL_THRUST, L3_NCH = 0.06671, 18.63
    # D8: OPAL <1-T> against the prediction at its hadron level (cnu/a) with that prediction's bands; L3 <n_ch> takes no
    # factor, so its prediction is the uncorrected one
    cc3 = d3['conv']['central'] if 'conv' in d3 else c3
    cside = lambda k, x: cc3['combined_up'][k] if x > cc3['imposed'][k] else cc3['combined_down'][k]
    tb['opal_pull'] = (OPAL_THRUST - cc3['imposed']['thrust'])/cside('thrust', OPAL_THRUST)
    if 'conv' in d3:
        tb['opal_prediction_at_its_hadron_level'] = cc3['imposed']['thrust']
    tb['l3_in_gen_bands'] = (L3_NCH - c3['imposed']['nch'])/g3['after']['nch']['sd']
    if SFX == '_lin':
        # E12, with the errors of the measurements as main.tex quotes them. OPAL, Eur. Phys. J. C 40 (2005) 287, Table 7,
        # 91 GeV: 0.06671 +- 0.00017 (stat.) +- 0.00066 (syst.), 0.00068 in quadrature (rivet_ref/OPAL_2004_S6132243.yoda
        # d15-x01-y01 holds 6.815e-4). L3, Phys. Lett. B 577 (2003) 109: 18.63 +- 0.11 (directlib.Model's nch_data).
        OPAL_THRUST_ERR, L3_NCH_ERR = 0.00068, 0.11
        tb['measurements'] = dict(opal_thrust=[OPAL_THRUST, OPAL_THRUST_ERR], l3_nch=[L3_NCH, L3_NCH_ERR])
        # TbOpalPull: OPAL minus the prediction at its hadron level, over the combined band on OPAL's side and OPAL's error
        # in quadrature (the band alone stays here as opal_in_combined_bands)
        tb['opal_in_combined_bands'] = tb['opal_pull']
        tb['opal_pull'] = float((OPAL_THRUST - cc3['imposed']['thrust'])/np.hypot(cside('thrust', OPAL_THRUST), OPAL_THRUST_ERR))
        # TbLThreeBands: L3 minus the prediction at the default tunes, over the generator band after imposing; TbLThreePull:
        # the same over the generator band and L3's error in quadrature
        tb['l3_pull'] = float((L3_NCH - c3['imposed']['nch'])/np.hypot(g3['after']['nch']['sd'], L3_NCH_ERR))
    textmacro('TbOpalPull', sig(tb['opal_pull'])); textmacro('TbLThreeBands', sig(tb['l3_in_gen_bands']))
    if SFX == '_lin':
        textmacro('TbLThreePull', sig(tb['l3_pull']))
    tb['theory_window_mean_percent'] = [100*c3['theory_up']['tau_win']/c3['imposed']['tau_win'], 100*c3['theory_down']['tau_win']/c3['imposed']['tau_win']]
    textmacro('TbTheoryTauWinUpPercent', sig(tb['theory_window_mean_percent'][0])); textmacro('TbTheoryTauWinDownPercent', sig(tb['theory_window_mean_percent'][1]))
    textmacro('CalcAlphas', f'{d3["calc"]["alpha_s"]:.4f}'); textmacro('CalcAlphaZero', f'{d3["calc"]["alpha_0"]:.3f}')
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
    tr = [f'{D}/three_bands_{THREE_TAG}{t}{SFX}.json' for t in ('', 's1', 's2')]
    if all([need(f, 'the three trainings at the central point (\\TbTrainSpread*)') for f in tr]):
        cs = [jload(f)['central'] for f in tr]
        assert all(np.allclose(cc['theta'], c3['theta']) for cc in cs)
        ratio = {k: float(np.std([cc['imposed'][k] for cc in cs], ddof=1)/c3['learning_band'][k]) for k in OBS_NAME}
        tb['training_spread_over_learning_band'] = ratio
        textmacro('TbTrainSpreadMin', sig(min(ratio.values()), 1)); textmacro('TbTrainSpreadMax', sig(max(ratio.values()), 1))
    # band_checks.py: the part of the generator band from the mixing fraction, and the largest weight
    fbc = f'{D}/band_checks_{THREE_TAG}{SFX}.json'
    if need(fbc, 'the checks of the bands (band_checks, \\TbXi*, \\TbMaxWeight*)'):
        bc = jload(fbc); assert np.allclose(bc['central'], c3['theta'])
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
    # the window under D3 (linear run, round B21): every bin inside the window is the window fraction P_win, which the
    # moments leave to the generators, times the shape inside the window, which the calculation fixes. The shape is the
    # twelve bins wholly inside the window (0.05 to 0.30) normalized over them, the *_win keys of the npz. Its generator
    # band (the draws, stored); its learning band (C_learn of the means and bins, through the normalization at the
    # central point); its theory band (Eq. (thband) at each xi on the displacement of the normalized mixture, sigma_+^2
    # and sigma_-^2 averaged over xi flat on [0, 1] as D9; the normalization bends in xi, so a midpoint rule on 4000
    # points, checked on the bins themselves against theory_hist_up and theory_hist_down), and the bins where its larger
    # side exceeds the shape's generator band of the generators alone; Sherpa's and Herwig's imposed shapes apart, in
    # percent of DELPHI's as \TbImposedApartWin; the imposed shape against the calculation's. Then P_win of each
    # generator and its bands in percent, the fraction of all events from 0.05 to 0.30 against DELPHI's (at DELPHI's
    # hadron level, D3, D5, D8), the pulls of the three highest window bins with the theory band alone, the deficit of the
    # window bins below 0.2, and the calculation against DELPHI in the bin across the upper edge of the window. An input
    # without these keys writes none of them, and the check of EXPECTED_LIN below names them
    W_KEYS = ('lo_win', 'hi_win', 'h_central_win', 'h_central_before_win', 'gen_hist_win', 'gen_hist_before_win', 'C_learn',
              'C_learn_before', 'cov_labels', 'np_labels', 'theory_var_hist_sherpa', 'theory_var_hist_herwig', 'np_hist_sherpa',
              'np_hist_herwig', 'h_central_sherpa', 'h_central_herwig', 'H_after', 'rows_after', 'obs')
    if SFX == '_lin' and all(k in z3.files for k in W_KEYS) and 'generators' in c3:
        hi3 = z3['hi']; wd = hi3 - lo; w12 = (lo >= 0.05 - 1e-9) & (hi3 <= 1/3 + 1e-9)
        assert np.allclose(lo[w12], z3['lo_win']) and np.allclose(hi3[w12], z3['hi_win']), 'not the twelve bins of the *_win keys'
        nrm = lambda h: np.asarray(h)[..., w12]/np.sum(np.asarray(h)[..., w12]*wd[w12], axis=-1, keepdims=True)
        assert np.allclose(nrm(hc), z3['h_central_win'], rtol=1e-12, atol=0)
        ws = dict(gen_before=100*z3['gen_hist_before_win']/z3['h_central_before_win'], gen_after=100*z3['gen_hist_win']/z3['h_central_win'])
        cvl = [str(l) for l in z3['cov_labels']]; ix = [cvl.index(f'thr_{a:.3f}_{b:.3f}') for a, b in zip(lo[w12], hi3[w12])]
        for key, ck, h in (('learn_before', 'C_learn_before', hb), ('learn_after', 'C_learn', hc)):
            S_ = float(np.sum(h[w12]*wd[w12])); s_ = h[w12]/S_
            J_ = np.zeros((len(ix), len(cvl))); J_[:, ix] = (np.eye(len(ix)) - np.outer(s_, wd[w12]))/S_     # d(h_i/S)/dh_k
            ws[key] = 100*np.sqrt(np.diag(J_ @ z3[ck] @ J_.T))/s_
        PAIRS_ = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"), ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
        SCHEME_, npl = "(1.0, 1.0, 'modR', 1.0)", [str(l) for l in z3['np_labels']]
        dg = {g: dict(zip(vl + npl, np.vstack([z3[f'theory_var_hist_{g}'], z3[f'np_hist_{g}']]))) for g in ('sherpa', 'herwig')}
        xg = ((np.arange(4000) + 0.5)/4000)[:, None]; hS, hH = z3['h_central_sherpa'], z3['h_central_herwig']

        def thb(dd, up):
            """sigma_+ (up) or sigma_- of Eq. (thband) from the displacements dd = {label: (xi, bin)}, its square averaged over xi"""
            f = (lambda v: np.maximum(v, 0.0)) if up else (lambda v: np.maximum(-v, 0.0))
            t = sum(np.maximum(f(dd[a]), f(dd[b]))**2 for a, b in PAIRS_) + f(dd[SCHEME_])**2 + sum(f(dd[l])**2 for l in npl)/len(npl)
            return np.sqrt(np.mean(t, axis=0))
        dbin = {l: (1 - xg)*dg['sherpa'][l] + xg*dg['herwig'][l] for l in dg['sherpa']}
        assert all(np.allclose(thb(dbin, u), z3[k], rtol=1e-6, atol=1e-9*np.max(z3[k])) for u, k in ((True, 'theory_hist_up'), (False, 'theory_hist_down'))), \
            'Eq. (thband) averaged over xi does not give the theory band of the bins (D9)'
        m0 = nrm((1 - xg)*hS + xg*hH)
        dshp = {l: nrm((1 - xg)*(hS + dg['sherpa'][l]) + xg*(hH + dg['herwig'][l])) - m0 for l in dg['sherpa']}
        ws['theory_up'], ws['theory_down'] = (100*thb(dshp, u)/z3['h_central_win'] for u in (True, False))
        big = np.maximum(ws['theory_up'], ws['theory_down']); ws['theory_minus_gen_before'] = big - ws['gen_before']
        textmacro('TbShapeWinBins', str(int(w12.sum()))); textmacro('TbShapeTheoryWiderBins', str(int(np.sum(big > ws['gen_before']))))
        for key, nm in (('gen_before', 'GenBeforeWin'), ('gen_after', 'GenAfterWin'), ('learn_before', 'LearnBeforeWin'), ('learn_after', 'LearnWin')):
            textmacro(f'TbShape{nm}Min', sig(ws[key].min())); textmacro(f'TbShape{nm}Max', sig(ws[key].max()))
        # P_win of each generator (the decimals of \TbPwinImposed), its bands in percent, and how the window bins follow it
        G3, ip = c3['generators'], [str(o) for o in z3['obs']].index('pwin')
        assert np.isclose(z3['rows_after'][:, ip].std(ddof=1), g3['after']['pwin']['sd'], rtol=1e-9, atol=0)
        dp_ = max(1 - int(np.floor(np.log10(abs(max(c3['combined_up']['pwin'], c3['combined_down']['pwin']))))), 0)
        pw = dict(sherpa=G3['sherpa']['imposed']['pwin'], herwig=G3['herwig']['imposed']['pwin'], mixture=c3['imposed']['pwin'],
                  gen_after_percent=100*g3['after']['pwin']['sd']/c3['imposed']['pwin'],
                  learn_percent=100*c3['learning_band']['pwin']/c3['imposed']['pwin'],
                  corr_with_window_bins=[float(np.corrcoef(z3['H_after'][:, i], z3['rows_after'][:, ip])[0, 1]) for i in np.where(win)[0]])
        textmacro('TbPwinSherpa', f'{pw["sherpa"]:.{dp_}f}'); textmacro('TbPwinHerwig', f'{pw["herwig"]:.{dp_}f}')
        textmacro('TbPwinGenAfterPercent', sig(pw['gen_after_percent'])); textmacro('TbPwinLearnPercent', sig(pw['learn_percent']))
        # the fraction of all events from 0.05 to 0.30, each prediction at DELPHI's hadron level as in \TbDeficitTop*
        fr = dict(delphi=float(np.sum((dat*wd)[w12])), mixture=float(np.sum((hcc*wd)[w12])))
        if os.path.exists(ftv):
            fr.update(sherpa=float(np.sum((np.asarray(tv['xi0'][ik])*wd)[w12])), herwig=float(np.sum((np.asarray(tv['xi1'][ik])*wd)[w12])))
            ws['sherpa_minus_herwig'] = 100*(nrm(tv['xi0']['h_imposed']) - nrm(tv['xi1']['h_imposed']))/nrm(dv_)
            ap_ = float(np.max(np.abs(ws['sherpa_minus_herwig']))); textmacro('TbShapeApartWin', sig(ap_, 1 if ap_ < 9.5 else 2))
        if os.path.exists(fcd):
            assert np.allclose(np.asarray(cd['lo']), lo[w12])
            fr['calculation'] = float(np.sum(np.asarray(cd['calc_over_delphi'])*(dat*wd)[w12]))
            ws['imposed_over_calc'] = np.asarray(cd['imposed_win_over_calc_win'])
            textmacro('TbShapeVsCalcMax', sig(100*float(np.max(np.abs(ws['imposed_over_calc'] - 1))), 1))
        below = {k: 100*(1 - v/fr['delphi']) for k, v in fr.items() if k != 'delphi'}
        for k, nm in (('mixture', ''), ('sherpa', 'Sherpa'), ('herwig', 'Herwig')):
            if k in below:
                textmacro(f'TbWinFracBelow{nm}', sig(below[k]))
        if 'calculation' in below:
            textmacro('TbWinFracCalcApart', sig(abs(below['calculation']), 1))
        tside = np.where(dat > hcc, cmp_('theory_hist_up'), cmp_('theory_hist_down'))      # the theory band on the side of the data
        p_th = np.abs(hcc - dat)/np.sqrt(err**2 + tside**2)
        textmacro('TbPullTheoryTopMin', sig(p_th[top].min())); textmacro('TbPullTheoryTopMax', sig(p_th[top].max()))
        textmacro('TbDeficitLowMin', sig(deficit[low].min())); textmacro('TbDeficitLowMax', sig(deficit[low].max()))
        fcb_ = f'{D}/calc_bins_nlo_point.json'
        if os.path.exists(fcb_):
            cbj = jl(fcb_); cdl = cbj['delphi']; el, eh = np.asarray(cdl['lo']), np.asarray(cdl['hi'])
            assert np.allclose(np.asarray(cdl['y'])[np.isin(np.round(el, 6), np.round(lo, 6))], dat), 'calc_bins holds other DELPHI values'
            k_ = np.where((el < cbj['window'][1]) & (eh > cbj['window'][1]))[0]; assert len(k_) == 1
            edge = dict(lo=float(el[k_[0]]), hi=float(eh[k_[0]]), calc_over_delphi=float(cdl['calc']['central'][k_[0]]/cdl['y'][k_[0]]))
            edge['below_percent'] = 100*(1 - edge['calc_over_delphi']); tb['calc_vs_delphi_edge_bin'] = edge
            textmacro('TbCalcBelowEdge', sig(edge['below_percent'], 1 if edge['below_percent'] < 9.5 else 2))
        tb['window_d3'] = dict(shape_bins=[[float(a), float(b)] for a, b in zip(lo[w12], hi3[w12])],
                               shape_percent={k: np.asarray(v).tolist() for k, v in ws.items()}, pwin=pw,
                               fraction_005_030=fr, fraction_below_delphi_percent=below,
                               pulls_top_theory_band_alone=p_th[top].tolist(), deficit_low_percent=deficit[low].tolist())
    numbers['three_bands'] = tb

# ---- the learning band: the fitted basis weights of the three trainings of the final mixture -----
WIFI_TAGS = os.environ.get('WIFI_TAGS', 'MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2').split()
ORD = ['One', 'Two', 'Three']
wifi = {}
for i, t in enumerate(WIFI_TAGS):
    ff, fd = f'{D}/wifi_{t}_fit.json', f'{D}/wifi_{t}_diag.json'
    if not need(ff, 'the learning band of each training (\\Wifi*)'):
        continue
    fit = jl(ff); row = dict(newton=fit['newton_iterations'], logw_median=fit['logw_sigma_median'],
                                          logw_p95=fit['logw_sigma_p95'], w=fit['w'], w_err=fit['w_err'])
    if need(fd, 'the learning band of each training (\\Wifi*)'):
        dg = jl(fd)
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
if need(fmap, 'the map of Stage C (\\Map*)'):
    vm = jl(fmap)
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

# ---- Sec. 3.2 and App. A.2: two halves of one fresh run, calibration constants, selected steps ---
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
if need(fgs, 'the split of the loss gain by generator (\\WifiGain*)'):
    gs = jl(fgs); numbers['wifi_gensplit'] = gs
    g = gs['gain']
    macro('WifiGainTotal', -g['loss'], 2); macro('WifiGainOther', -g['den_other'], 2)
    macro('WifiLossNum', g['num'], 2); macro('WifiLossOwn', g['den_same'], 2)
    macro('WifiZOther', gs['published']['mean_z_other'], 1)

# ---- single networks against the mean of members (single_members.py, Stage C, first training) ----
fsm = f'{D}/single_members_C_P2augfullfac0.json'
if need(fsm, 'the single networks of Stage C (\\Single*)'):
    sm = jl(fsm); numbers['single_members'] = sm
    ws = [v['width'] for k, v in sm.items() if k.startswith('member_')]
    macro('SingleMin', min(ws), 2); macro('SingleMax', max(ws), 2); macro('SingleMean', sm['mean_of_members']['width'], 2)
# ---- the three trainings at the central point of Sec. 5: the spread of <1-T> with the calculation imposed
th3 = [jload(f)['central']['imposed']['thrust'] for f in
       (f'{D}/three_bands_{t}{SFX}.json' for t in WIFI_TAGS) if os.path.exists(f)]
if len(th3) == 3:
    numbers['seed_thrust'] = th3
    textmacro('SeedThrustSd', sig(float(np.std(th3, ddof=1))))
    # the same in units of 10^-4, beside the learning band of <1-T> in the same unit
    textmacro('SeedThrustSdEFour', f'{1e4*float(np.std(th3, ddof=1)):.1f}')
    if 'three_bands' in numbers:
        textmacro('TbThrustLearnEFour', f'{1e4*numbers["three_bands"]["table"]["thrust"]["learning"]:.1f}')
# ---- the wide coupling scan (Sec. 3.4): three references of equal size, mean width per held point -
def _pt(f):
    """{coupling: mean width over the observables} of one stage-W result"""
    return {float(k.strip('()').split(',')[0]): float(np.mean(list(v.values()))) for k, v in load(f)['per_point'].items()}
# the references hold 6.3e4 events, so the networks train for 1.8e4 steps (DESIGN_TAG wb18), which
# shows each reference event as often as the 3.24e5 steps of the mixture show its 1.15e6 events
fs, fp = f'{D}/p2wb18single_W_fac_s0.json', f'{D}/p2wb18pool_W_fac_s0.json'
fl = [f'{D}/p2wb18local{r}_W_fac_s0.json' for r in range(9100, 9106)]
if all([need(f, 'the wide coupling scan (\\Wide*)') for f in [fs, fp] + fl]):
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

# ---- the DCTR network truncated to its leading singular components (Sec. 3.6) ---------------------
# r2_ladder.py with LADDER_TRUNCATE=1 on the first concatenation training of each model: the smallest
# number of components whose closure width comes within 0.02 of that of the full network
svd = {}
for S, nm in (('A', 'A'), ('B', 'B'), ('C', 'C'), ('E', 'Sherpa'), ('MIX', 'Mix')):
    f = f'{D}/p2trunc_{S}_cat_s0.json'
    if need(f, 'the truncated network (\\SvdRank*)') and 'truncation' in load(f):
        tr_ = load(f)['truncation']; svd[S] = tr_
        if tr_['rank_within_0_02'] is not None:
            textmacro(f'SvdRank{nm}', str(tr_['rank_within_0_02']))
numbers['concat_truncation'] = svd

# ---- the input keeps the 100 most energetic particles: the fraction of events truncated ------------
ftr = f'{D}/truncation_fraction.json'          # written by trunc_frac.py over the final training runs
if need(ftr, 'the truncation fraction (\\Trunc*, trunc_lmx.sbatch of the scoring side)'):
    tr = jl(ftr); numbers['truncation'] = tr
    fr = [100*v['fraction'] for v in tr.values()]
    textmacro('TruncMinPercent', sig(min(fr))); textmacro('TruncMaxPercent', sig(max(fr)))
    textmacro('TruncMaxMult', str(max(v['max'] for v in tr.values())))

# ---- the energy-fraction spectrum: its closure width between two halves of one run (xp_null.py) ----
# its uncertainty per bin counts the events that contribute to the bin, so the expectation is below one
fxp = f'{D}/xp_null.json'
if need(fxp, 'the x_p spectrum between two halves of one run (\\XpNullMean)'):
    xp = jl(fxp); numbers['xp_null'] = xp
    macro('XpNullMean', xp['mean'], 2)

# ---- the Gaussian toy with the paper's network and training (toy_final.py, Sec. toy) ---------------
# three trainings at K=48 and one at each K of the scan; the figure and the slopes are the first training's
toys = {}
# the linear run reads the nine trainings of toy_final_lmx.sbatch by name (every one required), the default run every file
TOY_FILES = [f'toy_final_K{K}_s{s}.json' for K, s in ((48, 0), (48, 1), (48, 2), (1, 0), (2, 0), (3, 0), (4, 0), (8, 0), (16, 0))]
for fn in (TOY_FILES if SFX == '_lin' else os.listdir(D)):
    if fn.startswith('toy_final_K') and fn.endswith('.json') and need(os.path.join(D, fn), 'the toy model (\\Toy*)'):
        t_ = jl(os.path.join(D, fn)); toys[(int(t_['K']), int(t_['seed']))] = t_
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
        # main.tex 2174: "kappa = \ToyKappa in each". Two decimals for a multiple of 0.01 (the published 1.00), three
        # otherwise, since the grid steps by 0.025 (1.025 must not print as 1.02, nor 0.875 as 0.88; round B11)
        macro('ToyKappa', kap[0], 2 if abs(100*kap[0] - round(100*kap[0])) < 1e-6 else 3)
    else:
        # the three trainings chose different points of the grid (steps of 0.025): the sentence becomes a range,
        # \ToyKappaMin to \ToyKappaMax, three decimals as \KappaMin and \KappaMax (EXPECTED_ALT; WRITING_NOTES.md)
        macro('ToyKappaMin', kap[0], 3); macro('ToyKappaMax', kap[-1], 3)
        print('TOY KAPPA DIFFERS: the three K = 48 trainings chose kappa ' +
              ', '.join(f'{toys[(48, s)]["kappa"]:g}' for s in range(3)) + ': \\ToyKappaMin and \\ToyKappaMax written, '
              'not \\ToyKappa (main.tex 2174 must give the range)')
    if 1 in scan:
        macro('ToyWidthKOne', scan[1], 2); macro('ToyQShortKOne', 100*(1 - toys[(1, 0)]['coef_q_ratio_at_021']), 0)
    rest = [v for k, v in scan.items() if k >= 2]
    macro('ToyWidthScanMin', min(rest), 2); macro('ToyWidthScanMax', max(rest), 2)

# ---- the cost per parameter point (Sec. 3.6, Fig. cost): bench_cost_final.py on the final mixture -----
# (a) the weights of the reference events alone, and (n) the weights and one backward pass, the derivatives
# of one prediction (Fig. cost), on one CPU process with four threads and on one H200. The rest of a step
# (the maximum-entropy weight and its derivatives, with a chi^2 that adds a few percent, the same for every
# network) on the CPU, quoted to one figure. The profile fit of COST_STEPS steps (\CostFit*) belongs to
# an earlier draft and is not used by the paper.
COST_STEPS = 6000
fc, fg = f'{D}/bench_final_cpu_4{SFX}.json', f'{D}/bench_final_cuda_12{SFX}.json'
COST_GPU = ('CostFacGpu', 'CostLateGpu', 'CostCatGpu', 'CostWCatGpu')   # the only ones the GPU benchmark alone gives (E9)
if os.path.exists(fg) or SFX == '_lin':        # the linear run needs the CPU benchmark (step 5a) in any case
    need(fc, 'the CPU benchmark (\\Cost*)')
if os.path.exists(fc):                         # without the GPU benchmark its four macros are missing by design (E9)
    bc = jload(fc); bg = jload(fg) if os.path.exists(fg) else None
    nc, rest, ac = bc['n_network_step_seconds'], bc['b_tilt_and_chi2_seconds'], bc['a_weights_seconds']
    ng, ag = (bg['n_network_step_seconds'], bg['a_weights_seconds']) if bg else (None, None)
    if 'concatenation' in nc and (bg is None or 'concatenation' in ng):
        fh = {k: COST_STEPS*(v + rest)/3600 for k, v in nc.items()}
        numbers['cost'] = dict(cpu=nc, gpu=ng, rest=rest, steps=COST_STEPS, fit_hours=fh, cpu_name=bc.get('cpu'),
                               gpu_name=bg.get('gpu'), gradcheck_ok=bg.get('gradcheck_ok')) if bg else \
            dict(cpu=nc, rest=rest, steps=COST_STEPS, fit_hours=fh, cpu_name=bc.get('cpu'))
        textmacro('CostFacCpu', sig(nc['factorized']) + '\\,s'); textmacro('CostRestCpu', sig(rest, 1) + '\\,s')
        textmacro('CostWFacCpu', sig(ac['factorized']) + '\\,s'); textmacro('CostWCatCpuMin', f'{ac["concatenation"]/60:.0f}')
        if bg:
            textmacro('CostWCatGpu', sig(ag['concatenation']) + '\\,s')
        textmacro('CostLateCpu', sig(nc['late']) + '\\,s'); textmacro('CostCatCpuMin', f'{nc["concatenation"]/60:.0f}')
        if bg:
            textmacro('CostFacGpu', sig(1e3*ng['factorized']) + '\\,ms'); textmacro('CostLateGpu', sig(ng['late']) + '\\,s')
            textmacro('CostCatGpu', sig(ng['concatenation']) + '\\,s')
        textmacro('CostFitFacHours', sig(fh['factorized'])); textmacro('CostFitLateDays', sig(fh['late']/24))
        textmacro('CostFitFacMin', f'{60*fh["factorized"]:.0f}')
        textmacro('CostFitCatDays', f'{fh["concatenation"]/24:.0f}')
        numbers['cost']['weights'] = dict(cpu=ac, gpu=ag) if bg else dict(cpu=ac)
        # the variations of Sec. 4.3 (400 in the earlier draft, 2000 in the linear run: the draws of the full box) need only
        # the weights at each point
        fe = f"{os.environ.get('ENV_DIR', D)}/envelope_test_MIXGEOV3S9_full{SFX}.json"
        NVAR = int(jload(fe)['ndraw']) if os.path.exists(fe) else 400
        need(fe, 'the full envelope box of D7 (\\Env*, the draws of \\CostVar*)')    # E11: the linear run takes no fallback
        nv = {k: NVAR*v for k, v in ac.items()}; numbers['cost']['variations'] = dict(n=NVAR, cpu_seconds=nv)
        textmacro('CostVarFacSec', f'{nv["factorized"]:.0f}'); textmacro('CostVarLateHours', f'{nv["late"]/3600:.1f}')
        textmacro('CostVarCatDays', f'{nv["concatenation"]/86400:.1f}')


# ---- the envelope of 400 variations against the linear covariance (Secs. 4.3 and 5.3, envelope_test.py) ---
# boxes full (the trained box), tune (every range a third of its width around the default tunes, xi in
# [1/3, 2/3]), tune_xifull (xi over [0, 1]), tune_sherpa and tune_herwig (xi fixed). Ratios are the linear
# standard deviation over that of the variations, before (prior only) or after (calculation imposed), for
# the bins of thrust and multiplicity and the seven means. Read from ENV_DIR (default: the results folder).
ENV = os.environ.get('ENV_DIR', D)
def _env(box):
    f = f'{ENV}/envelope_test_MIXGEOV3S9_{box}{SFX}.json'
    return jload(f) if os.path.exists(f) else None
def _grp(d, g):
    lab = d['labels']
    if g == 'means':
        return lab[:7]
    return [l for l in lab if l.startswith({'thrust': 'thr_', 'nch': 'nch_'}[g] )]
def _r(d, key, groups=('means', 'thrust', 'nch')):
    c = d['compare'][key]['ratio']
    return np.array([c[l] for g in groups for l in _grp(d, g) if c[l] is not None], float)
def _pct(x):
    return f'{100*x:.0f}'
EV = {b: _env(b) for b in ('full', 'tune', 'tune_xifull', 'tune_sherpa', 'tune_herwig', 'full_sherpa', 'full_herwig')}
if SFX == '_lin':                                # the six boxes of the rerun (tune_xifull has no counterpart, D1)
    need(f'{ENV}/envelope_test_MIXGEOV3S9_full{SFX}.json', 'the full envelope box of D7 (\\Env*, the draws of \\CostVar*)')
    for b in ('tune', 'tune_sherpa', 'tune_herwig', 'full_sherpa', 'full_herwig'):
        need(f'{ENV}/envelope_test_MIXGEOV3S9_{b}{SFX}.json', f'the envelope box {b} of D7 (\\Env*)')
    for b in ('full_sherpa', 'full_herwig'):
        need(f'{ENV}/envelope_test_MIXGEOV3S9_{b}{SFX}.npz', f'the draws of the envelope box {b} (\\Env*FullShift)')
    need(f'{ENV}/envelope_test_MIXGEOV3S9_full{SFX}.npz', 'the draws of the full envelope box (\\EnvFullShift, \\EnvFullCorr*)')
if EV['full'] and EV['tune']:
    ef, et = EV['full'], EV['tune']; env = {}
    textmacro('EnvNdraw', str(et['ndraw'])); textmacro('EnvNref', str(ef['nref']))
    def _bse(d):
        c = d['compare']['before/joint/lin_flat']; v = []
        for g in ('means', 'thrust', 'nch'):
            for l in _grp(d, g):
                if c['ratio'][l]:
                    v.append(c['ratio_se'][l]/c['ratio'][l])
        return min(v), max(v)
    for nm, d in (('Tune', et), ('Full', ef)):
        lo_, hi_ = _bse(d); textmacro(f'EnvBoot{nm}Min', _pct(lo_)); textmacro(f'EnvBoot{nm}Max', _pct(hi_)); env[f'boot_{nm}'] = [lo_, hi_]
    if f'before/draws{ef["ndraw"]}_vs_ref' in ef['compare']:       # NREF more draws on the full box
        rr = _r(ef, f'before/draws{ef["ndraw"]}_vs_ref'); textmacro('EnvNoiseSd', _pct(np.max(np.abs(rr - 1))))
        cn = max(ef['compare'][f'before/draws{ef["ndraw"]}_vs_ref'][f'corr_{g}']['max_abs_diff'] for g in ('thrust', 'nch'))
        textmacro('EnvNoiseCorr', f'{cn:.2f}')
    rt = _r(et, 'before/joint/lin_flat'); textmacro('EnvTuneSd', _pct(np.max(np.abs(rt - 1))))
    for g, nm in (('thrust', 'Thr'), ('nch', 'Nch')):
        textmacro(f'EnvTuneCorr{nm}', f"{et['compare']['before/joint/lin_flat'][f'corr_{g}']['max_abs_diff']:.2f}")
    rz = _r(et, 'before/joint/lin_z'); textmacro('EnvZMin', _pct(rz.min() - 1)); textmacro('EnvZMax', _pct(rz.max() - 1))
    textmacro('EnvLearnTune', _pct(np.max(np.abs(_r(et, 'before/omega/lin_flat') - 1))))
    textmacro('EnvLearnFull', _pct(np.max(np.abs(_r(ef, 'before/omega/lin_flat') - 1))))
    rb = _r(ef, 'before/joint/lin_flat', ('thrust', 'nch')); rm = _r(ef, 'before/joint/lin_flat', ('means',))
    textmacro('EnvFullBinMin', _pct(1 - rb.max())); textmacro('EnvFullBinMax', _pct(1 - rb.min()))
    textmacro('EnvFullMeanMin', _pct(1 - rm.max())); textmacro('EnvFullMeanMax', _pct(1 - rm.min()))
    if 'before/joint/xi_quad' in ef['compare']:                       # the earlier draft's quadrature in xi
        rq = _r(ef, 'before/joint/xi_quad'); textmacro('EnvXiQuadMin', _pct(rq.min())); textmacro('EnvXiQuadMax', _pct(rq.max()))
        rqt = _r(ef, 'before/joint/xi_quad', ('thrust',)); rqn = _r(ef, 'before/joint/xi_quad', ('nch',))
        textmacro('EnvXiQuadThrN', str(int(np.sum(np.abs(rqt - 1) <= 0.13)))); textmacro('EnvThrBins', str(len(rqt)))
        textmacro('EnvXiQuadNchMin', _pct(rqn.min())); textmacro('EnvXiQuadNchMax', _pct(rqn.max()))
        rqt2 = _r(et, 'before/joint/xi_quad'); textmacro('EnvTuneXiQuadMin', _pct(rqt2.min())); textmacro('EnvTuneXiQuadMax', _pct(rqt2.max()))
    else:
        textmacro('EnvThrBins', str(len(_r(ef, 'before/joint/lin_flat', ('thrust',)))))
    fz = f'{ENV}/envelope_test_MIXGEOV3S9_full{SFX}.npz'
    if os.path.exists(fz):                                       # the shift of the mean and the correlations at the pivot
        z = zload(fz); lab = [str(l) for l in z['labels']]; no = len(lab); Y = z['Y_joint'][:, :no]; yc = z['y_central'][:no]
        idx = {g: [lab.index(l) for l in _grp(ef, g)] for g in ('means', 'thrust', 'nch')}
        allx = sum(idx.values(), []); sh = np.abs(Y.mean(0) - yc)/Y.std(0, ddof=1)
        textmacro('EnvFullShift', f'{np.max(sh[allx]):.1f}')
        Ce = np.cov(Y, rowvar=False); Cl = z['C_lin_flat'][:no, :no] + z['C_learn'][:no, :no]
        cr = lambda A: A/np.sqrt(np.outer(np.diag(A), np.diag(A)))
        piv, els = [], []
        for g in ('thrust', 'nch'):
            ii = np.array(idx[g]); dd = np.abs(cr(Cl[np.ix_(ii, ii)]) - cr(Ce[np.ix_(ii, ii)]))
            p2 = np.argsort(Y.std(0, ddof=1)[ii]/np.abs(yc[ii]))[:2]       # the two bins that move least: the pivot
            iu = np.triu_indices(len(ii), 1); atp = np.isin(iu[0], p2) | np.isin(iu[1], p2)
            piv.append(dd[iu][atp].max()); els.append(dd[iu][~atp].max())
        textmacro('EnvFullCorrPivot', f'{max(piv):.1f}'); textmacro('EnvFullCorrElse', f'{max(els):.1f}')
    win = lambda d, l: l.startswith('thr_') and 0.05 <= float(l.split('_')[1]) < 1/3
    def _after(d, inside):
        c = d['compare']['after/joint/lin_flat']['ratio']
        return np.array([c[l] for l in _grp(d, 'thrust') if c[l] is not None and win(d, l) == inside], float)
    textmacro('EnvAfterWin', _pct(max(np.max(np.abs(_after(d, True) - 1)) for d in (et, ef))))
    textmacro('EnvAfterOutTune', _pct(np.max(np.abs(_after(et, False) - 1))))
    ro = _after(ef, False); textmacro('EnvAfterOutFullMin', _pct(1 - ro.max())); textmacro('EnvAfterOutFullMax', _pct(1 - ro.min()))
    cc = ef['compare']['after/c/lin_flat']; out = [l for l in _grp(ef, 'thrust') if not win(ef, l)]
    textmacro('EnvAfterNorm', f"{100*np.median([cc['sd_exact'][l]/ef['central']['after'][l] for l in out]):.1f}")
    for b, nm in (('tune_sherpa', 'Sherpa'), ('tune_herwig', 'Herwig'), ('full_sherpa', 'SherpaFull'), ('full_herwig', 'HerwigFull')):
        if EV[b]:
            textmacro(f'Env{nm}Sd', _pct(np.max(np.abs(_r(EV[b], 'before/joint/lin_flat') - 1))))
    # D1: every prediction is linear in xi and the formula takes the moments of xi exactly, so what bends over the full
    # box is each generator in its own parameters: each generator alone over its full ranges (xi = 0 or 1), as the full
    # box above. Bin and Mean: how much narrower the linear bands are than the spread of the draws, in percent (signed:
    # negative where wider); Corr: the largest difference of the bin correlations of either distribution; Shift: the
    # largest distance of the mean of the draws from the prediction at the centre, in standard deviations of the draws
    for b, nm in (('full_sherpa', 'SherpaFull'), ('full_herwig', 'HerwigFull')):
        if not EV[b]:
            continue
        d_ = EV[b]; cb_ = d_['compare']['before/joint/lin_flat']
        rb_, rm_ = _r(d_, 'before/joint/lin_flat', ('thrust', 'nch')), _r(d_, 'before/joint/lin_flat', ('means',))
        textmacro(f'Env{nm}BinMin', _pct(1 - rb_.max())); textmacro(f'Env{nm}BinMax', _pct(1 - rb_.min()))
        textmacro(f'Env{nm}MeanMin', _pct(1 - rm_.max())); textmacro(f'Env{nm}MeanMax', _pct(1 - rm_.min()))
        cm_ = max(cb_[f'corr_{g}']['max_abs_diff'] for g in ('thrust', 'nch')); textmacro(f'Env{nm}Corr', f'{cm_:.2f}')
        env[b] = dict(bins_narrower=[1 - rb_.max(), 1 - rb_.min()], means_narrower=[1 - rm_.max(), 1 - rm_.min()],
                      corr_max_abs_diff=cm_, bins_wider=int(np.sum(rb_ > 1)), n_bins=int(len(rb_)))
        fzb = f'{ENV}/envelope_test_MIXGEOV3S9_{b}{SFX}.npz'
        if os.path.exists(fzb):
            zb_ = zload(fzb); lab_ = [str(l) for l in zb_['labels']]; no_ = len(lab_)
            Yb_, ycb_ = zb_['Y_joint'][:, :no_], zb_['y_central'][:no_]
            ix_ = [lab_.index(l) for g in ('means', 'thrust', 'nch') for l in _grp(d_, g)]
            shb_ = float(np.max((np.abs(Yb_.mean(0) - ycb_)/Yb_.std(0, ddof=1))[ix_])); env[b]['shift_max'] = shb_
            textmacro(f'Env{nm}Shift', f'{shb_:.1f}')
    if SFX == '_lin':
        # round B21, the forms the text quotes: the largest difference of the bin correlations of the tune box over both
        # distributions; inside the window after imposing, the tune box (largest |ratio - 1|) and the full box (how much
        # narrower, signed as \EnvAfterOutFull*) apart, of which \EnvAfterWin is the larger; and each signed range of how
        # much narrower the linear bands are, as unsigned parts: the largest amount by which a linear band is wider than
        # the spread of the draws (Wider) and the largest by which it is narrower (Narrower), in percent, each 0 where
        # there is none, with the number of predictions where it is wider (WiderN)
        cbt_ = et['compare']['before/joint/lin_flat']
        textmacro('EnvTuneCorr', f"{max(cbt_[f'corr_{g}']['max_abs_diff'] for g in ('thrust', 'nch')):.2f}")
        rwt_, rwf_ = _after(et, True), _after(ef, True)
        textmacro('EnvAfterWinTune', _pct(np.max(np.abs(rwt_ - 1))))
        textmacro('EnvAfterWinFullMin', _pct(1 - rwf_.max())); textmacro('EnvAfterWinFullMax', _pct(1 - rwf_.min()))
        env['after_window'] = dict(tune_ratio=[float(rwt_.min()), float(rwt_.max())], full_ratio=[float(rwf_.min()), float(rwf_.max())])
        parts_ = [('FullBin', rb), ('FullMean', rm), ('AfterOutFull', ro)]
        parts_ += [(f'{nm}Bin', _r(EV[b], 'before/joint/lin_flat', ('thrust', 'nch')))
                   for b, nm in (('full_sherpa', 'SherpaFull'), ('full_herwig', 'HerwigFull')) if EV[b]]
        env['unsigned'] = {}
        for nm, r_ in parts_:
            wd_, nw_ = max(float(np.max(r_ - 1)), 0.0), max(float(np.max(1 - r_)), 0.0)
            textmacro(f'Env{nm}Wider', _pct(wd_)); textmacro(f'Env{nm}Narrower', _pct(nw_)); textmacro(f'Env{nm}WiderN', str(int(np.sum(r_ > 1))))
            env['unsigned'][nm] = dict(wider=wd_, narrower=nw_, n_wider=int(np.sum(r_ > 1)), n=int(len(r_)))
    if EV['tune_xifull']:
        rx = _r(EV['tune_xifull'], 'before/joint/lin_flat'); rxq = _r(EV['tune_xifull'], 'before/joint/xi_quad')
        textmacro('EnvXifullMin', _pct(rx.min())); textmacro('EnvXifullMax', _pct(rx.max()))
        textmacro('EnvXifullQuadMin', _pct(rxq.min())); textmacro('EnvXifullQuadMax', _pct(rxq.max()))
        cx = lambda k: max(EV['tune_xifull']['compare'][k][f'corr_{g}']['max_abs_diff'] for g in ('thrust', 'nch'))
        textmacro('EnvXifullCorr', f"{cx('before/joint/lin_flat'):.2f}"); textmacro('EnvXifullQuadCorr', f"{cx('before/joint/xi_quad'):.2f}")
    numbers['envelope'] = env

# ---- three numbers of the text from the propagation's own results (linear run only) -------------------------------
# App. C: the shift a_tau P of the calculation's power correction at the fixed point (calc_bins.py, lmx_calc_bins).
# The caption of Table 1: the closure width of two Gaussian samples with the paper's estimator (gauss_ref.py,
# lmx_gauss), at the caption's settings and full size. App. A.2: the parameter counts of the PFN cross-check
# (crosscheck_seeds_summary.json of lmx_pfn): the energyflow PFN as Keras counts it, and our event network with one
# output (the conditional PFN at K = 1 without its parameter network); the sentence says they are the same. And
# (round B15) the mean AUC over the five seeds of each of its four networks: the energyflow PFN, the plain PyTorch PFN,
# our K = 1 network with the package's ReLU (torchK1) and as the paper builds it (torchK1silu: SiLU, the sum over
# particles scaled), with the largest of their four standard deviations. A summary without torchK1silu is refused like
# a missing input: the sentence compares our network with SiLU against the package's ReLU, and only torchK1silu is it.
GAUSS_FULL = dict(n_events=40000, nbins=25, min_count=5, repetitions=20000, shift=0.05)
if SFX == '_lin':
    fcb, fgr, fpf = f'{D}/calc_bins_nlo_point.json', f'{D}/gauss_ref_lin.json', f'{D}/crosscheck_seeds_summary.json'
    if need(fcb, 'the calculation in the bins of the data (lmx_calc_bins: \\CalcShift)'):
        pc = jl(fcb).get('power_correction')
        if not pc:
            refuse(fcb, "no power_correction: not the rerun's calc_bins.py (\\CalcShift)")
        else:
            numbers['calc_shift'] = pc; textmacro('CalcShift', sig(pc['shift']))
    if need(fgr, 'the Gaussian reference of Table 1 (lmx_gauss: \\Gauss*)'):
        gr = jl(fgr); cs_ = gr.get('cases') or {}
        got = dict(n_events=gr.get('n_events'), nbins=gr.get('nbins'), min_count=gr.get('min_count'),
                   shift=cs_.get('shift', {}).get('shift_in_sd'))
        bad_ = [f'{k} = {v!r}, not {GAUSS_FULL[k]!r}' for k, v in got.items() if v != GAUSS_FULL[k]]
        if 'test' in gr or (gr.get('repetitions') or 0) < GAUSS_FULL['repetitions']:
            bad_.append(f'{gr.get("repetitions")} repetitions, not {GAUSS_FULL["repetitions"]} (a test)')
        if bad_:
            refuse(fgr, 'not the Gaussian reference of the caption at full size: ' + '; '.join(bad_))
        else:
            c0, c1 = cs_['null'], cs_['shift']
            numbers['gauss_reference'] = dict(null=c0, shift=c1, repetitions=gr['repetitions'], seed=gr['seed'])
            macro('GaussNullMean', c0['mean'], 2); macro('GaussNullSd', c0['sd'], 2); macro('GaussShiftMean', c1['mean'], 1)
    if need(fpf, 'the PFN cross-check of App. A.2 (lmx_pfn: \\PfnParams*, \\PfnAuc*)'):
        pf = jl(fpf); pe_ = pf.get('params_event_side') or {}; sm_ = pf.get('summary') or {}
        ef_, ours_, plain_ = (pf.get('params') or {}).get('ef'), pe_.get('torchK1'), pe_.get('torchplain')
        silu_, why_ = pe_.get('torchK1silu'), []
        if ef_ is None or ours_ is None:
            why_.append("no parameter counts (params of ef, params_event_side of torchK1): not the rerun's PFN job")
        if 'torchK1silu' not in sm_:
            why_.append("no torchK1silu, our network as the paper builds it (SiLU, the sum over particles scaled), which "
                        "App. A.2 compares with the package's ReLU: a PFN job of before round B15 (\\PfnAuc*)")
        elif silu_ is None:
            why_.append('no params_event_side of torchK1silu')
        for w_ in why_:
            refuse(fpf, w_)
        if not why_:
            numbers['pfn_params'] = dict(energyflow=ef_, ours_event_side_K1=ours_, plain_pytorch=plain_,
                                         ours_event_side_K1_silu=silu_, all_equal=len({ef_, ours_, plain_, silu_}) == 1)
            textmacro('PfnParamsEf', str(int(ef_))); textmacro('PfnParamsOurs', str(int(ours_)))
            if len({ef_, ours_, plain_, silu_}) > 1:
                print(f'PFN COUNTS DIFFER: energyflow {ef_}, our event network {ours_}, plain PyTorch {plain_}, '
                      f'our event network with SiLU {silu_} (App. A.2 says they are the same)')
            arms_ = dict(PfnAucEf='ef', PfnAucPlain='torchplain', PfnAucOursRelu='torchK1', PfnAucOurs='torchK1silu')
            for m_, k_ in arms_.items():
                macro(m_, sm_[k_]['mean'], 3)
            sdmax_ = max(sm_[k]['sd'] for k in arms_.values()); macro('PfnAucSdMax', sdmax_, 3)
            numbers['pfn_auc'] = dict({k: dict(mean=sm_[k]['mean'], sd=sm_[k]['sd'], n=sm_[k]['n']) for k in arms_.values()},
                                      sd_max=sdmax_)

# ---- the checks of the linear run, before anything is written ----------------------------------------
# The macros of MACROS.md: the 296 that main.tex uses, but \EnvXifull{Min,Max,Corr} (no counterpart, D1), the two
# pulls of E12, and those the rewrite needs (round B9): the draw counts, the boxes of each generator alone over its
# full ranges, \CalcShift, the Gaussian reference and the PFN counts, and the PFN AUCs (round B15), and those of the
# window under D3 and the unsigned forms of the envelope's signed ranges (round B21). Every one must be written; the
# four of COST_GPU only once the GPU benchmark exists (E9).
EXPECTED_LIN = set('''
AccAugACatStd AccAugACatStdSd AccAugACatStrict AccAugACatStrictSd AccAugAFacStd AccAugAFacStdSd AccAugAFacStrict
AccAugAFacStrictSd AccAugAUniStd AccAugAUniStrict AccAugBCatStd AccAugBCatStdSd AccAugBCatStrict AccAugBCatStrictSd
AccAugBFacStd AccAugBFacStdSd AccAugBFacStrict AccAugBFacStrictSd AccAugBUniStd AccAugBUniStrict AccAugCCatStd
AccAugCCatStdSd AccAugCCatStrict AccAugCCatStrictSd AccAugCFacStd AccAugCFacStdSd AccAugCFacStrict
AccAugCFacStrictSd AccAugCUniStd AccAugCUniStrict AccDenseHerwigCatStd AccDenseHerwigCatStdSd
AccDenseHerwigCatStrict AccDenseHerwigCatStrictSd AccDenseHerwigFacStd AccDenseHerwigFacStdSd
AccDenseHerwigFacStrict AccDenseHerwigFacStrictSd AccDenseHerwigUniStd AccDenseHerwigUniStrict AccDenseMixCatStd
AccDenseMixCatStdSd AccDenseMixCatStrict AccDenseMixCatStrictSd AccDenseMixFacStd AccDenseMixFacStdSd
AccDenseMixFacStrict AccDenseMixFacStrictSd AccDenseMixUniStd AccDenseMixUniStrict AccDenseSherpaCatStd
AccDenseSherpaCatStdSd AccDenseSherpaCatStrict AccDenseSherpaCatStrictSd AccDenseSherpaFacStd AccDenseSherpaFacStdSd
AccDenseSherpaFacStrict AccDenseSherpaFacStrictSd AccDenseSherpaUniStd AccDenseSherpaUniStrict AccFinalFacStdMax
AccFinalFacStdMin AccFinalFacStrictMax AccFinalFacStrictMin AccFinalUniStdMax AccFinalUniStdMin
CalcAlphaZero CalcAlphas
CalibAMax CalibAMin CalibBMax CalibBMin CalibCMax CalibCMin
CostCatCpuMin CostCatGpu CostFacCpu CostFacGpu CostLateCpu CostLateGpu CostRestCpu CostVarCatDays CostVarFacSec
CostVarLateHours CostWCatCpuMin CostWCatGpu CostWFacCpu
CovEightZWithMax CovEightZWithMin CovEightZWithoutMax CovEightZWithoutMin CovErrMax CovErrMin CovZWithMax
CovZWithMin CovZWithoutMax CovZWithoutMin
EnvAfterNorm EnvAfterOutFullMax EnvAfterOutFullMin EnvAfterOutTune EnvAfterWin EnvBootFullMax EnvBootFullMin
EnvBootTuneMax EnvBootTuneMin EnvFullBinMax EnvFullBinMin EnvFullCorrElse EnvFullCorrPivot EnvFullMeanMax
EnvFullMeanMin EnvFullShift EnvHerwigSd EnvLearnFull EnvLearnTune EnvNoiseCorr EnvNoiseSd EnvSherpaSd EnvTuneCorrNch
EnvTuneCorrThr EnvTuneSd EnvZMax EnvZMin
EnvNdraw EnvNref
EnvSherpaFullSd EnvSherpaFullBinMin EnvSherpaFullBinMax EnvSherpaFullMeanMin EnvSherpaFullMeanMax EnvSherpaFullCorr
EnvSherpaFullShift EnvHerwigFullSd EnvHerwigFullBinMin EnvHerwigFullBinMax EnvHerwigFullMeanMin EnvHerwigFullMeanMax
EnvHerwigFullCorr EnvHerwigFullShift
CalcShift GaussNullMean GaussNullSd GaussShiftMean PfnParamsEf PfnParamsOurs
PfnAucEf PfnAucPlain PfnAucOursRelu PfnAucOurs PfnAucSdMax
TbShapeWinBins TbShapeGenBeforeWinMin TbShapeGenBeforeWinMax TbShapeGenAfterWinMin TbShapeGenAfterWinMax
TbShapeLearnBeforeWinMin TbShapeLearnBeforeWinMax TbShapeLearnWinMin TbShapeLearnWinMax TbShapeTheoryWiderBins
TbShapeApartWin TbShapeVsCalcMax TbPwinSherpa TbPwinHerwig TbPwinGenAfterPercent TbPwinLearnPercent TbWinFracBelow
TbWinFracBelowSherpa TbWinFracBelowHerwig TbWinFracCalcApart TbPullTheoryTopMin TbPullTheoryTopMax TbDeficitLowMin
TbDeficitLowMax TbCalcBelowEdge
EnvTuneCorr EnvAfterWinTune EnvAfterWinFullMin EnvAfterWinFullMax EnvFullBinWider EnvFullBinNarrower EnvFullBinWiderN
EnvFullMeanWider EnvFullMeanNarrower EnvFullMeanWiderN EnvAfterOutFullWider EnvAfterOutFullNarrower EnvAfterOutFullWiderN
EnvSherpaFullBinWider EnvSherpaFullBinNarrower EnvSherpaFullBinWiderN EnvHerwigFullBinWider EnvHerwigFullBinNarrower
EnvHerwigFullBinWiderN
KappaMax KappaMin
LcMembersStrictAll LcRunsThreeStrictAll
MapBaryonMean MapBaryonRunMax MapBaryonRunMin MapBaryonZFar MapBaryonZNear MapBaryonZOverTwo MapNonbMeanMax
MapNonbMeanMin MapNonbRunMax MapNonbRunMin
MixHerwigPointsStd MixHerwigPointsStrict MixSherpaPointsStd MixSherpaPointsStrict
ObsDenseHerwigBtotStd ObsDenseHerwigBtotStrict ObsDenseHerwigMultStd ObsDenseHerwigMultStrict ObsDenseHerwigNbStd
ObsDenseHerwigNbStrict ObsDenseHerwigRhoHStd ObsDenseHerwigRhoHStrict ObsDenseHerwigStrangeStd
ObsDenseHerwigStrangeStrict ObsDenseHerwigThrustStd ObsDenseHerwigThrustStrict ObsDenseMixBtotStd
ObsDenseMixBtotStrict ObsDenseMixMultStd ObsDenseMixMultStrict ObsDenseMixNbStd ObsDenseMixNbStrict
ObsDenseMixRhoHStd ObsDenseMixRhoHStrict ObsDenseMixStrangeStd ObsDenseMixStrangeStrict ObsDenseMixThrustStd
ObsDenseMixThrustStrict ObsDenseSherpaBtotStd ObsDenseSherpaBtotStrict ObsDenseSherpaMultStd
ObsDenseSherpaMultStrict ObsDenseSherpaNbStd ObsDenseSherpaNbStrict ObsDenseSherpaRhoHStd ObsDenseSherpaRhoHStrict
ObsDenseSherpaStrangeStd ObsDenseSherpaStrangeStrict ObsDenseSherpaThrustStd ObsDenseSherpaThrustStrict
SingleMax SingleMean SingleMin
SvdRankA SvdRankB SvdRankC SvdRankMix SvdRankSherpa
TbAlephBelowTopMax TbAlephBelowTopMin TbAlephErrTopMax TbAlephErrTopMin TbBelowBandMax TbBelowBandMin
TbBinGenAfterAboveMax TbBinGenAfterAboveMin TbBinGenAfterWinMax TbBinGenAfterWinMin TbBinGenBeforeWinMax
TbBinGenBeforeWinMin TbBinLearnAboveMax TbBinLearnAboveMin TbBinLearnBeforeWinMax TbBinLearnBeforeWinMin
TbBinLearnWinMax TbBinLearnWinMin TbBinStatWinMax TbBinStatWinMin TbBinTheoryDownWinMax TbBinTheoryDownWinMin
TbBinTheoryUpTopMax TbBinTheoryUpTopMin TbBinTheoryUpWinMax TbBinTheoryUpWinMin TbBoxOffsetPartMax
TbBoxOffsetPartMin TbBoxOffsetShapesMax TbBoxOffsetShapesMin TbCalcBelowTopMax TbCalcBelowTopMin TbDeficitTopMax
TbDeficitTopMin TbHerwigAloneWin TbImposedApartWin TbImposedVsCalcMax TbLThreeBands TbLThreePull TbMaxWeightHerwig
TbMaxWeightMid TbMaxWeightSherpa TbNchImposed TbNodes TbOpalPull TbPullBelowMax TbPullGenOnly TbPullImposed
TbPullLowMax TbPullTopMax TbPullTopMin TbPullWithBand TbPwinChangePercent TbSchemeTopMax TbSchemeTopMin
TbSherpaAloneWin TbStatMeansMax TbTheoryTauWinDownPercent TbTheoryTauWinUpPercent TbTheoryWiderBins TbThrustComb
TbThrustImposed TbTrainSpreadMax TbTrainSpreadMin TbUpTopMax TbWinBins TbXiFixedMax TbXiFixedMin TbXiOnlyMax
TbXiOnlyMin
ToyCorr ToyDCorr ToyDSlope ToyKappa ToyQShortKOne ToySlope ToyUniform ToyWidthKOne ToyWidthMean ToyWidthScanMax
ToyWidthScanMin ToyWidthSd
TruncMaxPercent
WideLocalMax WideLocalMean WideLocalMin WidePoolMax WidePoolMean WidePoolMin WideSingleFiveMax WideSingleFiveMin
WideSingleMean WideSingleWorst
WifiLossFitOne WifiLossFitThree WifiLossFitTwo WifiLossMeanOne WifiLossMeanThree WifiLossMeanTwo WifiNewtonOne
WifiNewtonThree WifiNewtonTwo WifiWidthFitOne WifiWidthFitThree WifiWidthFitTwo WifiWidthMeanOne WifiWidthMeanThree
WifiWidthMeanTwo
XpNullMean
'''.split())
# a macro of EXPECTED_LIN that may come in another form, which then counts as written (round B10): \ToyKappa, or
# \ToyKappaMin and \ToyKappaMax when the three K = 48 toy trainings chose different kappa (main.tex 2174 changes)
EXPECTED_ALT = {'ToyKappa': ('ToyKappaMin', 'ToyKappaMax')}
PUBLISHED ='/mnt/beegfs/bassi/reweighting/maxent_code/'      # the published folder: never an input of the rerun (I4)
# the propagation's results in INDIR whose names do not end in _lin (lmx_calc_bins, lmx_pfn): like the *_lin files, the
# first submission's park moves any earlier copy aside (lmx_fingerprint.PROP_OUTPUTS), so in RUN they are of this rerun
PROP_RESULTS = ('calc_bins_nlo_point.json', 'crosscheck_seeds_summary.json')


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def data_table(real):
    """The one data file read, data_stagePURE17/meta.json, in its photon-free (_nb) folder (D12), which may lie inside
    the published maxent_code."""
    return os.path.basename(real) == 'meta.json' and re.search(r'_nb\d*$', os.path.dirname(real)) is not None


if SFX == '_lin':
    # I4: the scoring side's check, written by collect_lmx only when every scoring result carries the fingerprint of the
    # start mark and of the code. INDIR is RUN/output, and the check lists RUN-relative paths.
    RUN_DIR = os.path.dirname(os.path.abspath(D))
    fsc, fmark = f'{D}/scoring_check.json', f'{D}/.lmx_rerun_start'
    sc, mark = {}, {}
    if need(fsc, "the scoring side's check of its results (collect_lmx, I4)"):
        try:
            sc = json.load(open(fsc)); mark = json.load(open(fmark)) if os.path.exists(fmark) else {}
        except ValueError as e:
            refuse(fsc, f'not readable: {e}'); sc = {}
        if sc and (not sc.get('mark_run_id') or mark.get('run_id') != sc.get('mark_run_id')):
            refuse(fsc, f"made for the start mark {sc.get('mark_run_id')!r}, but {fmark} holds {mark.get('run_id')!r}")
    listed = {os.path.normpath(k): v for k, v in (sc.get('files') or {}).items()}
    if sc and not listed:
        refuse(fsc, "lists no file: collect_lmx writes it only with every scoring result (I4)")
    inputs = {}
    for p in sorted(READ):
        rel, real, h = os.path.relpath(os.path.abspath(p), RUN_DIR), os.path.realpath(p), md5(p)
        inputs[rel] = h
        if (real + '/').startswith(os.path.realpath(D) + '/published_copies/'):
            refuse(p, 'a published copy (output/published_copies), not a result of the rerun (I4)')
        elif real.startswith(PUBLISHED) and not data_table(real):
            refuse(p, f'it resolves into the published maxent_code ({real}), not a result of the rerun (I4)')
        elif not real.startswith(os.path.realpath(RUN_DIR) + '/') and not data_table(real):
            refuse(p, f'it resolves outside RUN ({real})')
        if rel in listed and listed[rel] != h:
            refuse(p, f"md5 {h}, but the scoring side's check lists {listed[rel]}: changed after collect_lmx (I4)")
        elif rel not in listed and not (p == META208 and data_table(real)) and not (
                (re.search(r'_lin\.(json|npz)$', os.path.basename(p)) or os.path.basename(p) in PROP_RESULTS)
                and os.path.dirname(os.path.abspath(p)) == os.path.abspath(D)):
            refuse(p, "not listed in the scoring side's check, and neither a result of the propagation (*_lin, "
                      "calc_bins_nlo_point.json or crosscheck_seeds_summary.json in INDIR) nor the data table: not a "
                      "result of this rerun (I4)")
    numbers['inputs'] = inputs
    written = {re.match(r'\\newcommand\{\\([A-Za-z]+)\}', m).group(1) for m in macros}
    missing = sorted(m for m in EXPECTED_LIN - written if not set(EXPECTED_ALT.get(m, (m,))) <= written)
    by_design = [m for m in missing if m in COST_GPU] if not os.path.exists(fg) else []
    if by_design:
        numbers['missing_by_design'] = dict(macros=by_design, why=f'no GPU benchmark {fg}: step 6c writes them (E9)')
        print(f'missing by design, without the GPU benchmark {fg}: ' + ' '.join(by_design))
    if len(missing) > len(by_design):
        refuse(os.path.join(OUTD, 'numbers_lin.tex'), 'macros of MACROS.md not written: ' +
               ' '.join('\\' + m for m in missing if m not in by_design))

head = '% written by phase2/make_numbers.py, do not edit\n'
if NUMBERS_TEST and (os.path.realpath(OUTD) + '/').startswith('/mnt/beegfs/bassi/reweighting/lmx_runs/'):
    sys.exit(f'NUMBERS_TEST=1 writes no test output under lmx_runs (E10): OUTDIR={OUTD}')
if REFUSED:                                      # E11, I4: the linear run refuses these inputs, and writes nothing
    why = '\n'.join(f'  {f}: {w}' for f, w in REFUSED)
    if not NUMBERS_TEST:
        sys.exit(f'make_numbers.py (SUFFIX=_lin): not the inputs of the full rerun (E11, I4), nothing written:\n{why}')
    print(f'NUMBERS_TEST=1: macros of inputs the linear run refuses (E11, I4), for tests only:\n{why}')
    numbers['test_only_refused_inputs'] = [dict(file=f, reason=w) for f, w in REFUSED]
    head += '% TEST ONLY (NUMBERS_TEST=1): made from inputs the linear run refuses, see numbers_lin.json\n'
os.makedirs(OUTD, exist_ok=True)
json.dump(numbers, open(os.path.join(OUTD, f'numbers{SFX}.json'), 'w'), indent=1)
open(os.path.join(OUTD, f'numbers{SFX}.tex'), 'w').write(head + '\n'.join(macros) + '\n')
print(f'{len(macros)} macros -> {os.path.join(OUTD, f"numbers{SFX}.tex")}, blocks:', ', '.join(numbers))
