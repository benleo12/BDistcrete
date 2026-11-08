#!/usr/bin/env python3
"""The Gaussian reference of the caption of Table 1 (main.tex): "Two samples of 4x10^4 events from one Gaussian
density, in 25 bins, give 1.01 +- 0.15, and 1.7 when their means differ by five percent of the standard deviation."

It computes these numbers with the closure width of Sec. 3.2 exactly as the paper's scorings compute it, by calling
r2_ladder.py's own functions (as the closure, the uniform-weight control and the calibration check of run_stage do):
 * the bins: make_bins(np.r_[a, b], obs) for a continuous observable: NBIN = 26 edges, so 25 bins, evenly spaced
   between the 0.5 and 99.5 percentiles of the two samples together;
 * the pulls: pulls(a, w, b, bins), the first sample as the reference side with uniform weights w = 1/N (as the
   uniform-weight control and the calibration check give it), the second as the fresh run: each bin's variance from
   the pooled density and the two sides' own counts, and a bin used only if both sides hold at least MIN_COUNT = 5;
 * the width: width_of(p) = sqrt(sum p^2/(n - 1)) over the n bins used.
Two cases: both samples standard Gaussian (null), and the second shifted by SHIFT = 0.05 standard deviations (shift).
Each case draws REPS pairs of samples of N = 40000 events with numpy's default_rng, seeded with SeedSequence(SEED)
spawned into one stream per case, so the result is fixed. It reports, per case, the mean and the standard deviation
(ddof = 1) of the width over the repetitions, the standard error of the mean, the median and the bins used.
REPS = 20000 puts the standard error of the null mean near 0.001, well inside the printed digits (two decimals for
the null mean and its standard deviation, one for the shifted mean).

    python -B gauss_ref.py           # in RUN: writes output/gauss_ref_lin.json (OUT names another file)

GAUSS_REPS (tests only) sets fewer repetitions; make_numbers.py refuses a file of fewer than 20000 or of other
settings than the caption's. The job lmx_gauss of RERUN_PROP.md (step 8) runs it; it takes a few minutes on one core.
"""
import hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_ladder as R                                   # the paper's estimator, not a copy of it

N, SHIFT, SEED, FULL_REPS = 40000, 0.05, 20261031, 20000
REPS = int(os.environ.get('GAUSS_REPS', str(FULL_REPS)))
OUT = os.environ.get('OUT', 'output/gauss_ref_lin.json')
assert (R.NBIN, R.MIN_COUNT) == (26, 5), (R.NBIN, R.MIN_COUNT)      # the binning and occupancy rule of the paper
assert 'gauss' not in R.DISCRETE


def width(rng, shift):
    a = rng.standard_normal(N); b = rng.standard_normal(N) + shift
    bins = R.make_bins(np.r_[a, b], 'gauss')
    p, nb = R.pulls(a, np.full(N, 1.0/N), b, bins)
    return R.width_of(p), nb, len(bins) - 1


t0 = time.time()
cases = {}
for (name, shift), ss in zip((('null', 0.0), ('shift', SHIFT)), np.random.SeedSequence(SEED).spawn(2)):
    rng = np.random.default_rng(ss)
    w, used, nbins = np.empty(REPS), np.empty(REPS, int), set()
    for i in range(REPS):
        w[i], used[i], n = width(rng, shift); nbins.add(n)
    cases[name] = dict(shift_in_sd=shift, mean=float(w.mean()), sd=float(w.std(ddof=1)), se_mean=float(w.std(ddof=1)/np.sqrt(REPS)),
                       median=float(np.median(w)), bins=sorted(nbins), bins_used_min=int(used.min()),
                       bins_used_mean=float(used.mean()))
    print(f'{name:5s} (means {shift} sd apart): width {w.mean():.4f} +- {w.std(ddof=1):.4f} (se of the mean '
          f'{cases[name]["se_mean"]:.4f}), median {np.median(w):.4f}, bins used {used.min()}-{used.max()} of {sorted(nbins)}', flush=True)
src = os.path.join(HERE, 'r2_ladder.py')
res = dict(what='the Gaussian reference of the caption of Table 1: the closure width of two Gaussian samples',
           estimator='r2_ladder.make_bins, pulls (uniform weights 1/N on the first sample), width_of',
           r2_ladder_md5=hashlib.md5(open(src, 'rb').read()).hexdigest(), n_events=N, nbins=R.NBIN - 1,
           min_count=R.MIN_COUNT, repetitions=REPS, seed=SEED, rng='numpy default_rng, SeedSequence(seed).spawn(2): null, shift',
           cases=cases, seconds=round(time.time() - t0, 1))
if REPS < FULL_REPS:
    res['test'] = f'GAUSS_REPS={REPS}: fewer repetitions than {FULL_REPS} (make_numbers.py refuses it)'
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
tmp = f'{OUT}.tmp{os.getpid()}'
json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)
print(f'null {cases["null"]["mean"]:.2f} +- {cases["null"]["sd"]:.2f}, shift {cases["shift"]["mean"]:.1f} -> {OUT}')
print(f'GAUSS REF DONE ({res["seconds"]:.0f} s)')
