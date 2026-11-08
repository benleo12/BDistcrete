# Imposing a calculation on the reference sample

The MIX17 head gives every event of its reference sample a weight at any point of its 17
parameters, for each generator: the head at the fraction 0 is Sherpa, at 1 Herwig
(`gentune.mixture`, README). `gentune.maxent` then imposes a calculation of thrust moments on
those weights, as the paper does: the weights change as little as possible in relative entropy,
within the calculation's covariance, until the window means of the moments agree with it. The
paper imposes the calculation on each generator alone and then mixes the two imposed predictions,
(1 - xi) <O>_S + xi <O>_H, and `gentune.mixture.impose` does the same. It needs only numpy. On the
two deposited files it gives the paper's numbers to 3.6e-8 relative (`examples/reproduce_paper.py`),
and each generator's imposed log weights agree with the paper's torch code (`analysis/directlib.py`)
to about 1e-12 on all reference events (`tests/compare_directlib.py`).

## What you need

- numpy, and this package (`gentune`). Run the code below from `analysis/release` of a checkout,
  where `gentune/` and `models/` sit, or install the package with `pip install analysis/release`
  from the top of the checkout and give the paths to `models/` and the two files below.
- `models/MIX17_head.npz`, in this repository
- `reference_MIX17.npz`, the reference sample: 1,149,840 events, 865 MB (`REFERENCE.md`
  describes it). It holds `AE`, the event vectors the head reads, shape (4, N, 48), and the
  per-event observables `1_minus_thrust`, `B_total`, `rho_heavy`, `nch`, `nbaryon` and others,
  all without the photons radiated by the beams.
- a targets file. `targets_thrust_nlo.npz` is the paper's NNLL+NLO calculation. You can also
  write your own (below).

The two files are in `deposit/` at the top of the repository, with `build_reference.json`, the
record of the build of the reference sample, and `SHA256SUMS`, the sha256 of the three files.
GitHub refuses files over 100 MB, so the reference sample is stored there in 19 parts. From
`analysis/release`, join the parts, copy the targets file and check both:

```bash
cat ../../deposit/reference_MIX17.npz.part-* > reference_MIX17.npz
cp ../../deposit/targets_thrust_nlo.npz .
grep -v build_reference ../../deposit/SHA256SUMS | shasum -a 256 -c
```

`shasum` must print OK for both files (on Linux, write `sha256sum -c` in place of
`shasum -a 256 -c`). `SHA256SUMS` also lists `build_reference.json`, which stays in `deposit/`, so
`grep -v` leaves its line out. The two files are then in `analysis/release`, and git ignores them.
If a line says FAILED, do not use that file: run the three lines again, and if it fails again,
clone the repository again.

On a MacBook Pro with an Apple M3 Max (14 cores, 36 GB) and numpy 2.5.4, the first block below takes
3 seconds, with a peak of 2.9 GB of memory, and every further point 0.52 seconds for the weights of
both generators and the calculation imposed on each.

## Weights, the calculation, predictions

```python
import numpy as np
from gentune import load_head, axes, mixture
from gentune.maxent import load_targets

head = load_head('models/MIX17_head.npz')
ref = np.load('reference_MIX17.npz')
A = head.pack(ref['AE'], np.float64)         # once: the event vectors as one float64 matrix
tau, nch = ref['1_minus_thrust'], ref['nch']
calc = load_targets('targets_thrust_nlo.npz')
inwin, F = calc.functions(tau)               # once: the window and the 14 moment functions

theta = axes.default('MIX17')                # the paper's central point
w = mixture.weights(head, A, theta)          # {'sherpa': w_S, 'herwig': w_H}, each sums to one
wc, info = mixture.impose(w, inwin, F, calc) # the calculation imposed on each generator alone
print(mixture.mean(wc, tau, theta), mixture.mean(wc, nch, theta))   # 0.0626911 and 17.8001
print(mixture.mean(w, tau, theta), mixture.mean(w, nch, theta))     # 0.0620683 and 17.7781
```

This `theta` is the paper's central point: the default tunes of both generators and a fraction
of one half (the table in the README). The first line prints `<1-T>` and `<n_ch>` with the
calculation imposed, the second the same without it. Any mean is `mixture.mean(wc, x, theta)` for a
per-event array `x`, and a mean over part of the events, such as that of 1 - T inside the window,
is `mixture.mean(wc, tau, theta, mask=inwin)`: each generator's own mean there, mixed. A
distribution is `mixture.histogram(b, wc, width, theta)` with `b = bands.bin_index(x, lo, hi)`,
each bin the fraction of all events in it over its width. `mixture.combine` mixes any pair of
per-generator results, and multiplies each generator's by its own factor first if you pass one,
such as the factor that takes a prediction to the hadron level of a measurement.

`info` holds, for each generator (`info['sherpa']`, `info['herwig']`), the window means after the
tilt and their pulls against the calculation, the window fraction before and after, the relative
entropy, the effective sample size `1/sum(w**2)`, whether the solve converged and in how many
Newton steps, and the multipliers `lam`. For a scan over many points, pass the previous
multipliers, `lam0={g: info[g]['lam'] for g in info}`, to start each solve there.

The statistical error of a prediction with the calculation imposed is not that of fixed weights:
the paper solves for the multipliers again on every resample of the events, which takes out of
each event's term its projection on the moment functions. Pass the solution:

```python
imp = (inwin, F, info)
print(mixture.mean_and_error(wc, tau, theta, imposed=imp))               # <1-T> and its error
print(mixture.mean_and_error(wc, tau, theta, mask=inwin, imposed=imp))   # 1 - T in the window
print(mixture.learning_band(head, A, theta, tau, 'models/MIX17_wifi.npz', imposed=imp))
```

The last line is the learning band of `<1-T>` with the calculation imposed, its derivative taken
through the multipliers solved again (README, The learning band of MIX17). Without `imposed`,
both are those of the generators alone.

## The theory band

The variations of the calculation are in `calc.variations`, by name. Imposing each in turn on
each generator gives its displacements, and the paper's theory band combines them:
each of the two scale pairs once, with the larger of its two displacements in each direction, the
change of scheme once, and the three variations of the dispersive model as their mean square. For
the mixture the paper applies this at each xi to the displacement (1 - xi) d_S + xi d_H of the
mixed prediction and averages its square over xi flat on [0, 1]:

```python
from gentune import bands

labels = [l for pair in bands.SCALE_PAIRS for l in pair] + [bands.SCHEME] + list(bands.DISPERSIVE)
d = {g: {} for g in wc}
for lab in labels:
    wv, _ = mixture.impose(w, inwin, F, calc, which=lab)
    for g in wv:
        d[g][lab] = wv[g] @ tau - wc[g] @ tau      # how generator g moves under this variation
up = bands.theory_band_mix(d['sherpa'], d['herwig'], up=True)
down = bands.theory_band_mix(d['sherpa'], d['herwig'], up=False)
print(up, down)                                    # 0.0002277 and 0.001474
```

`bands.theory_band(d['sherpa'])` is the band of Sherpa alone. Any per-event array, or a histogram
of each generator, can take the place of `wv[g] @ tau`.

## What a loaded calculation holds

`load_targets` returns a `gentune.maxent.Targets`, and everything it imposes can be read off it:

```python
print(calc)                     # Targets(14 moments, window [0.05, 0.3333), floor 0.002, 12 variations)
print(calc.names)               # ['logt', 'tau', 'tlogt', 'log2t', ...]: f_k(tau) = tau^a ln^b tau
print(calc.powers)              # [(0, 1), (1, 0), (1, 1), (0, 2), ...]: the (a, b) of each
print(calc.c)                   # the 14 calculated window means, calc.c[1] = <tau> = 0.111864
print(calc.sigma.shape)         # (14, 14): their covariance, here sigma_pert, before the floor
print(np.sqrt(np.diag(calc.sigma)))     # the error of each mean, <tau>: 0.00348
print(calc.window, calc.floor_rel)      # (0.05, 0.3333333333333333) 0.002
print(list(calc.variations))    # the 12 variations by name, "(2.0, 1.0, 'logR', 1.0)" ... 'sub2'
print(calc['milan_up'])         # the c of one variation; calc['central'] is calc.c
print(calc.meta)                # alpha_s 0.1137, alpha_0 0.524 and the calculated window fraction
```

`c` and every variation are window-conditional means: the means of the functions over the
events with `lo <= tau < hi`, normalized inside the window. `impose` uses
`calc.sigma + diag((calc.floor_rel*c)**2)`, and `info[g]['sigma_used']` is that matrix.

## A changed calculation: the paper's, shifted by 1 percent

`impose` takes any `c`, so a changed calculation needs no file: give `mixture.impose` the pair
`(c, Sigma)`. Here every moment of the paper's calculation is made 1 percent larger:

```python
shifted = (1.01*calc.c, calc.sigma)
w1, info1 = mixture.impose(w, inwin, F, shifted, floor_rel=calc.floor_rel)
print(mixture.mean(w, inwin, theta), mixture.mean(w1, inwin, theta))   # 0.4114 and 0.0054: the window has emptied
print(mixture.mean(w1, tau, theta))                                    # 0.0289, which is not a prediction
print(max(abs(i['pulls']).max() for i in info1.values()))              # 5.4, the largest pull
```

`impose` warns about this result, and it is the useful lesson of the example. The fourteen
moments of one distribution are tied to each other, and the same factor on all of them is not
what any distribution of tau gives within the 0.2 percent floor, so no reweighting of the
reference events comes near it, and the pulls stay large. The penalty of the problem is weighted
by the square of the window fraction, so the cheapest answer is then to move the weight out of
the window: the first line prints the window fraction before and after, and `<1-T>` over all events
falls with it. A shift by 0.1 percent still takes the window fraction to 0.151:

```python
w01, _ = mixture.impose(w, inwin, F, (1.001*calc.c, calc.sigma), floor_rel=calc.floor_rel)
print(mixture.mean(w01, inwin, theta))                                 # 0.151
```

A calculation that is a distribution, the paper's or any of its twelve variations, leaves the
window fraction where it was. So change the calculated distribution and take the moments of the
changed one, as the variations do, rather than the moments themselves, and check
`info[g]['pwin_after']` against `info[g]['pwin_before']` on anything you impose. Imposed by name, a
changed `c` reads

```python
calc.variations['mine'] = calc.variations["(2.0, 1.0, 'logR', 1.0)"]   # any 14 window means
w2, info2 = mixture.impose(w, inwin, F, calc, which='mine')
print(mixture.mean(w2, inwin, theta), mixture.mean(w2, tau, theta))    # 0.4108 and 0.0629043
```

## Your own calculation

A calculation is a window in tau = 1 - T, k functions of tau, their calculated means `c` and
the covariance `Sigma` of those means:

```python
from gentune.maxent import Targets

# for the example, three of the paper's moments with their part of its covariance; your own
# calculation puts its numbers here
j = [calc.names.index(n) for n in ('tau', 'tau2', 'logt')]
c, Sigma = calc.c[j], calc.sigma[np.ix_(j, j)]
c_up, c_down = calc["(2.0, 1.0, 'logR', 1.0)"][j], calc["(0.5, 1.0, 'logR', 1.0)"][j]

mine = Targets(['tau', 'tau2', 'logt'], c, Sigma, window=(0.05, 1/3),
               variations={'muR up': c_up, 'muR down': c_down})
mine.save('my_targets.npz')                # or .json; read back with load_targets
inwin3, F3 = mine.functions(tau)
w3, info3 = mixture.impose(w, inwin3, F3, mine)
print(mixture.mean(w3, tau, theta))
```

- `c` are means normalized inside the window: c_j is the integral of f_j(tau) dsigma/dtau over
  lo <= tau < hi, divided by the integral of dsigma/dtau over the same range. The fraction of
  events in the window is not imposed. Only the shape inside the window is.
- The functions are tau^a ln^b tau, named `tau`, `tau2`, `logt`, `log2t`, `tlogt`, `t2log2t`
  and so on. The paper uses the fourteen in `gentune.maxent.THRUST_MOMENTS`. With
  `powers=[(a, b), ...]` the names can be any labels.
- `Sigma` is a k by k covariance. Use `np.diag(sd**2)` for uncorrelated errors. The paper's is a
  sum of d d^T with d = c_v - c over five scale variations: the two scale pairs, each variation at
  weight 1/2, and the change of scheme at weight 1 (`REFERENCE.md`).
- `impose` adds a floor of 0.2 percent of each moment to the diagonal of `Sigma`, as the
  paper does. Set `floor_rel` in the `Targets` (or the file) to change it, 0 for none. Keep it
  for the paper's calculation: its `Sigma` has rank five, so with no floor nine combinations of
  the fourteen moments would be imposed exactly, the problem is singular in float64 (condition
  number near 1e21), and the solve is not reliable.
- A variation is another `c`. It keeps the same `Sigma`, with the floor taken on the varied `c`.
- Every `c` has to be the moments of one distribution of tau. Moments changed one by one, or all
  by the same factor, are not, and the solve then empties the window (the section on the 1
  percent shift above). `impose` warns when the window fraction changes by more than 10 percent
  (the paper's calculation and its variations change it by well under one percent for each
  generator); small pulls do not mean agreement when that warning fires.

The file is an `.npz` or `.json` with the fields `moment_names`, `window`, `central`, `sigma`,
and optionally `moment_powers`, `floor_rel` and `variations` (in `.npz`, `variation_names` and
`variation_c`). `targets_thrust_nlo.npz` has the same fields, except that it calls the
covariance `sigma_pert` and stores its variations under `scale_labels`/`scale_variations` and
`np_labels`/`np_variations`. `load_targets` reads both.

## Limits

- One observable and one window per call. `functions(x)` builds x^a ln^b x for whatever
  per-event array you pass, so moments of another observable in the reference file work the
  same way: give the `Targets` a window in that observable and call, for example,
  `functions(ref['rho_heavy'])`.
- For functions of another form, compute them yourself and call `gentune.maxent.impose` on one
  generator's log weights, or `mixture.impose` with the pair `(c, Sigma)`: `inwin` is a boolean
  mask of the events in your window, `F` is an (n_window, k) array of your functions on those
  events, in event order.
- An observable the reference file does not carry cannot be computed from it. The file holds the
  event vectors and the listed observables, not the particles.
- Numpy gives the weights, the predictions, their statistical errors and their learning bands,
  not derivatives with respect to the generator parameters. For fits that need those, use the
  torch code in `analysis/directlib.py`. For a few derivatives, finite differences with `lam0`
  warm starts work.
- The head is only to be trusted inside its training box. `head.check(theta)` raises outside it
  and warns outside the region that was sampled.

## Checks

`python tests/test_maxent.py [targets_thrust_nlo.npz]` checks the solution on made-up events, and
`python tests/test_mixture.py [reference_MIX17.npz]` the mixture, the bands and the beam-photon
rule on the self-test events, both with numpy alone. The command below runs the same problems
through the paper's torch code, on made-up events and on reference events, the maximum-entropy step
and the mixture with its errors and bands:

```bash
python tests/compare_directlib.py targets_thrust_nlo.npz --reference reference_MIX17.npz --linear-events 0
```

With `--linear-events 0` the mixture part uses all the reference events; without it, the 2000
events of the self-test. It is not numpy only: it needs torch, scipy and the paper's
`analysis/directlib.py` with the modules it imports, so a full clone of the repository. It takes a
few minutes and about 6 GB of memory.

`python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz`
reproduces the paper from the two deposited files with numpy alone and prints PASS or FAIL. At
the central point it gives the seven means of the paper (the mean of 1 - T in the window and the
window fraction, then <1-T>, <B_T>, <rho_H>, <n_ch> and <n_baryon>) and DELPHI's thrust
distribution in its twenty bins, before and after the calculation is imposed, with their
statistical errors, learning bands and, after it, the theory band. Over the 2000 points of the
paper's Latin hypercube (seed 11) it gives the generator band, before and after. It gives OPAL's
`<1-T>` and DELPHI's distribution with the factors of the measurement as well
(`examples/data/convention_factors.json`, README, Comparing with a measurement), with their errors
and bands. It compares every number with the paper's, in `examples/paper_numbers.json`, and passes
when each agrees to 1e-6 relative (`--tol`). On the same machine (What you need, above), the largest
difference was 3.6e-8, and the run took 18 minutes and 4.1 GB of memory; `--points 0`, the central
point alone, takes 14 seconds.

The two band scripts of the README give the same bands from the same file, with the same
functions:

```bash
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper --targets targets_thrust_nlo.npz
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper --targets targets_thrust_nlo.npz
```
