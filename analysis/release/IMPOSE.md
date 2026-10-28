# Imposing a calculation on the reference sample

The MIX17 head gives every event of its reference sample a weight at any point of its 17
parameters. `gentune.maxent` then imposes a calculation of thrust moments on those weights,
as the paper does: the weights change as little as possible in relative entropy, within the
calculation's covariance, until the window means of the moments agree with it. It needs only numpy.
On the full sample it reproduces the paper's torch code (`analysis/directlib.py`) to about
1e-13 in log w, and the paper's central predictions to 1e-9.

## What you need

- numpy, and this package (`gentune`). Run the code below from `analysis/release` of a checkout,
  where `gentune/` and `models/` sit, or install the package with `pip install analysis/release`
  from the top of the checkout and give the paths to `models/` and the two files below.
- `models/MIX17_head.npz`, in this repository
- `reference_MIX17.npz`, the reference sample: 1,149,840 events, 865 MB (`REFERENCE.md`
  describes it). It holds `AE`, the event vectors the head reads, shape (4, N, 48), and the
  per-event observables `1_minus_thrust`, `B_total`, `rho_heavy`, `nch`, `nbaryon` and others.
- a targets file. `targets_thrust_nlo.npz` is the paper's NNLL+NLO calculation. You can also
  write your own (below).

The two files are public at https://github.com/benleo12/BDistcrete/releases/tag/data-r8, with
`REFERENCE.md` and their checksums, and will also be archived at Zenodo. In `analysis/release`:

```bash
for f in reference_MIX17.npz targets_thrust_nlo.npz REFERENCE.md SHA256SUMS; do
  curl -LO https://github.com/benleo12/BDistcrete/releases/download/data-r8/$f
done
shasum -a 256 -c SHA256SUMS          # sha256sum -c SHA256SUMS on Linux; every line must say OK
```

About 2 GB of memory is enough. On 8 cores one parameter point takes about 4 s for the head and
under 1 s for the imposition.

## Weights, the calculation, predictions

```python
import numpy as np
from gentune import load_head
from gentune.maxent import load_targets

head = load_head('models/MIX17_head.npz')
ref = np.load('reference_MIX17.npz')
A = head.pack(ref['AE'])                   # once: the event vectors as one (N, 192) matrix
tau = ref['1_minus_thrust']
calc = load_targets('targets_thrust_nlo.npz')
inwin, F = calc.functions(tau)             # once: the window and the 14 moment functions

theta = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18,                          # Sherpa
         0.102337, 0.654714, 3.528693, 1.849375, 0.914156, 0.374094, 0.33107, 0.78,  # Herwig
         0.5]                                                                      # fraction
logw0 = head.logit(A, theta)/head.T        # the generators alone
logw, lam, info = calc.impose(logw0, inwin, F)
w = np.exp(logw)                           # normalized, sums to one
print(w @ tau, w @ ref['nch'])             # 0.065110 and 17.8311
```

This `theta` is the paper's central point: the default tunes of both generators and a fraction
of one half. Without the calculation (`w = np.exp(logw0 - logw0.max())`, normalized) the same
two means are 0.064832 and 17.8146. Any mean is `w @ x` for a per-event array `x`. A histogram
is `np.histogram(x, bins, weights=w)`.

`info` holds the window means after the tilt and their pulls against the calculation, the window
fraction before and after, the relative entropy, the effective sample size `1/sum(w**2)`, and
whether the solve converged. For a scan over many points, pass the previous `lam` as `lam0=` to
start the solve there.

The variations of the calculation are in `calc.variations`, by name. Imposing each in turn gives
the theory band. With the paper's combination (the two scale pairs at half weight, the change
of scheme once) this reproduces its band on `<1-T>`, 0.001524:

```python
def mean(which):
    lw, _, _ = calc.impose(logw0, inwin, F, which=which)
    return np.exp(lw) @ tau

d = {v: mean(v) - mean('central') for v in calc.variations}
pairs = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"),
         ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)")]
band = np.sqrt(sum(0.5*d[a]**2 + 0.5*d[b]**2 for a, b in pairs) + d["(1.0, 1.0, 'modR', 1.0)"]**2)
```

The dispersive-model variations `milan_up`, `milan_dn` and `sub3` are not in that sum. The
paper's signed bands, which do include them, are built in `analysis/three_bands.py`.

## What a loaded calculation holds

`load_targets` returns a `gentune.maxent.Targets`, and everything it imposes can be read off it:

```python
print(calc)                     # Targets(14 moments, window [0.05, 0.3333), floor 0.002, 12 variations)
print(calc.names)               # ['logt', 'tau', 'tlogt', 'log2t', ...]: f_k(tau) = tau^a ln^b tau
print(calc.powers)              # [(0, 1), (1, 0), (1, 1), (0, 2), ...]: the (a, b) of each
print(calc.c)                   # the 14 calculated window means, calc.c[1] = <tau> = 0.112045
print(calc.sigma.shape)         # (14, 14): their covariance, here sigma_pert, before the floor
print(np.sqrt(np.diag(calc.sigma)))     # the error of each mean, <tau>: 0.00348
print(calc.window, calc.floor_rel)      # (0.05, 0.3333333333333333) 0.002
print(list(calc.variations))    # the 12 variations by name, "(2.0, 1.0, 'logR', 1.0)" ... 'sub3'
print(calc['milan_up'])         # the c of one variation; calc['central'] is calc.c
print(calc.meta)                # alpha_s 0.1135, alpha_0 0.4312 and the calculated window fraction
```

`c` and every variation are window-conditional means: the means of the functions over the
events with `lo <= tau < hi`, normalized inside the window. `impose` uses
`calc.sigma + diag((calc.floor_rel*c)**2)`, and `info['sigma_used']` is that matrix.

## A changed calculation: the paper's, shifted by 1 percent

`impose` takes any `c`, so a changed calculation needs no file. Here every moment of the paper's
calculation is made 1 percent larger:

```python
from gentune.maxent import impose

shifted = 1.01*calc.c
logw1, lam1, info1 = impose(logw0, inwin, F, shifted, calc.sigma, floor_rel=calc.floor_rel)
print(info1['pwin_before'], info1['pwin_after'])   # 0.4332 and 0.0059: the window has emptied
print(np.exp(logw1) @ tau)                         # 0.0298, which is not a prediction
```

`impose` warns about this result, and it is the useful lesson of the example. The fourteen
moments of one distribution are tied to each other, and the same factor on all of them is not
what any distribution of tau gives within the 0.2 percent floor, so no reweighting of the
reference events comes near it (`info1['pulls']` still reaches 5). The penalty of the problem is
weighted by the square of the window fraction, so the cheapest answer is then to move the
weight out of the window, from 0.433 to 0.006, and `<1-T>` over all events falls from 0.0651 to
0.0298. A shift by 0.1 percent still takes the window fraction to 0.16. A calculation that is a
distribution, the paper's or any of its twelve variations, leaves it at 0.4330. So change the
calculated distribution and take the moments of the changed one, as the variations do, rather
than the moments themselves, and check `info['pwin_after']` against `info['pwin_before']` on
anything you impose. Imposed by name, a changed `c` reads

```python
calc.variations['mine'] = calc.variations["(2.0, 1.0, 'logR', 1.0)"]   # any 14 window means
logw2, _, info2 = calc.impose(logw0, inwin, F, which='mine')
print(info2['pwin_after'], np.exp(logw2) @ tau)    # 0.4330 and 0.065349
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
inwin, F = mine.functions(tau)
logw, lam, info = mine.impose(logw0, inwin, F)
```

- `c` are means normalized inside the window: c_j is the integral of f_j(tau) dsigma/dtau over
  lo <= tau < hi, divided by the integral of dsigma/dtau over the same range. The fraction of
  events in the window is not imposed. Only the shape inside the window is.
- The functions are tau^a ln^b tau, named `tau`, `tau2`, `logt`, `log2t`, `tlogt`, `t2log2t`
  and so on. The paper uses the fourteen in `gentune.maxent.THRUST_MOMENTS`. With
  `powers=[(a, b), ...]` the names can be any labels.
- `Sigma` is a k by k covariance. Use `np.diag(sd**2)` for uncorrelated errors. The paper's is a
  sum of d d^T with d = c_v - c over the five scale variations of the band above, with the same
  weights.
- `impose` adds a floor of 0.2 percent of each moment to the diagonal of `Sigma`, as the
  paper does. Set `floor_rel` in the `Targets` (or the file) to change it, 0 for none. Keep it
  for the paper's calculation: its `Sigma` has rank five, so with no floor nine combinations of
  the fourteen moments would be imposed exactly, the problem is singular in float64 (condition
  number near 1e21), and the solve is not reliable.
- A variation is another `c`. It keeps the same `Sigma`, with the floor taken on the varied `c`.
- Every `c` has to be the moments of one distribution of tau. Moments changed one by one, or all
  by the same factor, are not, and the solve then empties the window (the section on the 1
  percent shift above). `impose` warns when the window fraction changes by more than 10 percent
  (the paper's calculation and its variations change it by less than 0.1 percent); small pulls do
  not mean agreement when that warning fires.

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
- For functions of another form, compute them yourself and call `impose` directly:
  `inwin` is a boolean mask of the events in your window, `F` is an (n_window, k) array of your
  functions on those events, in event order, and `impose(logw0, inwin, F, c, Sigma)` does the rest.
- An observable the reference file does not carry cannot be computed from it. The file holds the
  event vectors and the listed observables, not the particles.
- Numpy gives the weights and the predictions, not derivatives through the solve. For fits that
  need gradients, use the torch code in `analysis/directlib.py`. For a few derivatives, finite
  differences with `lam0` warm starts work.
- The head is only to be trusted inside its training box. `head.check(theta)` raises outside it
  and warns outside the region that was sampled.

## Checks

`python tests/test_maxent.py [targets_thrust_nlo.npz]` checks the solution on made-up events
with numpy alone.
`python tests/compare_directlib.py targets_thrust_nlo.npz --reference reference_MIX17.npz` runs
the same problems through the paper's torch code, on made-up events and on the 2000 reference
events of `models/MIX17_selftest.npz`. It is not numpy only: it needs torch, scipy and the
paper's `analysis/directlib.py` with the modules it imports, so a full clone of the repository.

`python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz`
reproduces the paper from the two deposited files with numpy alone and prints PASS or FAIL.
At the central point it gives the seven means of the paper (the window mean of 1 - T and the
window fraction, then <1-T>, <B_T>, <rho_H>, <n_ch> and <n_baryon>) before and after the
calculation is imposed, and the DELPHI thrust distribution. Over the paper's 200 points of a
Latin hypercube (seed 11) it gives the generator band, before and after. It compares
everything with the paper's numbers, which it carries, and agrees to a few 1e-8 relative or
better: the means to 1e-9, the distribution to 7e-9 and the bands to 1.2e-8. What remains
comes from the head, which the paper's code evaluated in float32 and gentune evaluates in
float64. It takes about six minutes on 8 cores and 2.5 GB of memory.

The bands of the generator alone, before a calculation is imposed, come from the same file with
the two band scripts of the README, and agree with the paper as closely: the generator band to
1e-9 on the means and 1.2e-8 on the bins, the learning band to 1e-5, and the statistical errors
to 1e-9.

```bash
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper
```
