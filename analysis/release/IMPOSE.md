# Imposing the QCD rate and shape

The paper uses two steps, separately for Sherpa and Herwig before mixing them.
A soft maximum-entropy fit constrains fourteen conditional thrust moments in
`0.05 <= 1-T < 1/3`. A second normalization sets the event fraction in this
window to the calculated rate. The second step preserves the fitted shape
inside the window and the conditional distribution outside it.

This is a sequential prescription. The rate update minimizes relative entropy
with respect to the shape-fitted distribution subject to the rate constraint;
it is not a joint soft fit of fifteen constraints. Theory variations change
the moments and rate together, using the same varied QCD calculation.

## Inputs

Run from `analysis/release` of a checkout, with numpy and these files:

- `models/MIX17_head.npz` and `models/MIX17_wifi.npz`;
- `reference_MIX17.npz`, the deposited 1,149,840 reference events;
- `targets_thrust_rate_shape.npz`, the current NNLL+NLO calculation with a
  dispersive power correction.

Join the reference parts and copy the targets from the repository's deposit:

```bash
cat ../../deposit/reference_MIX17.npz.part-* > reference_MIX17.npz
cp ../../deposit/targets_thrust_*.npz .
grep -v build_reference ../../deposit/SHA256SUMS | shasum -a 256 -c
```

Every listed file must report OK. `REFERENCE.md` describes the reference,
particle definitions and checksums. The older `targets_thrust_nlo.npz` omits
an explicit rate constraint and remains available for the archived shape-only
prescription.

## Weights and predictions

```python
import numpy as np
from gentune import load_head, axes, mixture
from gentune.maxent import load_targets

head = load_head('models/MIX17_head.npz')
ref = np.load('reference_MIX17.npz')
A = head.pack(ref['AE'], np.float64)          # once; about 1.8 GB
calc = load_targets('targets_thrust_rate_shape.npz')
tau, nch = ref['1_minus_thrust'], ref['nch']
inwin, F = calc.functions(tau)
theta = axes.default('MIX17')                # both default tunes, xi = 1/2
w = mixture.weights(head, A, theta)
wc, info = mixture.impose(w, inwin, F, calc)

print(mixture.mean(wc, tau, theta))          # 0.0658609
print(mixture.mean(wc, nch, theta))          # 18.0409
print(mixture.mean(wc, inwin, theta))        # 0.44858922375303223
```

`w` and `wc` are dictionaries with `sherpa` and `herwig` arrays, each normalized
to one over all reference events. The calculation is imposed on each generator
alone; `mixture.mean` combines their predictions with the fraction in `theta`.
A conditional mean uses `mask=inwin`. A histogram uses
`mixture.histogram(bin_index, wc, bin_width, theta)`. Factors for a measurement's
particle definition are applied per generator before mixing (`README.md`).

For each generator, `info[g]` records the shape multipliers `lam`, convergence
and residual of the shape solve, `pwin_before`, `pwin_shape`, `rate_target` and
`pwin_after`. It also records the final effective sample size, relative entropy
and the two rate factors. The shape residual describes the first step; the
second step does not solve for new shape multipliers. For nearby points,
`lam0={g: info[g]['lam'] for g in info}` supplies warm starts.

## Uncertainty propagation

The generator band comes from repeating both steps at every parameter point.
For statistical and learning uncertainties, pass the solution so that the
response includes the shape fit and the rate normalization:

```python
imp = (inwin, F, info)
print(mixture.mean_and_error(wc, tau, theta, imposed=imp))
print(mixture.learning_band(head, A, theta, tau,
                           'models/MIX17_wifi.npz', imposed=imp))
```

The learning variance is averaged over uniform `xi` by default; use
`xi_prior=None` for the fraction at `theta`. The two generators share reference
events and learning coefficients, so their covariance is retained when mixing.
The full-window fraction has no generator, learning or reference-statistical
variation at fixed QCD input: it is an imposed number. Its theory uncertainty
remains. Agreement with that input is therefore not an independent test.

Impose each named theory variation to propagate its paired moments and rate:

```python
from gentune import bands

labels = [v for pair in bands.SCALE_PAIRS for v in pair]
labels += [bands.SCHEME] + list(bands.DISPERSIVE)
d = {g: {} for g in wc}
for label in labels:
    wv, _ = mixture.impose(w, inwin, F, calc, which=label)
    for g in wc:
        d[g][label] = (wv[g] - wc[g]) @ tau
up = bands.theory_band_mix(d['sherpa'], d['herwig'], up=True)
down = bands.theory_band_mix(d['sherpa'], d['herwig'], up=False)
```

This applies the paper's signed scale, scheme and dispersive combination at
each `xi`, then averages the variance over uniform `xi`. It retains the
correlation between the QCD rate and shape. Do not add another independent rate
error to this band.

## A calculation of your own

A `Targets` object contains moment names, conditional means `c`, their
covariance `sigma`, a window and, optionally, a rate and paired variations:

```python
from gentune.maxent import Targets

j = [calc.names.index(n) for n in ('tau', 'tau2', 'logt')]
labels = ["(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"]
mine = Targets(
    ['tau', 'tau2', 'logt'], calc.c[j], calc.sigma[np.ix_(j, j)],
    window=calc.window,
    variations={v: calc[v][j] for v in labels},
    rate=calc.rate,
    rate_variations={v: calc.rate_variations[v] for v in labels},
)
mine.save('my_targets.npz')                  # JSON also supported
inwin3, F3 = mine.functions(tau)
w3, info3 = mixture.impose(w, inwin3, F3, mine)
```

`c[k]` is the mean of the corresponding function normalized inside the window.
The functions are `tau**a * log(tau)**b`, named `tau`, `tau2`, `logt`, etc.;
`powers=[(a,b), ...]` allows other labels. Every variation must be the moments
of a coherent varied distribution. Changing individual moments arbitrarily
can make the constraints incompatible.

The soft shape covariance is `sigma + diag((floor_rel*c)**2)`, with
`floor_rel=0.002` by default. The paper's covariance has rank five, so its floor
is needed to regularize the fourteen-moment solve. A variation uses the same
covariance with its own central values in the floor.

`rate=None` requests a shape-only fit. With a rate, every moment variation must
have a matching entry in `rate_variations`; both are applied by `which=label`.
The NPZ or JSON format stores these fields explicitly, and old target files
without a `rate` field retain their shape-only behavior. A `pwin` entry in
legacy metadata alone does not activate rate matching.

## Checks and reproduction

```bash
python tests/test_maxent.py
python tests/test_mixture.py
python tests/test_rate_matching.py
python examples/reproduce_paper.py --reference reference_MIX17.npz \
    --targets targets_thrust_rate_shape.npz --points 0
```

The first tests cover the shape solver and mixture. The rate test checks
normalization, preserved conditional distributions, paired variations,
finite-difference derivatives and file round trips. The reproduction command
checks central predictions and their statistical, learning and theory
components against `examples/paper_numbers.json` to relative tolerance 1e-6.
Omit `--points 0` to evaluate all 2000 generator points as well.

`analysis/check_rate_matching.py` independently compares all 2000 thrust
predictions with algebraically corrected archived draws and checks directional
derivatives on the full reference sample. The old
`tests/compare_directlib.py targets_thrust_nlo.npz` checks the archived
shape-only solver against the original torch implementation; that torch path
does not include the new rate correction.
