# Continuous generator-parameter reweighting

This is the released code and trained networks of *Combined Uncertainties from Event Generators
and Precision QCD* (Assi and Thaler). It answers one question, per event, without running a
generator:

> I have an event sample from Sherpa or Herwig. What weights turn it into the sample I would
> have got with different hadronization and shower parameters?

A trained classifier gives the log-ratio of the two event densities, so the weights are exact
in the limit that the classifier is exact, and the accuracy is measured rather than assumed.
The point is that the parameters move **continuously**. A conventional shower or hadronization
uncertainty is a handful of discrete alternative tunes, and there is no way to interpolate
between them, to put a prior on them, or to condition them on data. Here the parameters are the
argument of a function.

Everything here is numpy. There is no deep learning framework to install and no version of one
to match.

## Start here

Four steps, from nothing to a checked install:

```
git clone https://github.com/benleo12/BDistcrete
cd BDistcrete && python3 -m venv .venv && source .venv/bin/activate
pip install numpy
cd analysis/release && python -m gentune.selftest models
```

numpy is the only dependency (`pyproject.toml`: Python 3.9 or later, numpy 2.0 or later; 2.0 is the
oldest numpy the release was tested with). The clone downloads about 1 GB, most of it the paper's
reference sample (What is here, below), and takes about 2000 MB on disk.
The self-test takes 3 seconds on a MacBook Pro with an Apple M3 Max (14 cores, 36 GB) and numpy
2.5.4. Every command below runs from `analysis/release`, and the scripts in `examples/` import
`gentune` from the folder next to them, so nothing else needs installing. If the self-test does not
print PASS for the stage you intend to use, stop.

Two tests check the rest of `gentune` with numpy alone: the mixture, the bands and the beam-photon
rule on the events of the self-test, and the maximum-entropy step on made-up events. Each prints
PASS or FAIL for every check, and `all checks pass` when none fails:

```
python tests/test_mixture.py
python tests/test_maxent.py
```

Then weight a sample, here the Herwig example file, from the default tune to a larger diquark
weight:

```
python examples/reweight_own_events.py --stage MIX17 --hepmc examples/data/herwig_default.hepmc.gz \
    --from herwig-default --to herwig-default H:pwtdiquark=0.40 --out weights.npz
```

`weights.npz` holds `weights`, one per event in the order of the file, so `weights[i]` is the
weight of the i-th event of the file, and they sum to one: multiply by the number of events for
weights that average one, as a histogram filled event by event expects. `event_number[i]` is the
number that event carries in the file (Herwig counts from 1, Sherpa from 0). Next to them are both
points, the parameter names, and the per-event observables of `gentune.bands.observables` (Bands,
below), `per_event` with one column per name in `observables`. For `MIX17` there are also the
weights of each generator whose share of the mixture is not zero, `weights_sherpa` and
`weights_herwig` (Two generators at once, below).

`--from` is the point the sample was generated at and `--to` the point you want. Each is either
one number per parameter, in the order the listing prints, or a preset followed by the
parameters to change, as `name=value`. `sherpa-default` and `herwig-default` are the generators'
own defaults, the values a card that sets nothing gets, and `default` is the paper's central
point (below). For `MIX17` a sample made by one generator sits at an edge of the fraction, 0 for
Sherpa and 1 for Herwig, and `--from` must be such a point. The presets put the other generator's
parameters at the centre of their box, where the training runs had them; they do not enter the
weights.

With `--list`, or without `--hepmc`, it prints the parameters, their ranges and defaults, and the
defaults as card lines (`python examples/reweight_own_events.py --stage MIX17 --list`). A point
with the wrong number of values, or outside the box, is refused with the name, value and range of
each parameter that is wrong.

To look at what a head does across one parameter, with the 2000 events that ship with it:

```
python examples/parameter_scan.py --stage E --axis alphas
```

## What the self-test checks

`python -m gentune.selftest models` has two stages, and a check of the beam-photon rule between
them.

The first stage runs the trunk and the head over the 2000 reference events that every released stage
ships, and compares with the logits that the shipped files produce for them. The stored events
hold the six per-particle columns already constructed, so this stage exercises the two networks
and the final column mapping but not the HepMC parsing or the thrust frame that produced those
columns. The training run's own logits are stored alongside as `logit_probe_from_export`, with the
largest difference between the two as `precision_gap`, so that gap is on the record rather than
hidden. Each line also prints `dA(vs training)`, the largest difference between the event vectors
the shipped trunk computes from the stored features and those the training run exported. It is
printed for the record; the logits are the test. The line of `MIX17` prints a small `N_eff/N` and
a note under it: its stored probe pair moves only the fraction, from 0.5 to 0.9, through the
network's own interpolation between the generators, so most of the reference events lose their
weight. That is expected and is not a failure. The other stages move one parameter and keep
`N_eff/N` near one.

Between the two stages, the rule that finds the photons radiated by the beams
(`gentune.beam_photons`) runs on the stored events of every stage, and the self-test checks what
the rule states, event by event. The networks read every particle, but the paper leaves these
photons out of every observable (below).

The second stage covers the rest of the path. It reads the two example files in `examples/data/`, 30
Herwig 7.3.0 events in HepMC2 and 40 Sherpa 3.0.4 events in HepMC3, both at the generators'
default tunes and made with the training cards. It runs them through the parsing, the thrust
frame, the trunk and the `MIX17` head, and compares the features and the weights of one parameter
pair per file with `examples/data/expected.npz`. The comparison allows for the thrust axis, which
comes from an iteration that can end on the other of two nearly tied axes when the last bit of a
dot product differs between machines. So the columns that do not depend on the axis must agree in
every event to 1e-5, at most 5 percent of the events may move their axis, and every other event
must reproduce its log weight to 0.01. `examples/data/README.md` says how the files were made and
where these tolerances come from. If the example files are not there, this stage says which are
missing and is skipped, and the results above it stand.

`E`, `F` and `MIX17` were trained on per-particle features stored as float16, and their trunk
files say so (`feature_precision`). `Trunk` rounds your features the same way before the network
sees them, which reproduces the training inputs. Read in float32 instead, the reweighted means of
`MIX17` would move by up to 0.17 percent, against an accuracy floor of 0.15 percent (Accuracy,
below), and those of `E` and `F` by 0.02 percent (`analysis/precision_gap.py`).

## Installing the package alone

The package also installs with pip, without a checkout:

```
pip install 'git+https://github.com/benleo12/BDistcrete#subdirectory=analysis/release'
```

This installs `gentune` and numpy and nothing else. The networks in `models/`, the example files
in `examples/data/` and the scripts in `examples/` are not part of the package, so they still come
from a checkout, and the installed package is pointed at them by path, here from the top of the
checkout:

```
python -m gentune.selftest analysis/release/models analysis/release/examples/data
```

Without the second argument the second stage of the self-test is skipped, because by default it
looks for the example files next to the installed package, where they are not. From a checkout,
`pip install analysis/release` at the top of it installs `gentune` from there.

## The two things you can ask for, which are not the same

```python
from gentune import Head, Trunk, features

head  = Head('models/E_head.npz')
trunk = Trunk('models/E_trunk.npz')

F, M, _ = features.features_from_events(features.parse_hepmc('examples/data/sherpa_default.hepmc.gz'))
A = trunk.embed(features.trunk_features(F), M)

theta_from = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18]   # where the sample was made
theta_to   = [0.120, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18]
w = head.ratio_weights(A, theta_to, theta_from)   # your sample, generated at theta_from
print(head.n_eff(w))
```

`ratio_weights` is almost always the one you want. The classifier learned the log-ratio of each
parameter point to a common reference, which is the pooled union of all the training runs. The
difference of two of those logits is the log-ratio between the two points with the reference
cancelled exactly, so it applies to a sample generated at either point and needs no reference
events at all. Pass the parameters your sample was actually generated at as `theta_from`.

`head.weights(A, theta)` reweights the reference sample itself. It is correct only on reference
events, which means the events shipped with the export or events you generated from the same
pooled configuration. Using it on a sample from a single parameter point is a mistake that
produces plausible looking and wrong numbers. The 2000 events of every
`models/<stage>_selftest.npz` are reference events, drawn from that stage's reference sample, and
`reference_index` gives the place of each one in it, so `weights` is correct on them. Their event
vectors are stored as `AE`.

Both return weights normalized to sum to one over the events you passed in, in the order you
passed them: `w[i]` belongs to the event of row `i` of `A`. `parse_hepmc` returns the events in
the order of the file, with the event number of each in `.numbers`, so for a HepMC file `w[i]` is
the weight of its i-th event, numbered `events.numbers[i]` there. Use `head.n_eff(w)` to see how
many events are effectively carrying the result. When that falls well below the event count the
weights are still correct, and the errors on anything computed from them grow accordingly.

## Two generators at once: MIX17

`MIX17` holds Sherpa and Herwig in one network, so that "which generator" stops being a discrete
choice you cannot interpolate and becomes a number you can put a prior on: the fraction xi of
Herwig. A point is the two generators' parameters followed by the fraction,
`theta = (*theta_S, *theta_H, xi)`, with xi in [0, 1]. The paper's prediction at that point mixes
the predictions of the two generators,

    <O>(theta)  =  (1 - xi) <O>_S(theta_S)  +  xi <O>_H(theta_H) ,

as if each event came from Sherpa with probability 1 - xi and from Herwig with probability xi.
Each generator's prediction comes from its own weights on the same events: the head at xi = 0 is
the log ratio of Sherpa at theta_S, and at xi = 1 that of Herwig at theta_H, and each set of
weights sums to one. `gentune.mixture` does this:

```python
import numpy as np
from gentune import load_head, axes, bands, mixture

head = load_head('models/MIX17_head.npz')           # the right class for any export
sb = np.load('models/MIX17_selftest.npz')          # 2000 reference events of MIX17
A = head.pack(sb['AE'], np.float64)                # their event vectors, converted once
obs = bands.observables(sb['particles'], sb['mask'].astype(bool))
theta = axes.default('MIX17')                      # both default tunes, xi = 1/2
w = mixture.weights(head, A, theta)                # {'sherpa': w_S, 'herwig': w_H}
print(mixture.mean_and_error(w, obs['n_ch'], theta), mixture.n_eff(w))
```

`mixture.mean(w, O, theta)` gives the mixed mean of any per-event array `O`, and
`mixture.histogram` the mixed distribution, each bin the fraction of all events in it over its
width. A mean over part of the events, such as the mean of 1 - T inside the window of a
calculation, is each generator's own mean there, mixed with the same (1 - xi, xi):
`mixture.mean(w, O, theta, mask=...)`. A calculation is imposed on each generator alone and the
two imposed predictions are then mixed (`mixture.impose`, [`IMPOSE.md`](IMPOSE.md)).
`mixture.event_weights(w, theta)` gives one weight per event, (1 - xi) w_S + xi w_H, for a
histogram filled event by event; its mean over all events is the mixed mean, its mean over part of
the events is not. A generator whose coefficient is zero, Herwig at xi = 0 or Sherpa at xi = 1, is
left out.

For a sample of your own, made by one generator, pass the point it was made at as `theta_from`,
with the fraction 0 for Sherpa or 1 for Herwig: `mixture.weights(head, A, theta, theta_from=src)`
then gives the ratio weights of each generator against your sample, with `A` from `MIX17`'s own
trunk. A head can take the vectors of another stage's trunk with no error, and the weights are
then wrong. On a sample of one generator, the other generator's prediction comes from reweighting
its events across the generators, which costs effective sample size; `mixture.n_eff(w)` gives it
for each generator.

The head at an interior xi has a value of its own. `head.logit(A, theta)` there is the network's
interpolation `(1 - xi) l_S + xi l_H` between the log ratios `l_S` and `l_H` of the two generators,
and `head.weights(A, theta)` the weights of the density q_S^(1 - xi) q_H^xi it stands for. The
network was trained on runs at xi = 0 and 1 only, and the paper does not use it in between, so
`weights` and `ratio_weights` warn there (`GeometricFractionWarning`). The edges are exact rather
than approximate: at xi = 0 the head is `l_S` itself and its derivative with respect to `theta_H`
vanishes identically, and the same holds at xi = 1 for `theta_S`. `split()` divides a parameter
vector into its three pieces, and the block sizes `head.nS` and `head.nH` are read from the
exported networks rather than assumed.

A mixture head inherits the accuracy of both sides, so the weaker of the two sets the floor, and
`STAGES.md` reports it measured rather than inferred.

## The learning band of MIX17

The paper's learning band comes from the WiFi fit of Benevedes and Thaler (arXiv:2506.00113).
For one generator at its own fraction, its basis functions are the cross terms of the M = 4
members, the event vector of member `i` against the parameter vector of member `j`, plus a
constant,

    log r(Phi, theta) = w_0 + sum_ij w_ij < a_i(Phi), b_j(theta) > ,

with `b_j` the parameter network of Sherpa at xi = 0 and that of Herwig at xi = 1.
`models/MIX17_wifi.npz` holds the fitted coefficients `w` (`w_0` first, then `w_ij` row by row) and
their covariance `C`. The prediction uses the mean of the members in the same basis, `w_ij` equal
to 1/(M T) on the diagonal and zero elsewhere, with T the head's calibration constant. The learning
band of one generator's weighted mean `<O>_g` is `sqrt(g C g)`, where `g`, the derivative of the
mean with respect to `w`, is the weighted covariance of `O` with the basis functions. The mixed
prediction has the derivative `(1 - xi) g_S + xi g_H`, and as in the paper its square is averaged
over xi flat on [0, 1],

    band^2 = ( g_S C g_S + g_H C g_H + g_S C g_H ) / 3 ,

so both generators enter whatever the xi of the point. With a calculation imposed, the derivative
also runs through the multipliers solved again. `analysis/wifi_fit.py` makes the fit and
`analysis/three_bands.py` propagates it.

```python
wifi = 'models/MIX17_wifi.npz'
print(mixture.learning_band(head, A, theta, obs['n_ch'], wifi))   # the mixture, xi flat on [0, 1]
th_S = mixture.at_fraction(theta, 0.0)                               # Sherpa alone at theta_S
B = head.wifi_basis(A, th_S)                                         # (events, 1 + M^2)
print(head.learning_band(A, th_S, obs['n_ch'], wifi))                # sqrt(g C g) of Sherpa alone
```

Column 0 of the basis is the constant and column `k = 1 + i*M + j` is `< a_i(Phi), b_j(theta) >`,
with `i` the member of the event vector and `j` the member of the parameter vector, both counted
from zero. That is the order of `w`. `O` may be one value per event or one column per observable.
`mask` restricts each generator's mean to a subset of the events, normalized over it, which gives
the band of a mean inside a window. `theta_from` takes the ratio weights of a sample of your own,
and `counts` multiplies the weights event by event, which with `gentune.bands.resample_counts` is
a bootstrap. `xi_prior=None` takes the band at the xi of the point instead of averaging over it.
With `imposed=(inwin, F, info)` of `mixture.impose` it is the band with the calculation imposed;
without it, the band of the generators alone.

## Bands

The generator band and the learning band of the paper come from two scripts. On the paper's
reference sample, `reference_MIX17.npz`, joined from its parts in `deposit/` (below), they give the
paper's bands, on its observables and in its thrust bins, of the generators alone, and with
`--targets` with the paper's calculation imposed. They are bands at the hadron level of these
samples, without the factors of a measurement (Comparing with a measurement, below), so they are not
compared with data as they are:

```
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper --targets targets_thrust_nlo.npz
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper --targets targets_thrust_nlo.npz
```

On the same machine (Start here, above), the generator band takes 5 minutes, and 22 minutes with
`--targets`, with at most 3.1 GB of memory; the learning band takes 5 seconds and 4.7 GB. On the
2000 reference events that ship with the networks, without the full sample, the generator band
takes 1.7 seconds and the learning band 0.14 seconds, with statistical errors 24 times larger:

```
python examples/generator_band.py --stage MIX17 --reference
python examples/learning_band.py  --stage MIX17 --reference
```

`generator_band.py` reweights the events to the 2000 points of the paper's generator band, a
Latin hypercube over the whole box with seed 11 and the fraction uniform on [0, 1]. They are built
as `analysis/three_bands.py` builds them and are the paper's points to the last digit
(`gentune.bands.lhs_nodes`). For `MIX17` the prediction at every point is the mixture above, and
with `--targets` the calculation is imposed on each generator alone at every point before the
mixing. For every observable it prints the mean over the points and the band, which is the
standard deviation of the prediction over the points, with its 16th and 84th percentiles, and with
`--targets` the band before the calculation next to it. Next to them are the statistics that limit
the band. `stat. error` is the median over the points of the statistical error of the prediction
at one point (with `--targets` taken at every tenth point, which is enough for a median).
`--bootstrap N` resamples the events N times with replacement and reweights every resample to
every point, which gives `boot sd`, the bootstrap standard deviation of the prediction at a point
(the median over the points), and `boot sd of band`, that of the band itself. It runs by default,
with 100 resamples, for `--reference` and `--hepmc`, is off by default for `--reference-file`, and
is never taken with `--targets`, which would have to impose the calculation again on every
resample. A band whose statistical error at a point is as large as the band, or whose bootstrap
standard deviation is more than a third of it, is marked `*` and named in a warning as not
significant, and a second warning says when the median `N_eff` over the points is below 1000. It
also prints the median, 5th percentile and smallest `N_eff/N` over the points, for each
generator, and how many of the points lie where the training design reached (`head.support()`).
`--nodes` and `--seed` change the hypercube.

`learning_band.py` evaluates the learning band at one point, `--at`, by default the paper's
central point, averaged over xi flat on [0, 1] (`--xi point` takes it at the xi of `--at`).
`--at` is one number per parameter, or a preset followed by `name=value` changes, as `--to` of
`reweight_own_events.py`, for example `--at default H:pwtdiquark=0.40`. For each observable it
prints the value at the point, the learning band and the statistical error, and with
`--bootstrap N` (not with `--targets`) the bootstrap standard deviation of the band. The WiFi coefficients ship for
`MIX17` only, so this script runs on `MIX17` only.

**Where the events come from.** `--reference-file FILE` takes the paper's reference sample and
reweights it with the weights of the reference, which are exact there. `--reference` takes the
2000 reference events of `models/<stage>_selftest.npz` in the same way. For `MIX17` they are drawn
from the reference sample of 1,149,840 events, so every statistical error from them is about
`sqrt(1149840/2000)`, 24 times, that of the paper. They check the machinery; a band to quote needs
the full reference sample. A bin of a distribution holds about a hundred of the 2000 events, so the
band of a bin from them is mostly statistical noise. `--hepmc FILE --from POINT` takes a sample of your own
and reweights it with ratio weights, `POINT` written as for `--at`. On a sample of your own the
learning band is that of the log ratio between `--from` and `--at`, so for one generator it
vanishes when the two are the same point. The paper quotes the bands only on the reference sample.

**Which stage.** `--stage F` (Herwig) or `--stage E` (Sherpa) gives the band of one generator's
own parameters, the band within that generator. `--stage MIX17` gives the paper's band, which
also sweeps the fraction from 0 to 1 and so includes the choice between Sherpa and Herwig. On a
sample of one generator it does that by reweighting the events toward the other generator, which
costs effective sample size, as `N_eff/N` of each generator shows. For a Herwig sample at its
default tune the two are

```
python examples/generator_band.py --stage F     --hepmc examples/data/herwig_default.hepmc.gz --from herwig-default
python examples/generator_band.py --stage MIX17 --hepmc examples/data/herwig_default.hepmc.gz --from herwig-default
```

and for a Sherpa sample `--stage E` and `--from sherpa-default`, with
`examples/data/sherpa_default.hepmc.gz`. Your own file goes in place of the example file.

**The observables.** With `--reference-file` they are the paper's own, read from the file: `1-T`
on the exact thrust axis, `B_T`, `rho_H`, `n_ch` (among the 100 hardest particles, as in the
paper), `n` and `n_baryon` (the whole event). With `--reference` and `--hepmc` they are built from
the per-particle features by `gentune.bands.observables`: `1-T`, `n_ch`, `n` and `n_baryon`, each
over the at most 100 particles the features keep. Both leave out the photons radiated by the
beams, as the paper does in every observable (below). `1-T` from the features is read off the
iterative thrust axis the features are built on, which puts its mean above that of the exact axis
(`generators/README.md`). These names are the values `--hist` takes, `B_T` and `rho_H` with
`--reference-file` only.

**The bins.** `--bins EDGE ...` gives the band of every bin of the distribution of `--hist`, `1-T`
by default. `--bins paper` gives the paper's: DELPHI's thrust distribution at 91.2 GeV from
charged and neutral particles (Rivet `DELPHI_1996_S3430090`, `d11-x01-y02`), all twenty of its
bins, with the edges

    0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20 0.25 0.30 0.35 0.40 0.50

kept as the paper's code reads them, rounding included (`gentune.bands.PAPER_THRUST_BINS`), so
every event falls in the paper's bin. Giving these twenty-one numbers to `--bins` gives the same
bins up to that rounding. Each bin is the fraction of all events in it over its width, as in the
paper, which normalizes no distribution over a range of bins. The paper compares with the
measurement as published, which integrates to 0.990 over these bins.

**What `--out` holds.** Both scripts write an npz with the per-event observables (`per_event`,
one column per name in `observables`) and the event numbers (`event_number`: the number on the
event's line in the HepMC file, or for reference events the index in the reference sample), next
to the points, the predictions and their errors, `N_eff`, the histograms and the bootstrap;
`generator_band.py --targets` adds the predictions before the calculation (`means_before`,
`hists_before`).
`generator_band.py --save-weights` adds the weights at every point, one row per point, each row in
the order of the events and summing to one; for `MIX17` they are (1 - xi) w_S + xi w_H, so the band
of the mean of any other per-event quantity `x` of the same events is
`(weights @ x).std(ddof=1)`. The pieces are in `gentune.bands` and `gentune.mixture` for scripts of
your own, which need not import anything from `examples/`: `lhs_nodes`, `observables`,
`reference_observables`, `mean_and_error`, `bin_index`, `histogram`, `resample_counts`,
`resampled_sums`, `theory_band`, `theory_band_mix` and `PAPER_THRUST_BINS` in `bands`, and
`weights`, `mean`, `mean_and_error`, `histogram`, `impose`, `learning_band`, `event_weights` and
`n_eff` in `mixture`.

The paper's central point is both generators at their own defaults with a fraction of one half,
`gentune.axes.default('MIX17')` and the `CENTRAL` of `analysis/symmetry/three_bands.sbatch`:

| block | parameters, in the order of the head |
|---|---|
| Sherpa 3.0.4 | `alphas` 0.118, `pt_max` 0.68, `alpha_l` 3.9, `gamma_l` 0.48, `strange_fraction` 0.46, `baryon_fraction` 0.17, `alpha_g` 0.97, `beta_l` 0.18 |
| Herwig 7.3.0 | `alpha_fsr` 0.102337, `ptmin` 0.654714, `clmax` 3.528693, `clpow` 1.849375, `psplit` 0.914156, `pwtsquark` 0.374094, `pwtdiquark` 0.33107, `clsmr` 0.78 |
| fraction | 0.5 |

**Against the paper.** [`examples/reproduce_paper.py`](examples/reproduce_paper.py) runs all of
this on the two deposited files, the central predictions before and after the calculation with
their statistical errors, learning bands and theory band, and the generator band over the 2000
points before and after, and compares every number with the paper's (`examples/paper_numbers.json`,
copied from the output of `analysis/three_bands.py`):

```
python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz
```

It prints the largest relative difference of each comparison and PASS or FAIL against `--tol`, 1e-6
by default. On the same machine (Start here, above), the largest difference was 3.6e-8, and the run
took 18 minutes and 4.1 GB of memory; with `--points 0`, the central point alone, 14 seconds. What
is left comes from the head, which the paper's code evaluates in float32 and gentune in float64. In
the statistical errors and learning bands with the calculation imposed, it also comes from the
paper's code, which inverts the matrix of the Newton step without rescaling it.

**Comparing with a measurement.** A measurement defines its own hadron level. DELPHI, for one,
decays K0S, Lambda and the other hyperons (c*tau between 10 and 300 mm), which these samples keep.
`examples/data/convention_factors.json` holds the factors that take each generator's prediction
from the hadron level of these samples to that of a measurement, per bin of DELPHI's thrust
distribution and for the means, measured with each generator at its default tune. Each entry has
several ratios, and `meta.application` of the file says which goes with which measurement: DELPHI's
distribution takes `c_over_a`, OPAL's `<1-T>` takes `cnu_over_a` (the same decays, with the
neutrinos counted too), and L3's `<n_ch>` takes none. The file is a copy of the paper's
`analysis/conventions/factors.json`, which `analysis/conventions/make_factors.py` made. As in the
paper, each generator's prediction is multiplied by its own factors after the calculation is
imposed and before the mixing (`factors=` of `gentune.mixture`), and `reproduce_paper.py` compares
OPAL's `<1-T>` and DELPHI's distribution with the factors too.

## Imposing your own calculation

[`IMPOSE.md`](IMPOSE.md) describes how to impose a calculation, its moments and their
covariance, on the weights of the `MIX17` reference sample with `gentune.maxent` and
`gentune.mixture`, in numpy, both for the paper's thrust calculation and for one of your own, how
to read a loaded calculation, and what a changed one has to satisfy. It needs the reference sample
and the paper's calculation, the two files below, which [`REFERENCE.md`](REFERENCE.md) describes.

## The photons radiated by the beams

The incoming electron and positron radiate photons along the beam, and the generators write them
into the event record. The paper leaves them out of every observable it computes from the
particles of an event: 1 - T, the broadening, the heavy jet mass, the multiplicity of all
particles and the kaon fraction among them. The networks still read every particle, as they were
trained, so the weights are unchanged. The stored particles do not carry their origin, so a rule
finds these photons from the features alone: a massless neutral particle that lies in the plane of
the thrust axis and the beam, with three refinements for photons on the axis itself and for
flags that disagree (`REFERENCE.md`). `gentune/beam_photons.py` holds the rule with its statement
and its self-test, a copy of the paper's `analysis/beam_photons.py`, and `REFERENCE.md` says how
often it acts and where it can be wrong. For your own observables, use the particles
`M & ~beam_photons.beam_photon_mask(F, M)`; `gentune.bands.observables` does this.
`generators/compute_efps.py` removes the same photons before every event shape it writes.

## Being inside the box is not the same as having been sampled

```python
head.check(theta)          # raises outside the box, warns outside the sampled region
head.support(theta)        # score, threshold, ok, nearest
```

`box` is the range of each parameter. `support` is the question of whether the head ever saw a
point like yours, and the two can come apart. Both eight-parameter designs fill their boxes. Each
design holds the points of the earlier design of its generator and 256 points of a scrambled Sobol
sequence over the whole box. The first Herwig design follows the published retunes, which move the
shower coupling with the shower cutoff and the cluster fission mass with its power, and covers a
six-dimensional locus. The Sobol points fill the rest. `check` still tells you when a point is far
from every training point, which `in_box` alone does not, and `generator_band.py` prints how many
of its points pass `support`. A point outside the box, or with the wrong number of values, raises a
`ValueError` that names the index, the name, the value and the range of each parameter at fault.

## Conventions you have to match

A head gives the weights for samples made the way its training samples were made. Your sample
has to match on the following.

- **The process and the energy.** e+e- -> gamma*/Z -> q qbar at sqrt(s) = 91.2 GeV, the Z pole,
  with the shower and hadronization of the generator. No other energy was trained on.
- **The generators.** Sherpa 3.0.4 and Herwig 7.3.0 (the CVMFS build 7.3.0p1, with ThePEG
  2.3.0), pinned in `generators/cvmfs_env.sh`.
- **The cards.** The ones below, from `generators/gen_stageE.sh` and `generators/gen_stageF.sh`,
  with the parameters at the defaults of each generator. Two settings in them are not defaults:
  Sherpa's `ORDER_ALPHAS: 1` (the default is 2) and Herwig's `MaxLifeTime 10*mm` (the default is
  -1, no cut).
- **The stable hadrons.** Every hadron with `c*tau` above 10 mm is left undecayed, in both
  generators (below).
- **The units.** Momenta and masses in GeV. `features.parse_hepmc` reads the units line that
  HepMC writes after every event line, `U GEV MM` or `U MEV MM`, and converts a MeV file to GeV,
  which gives the same features as the GeV file bit for bit. A file without a units line is read
  as GeV and then checked: if its charged pions do not come out within a factor of ten of their
  mass, it raises a `ValueError` that says what was assumed. `parse_hepmc(path, units='MEV')`
  then reads it, and so does the file with the units line added. The example scripts take the
  same as `--units GEV` or `--units MEV`.
- **The file format.** `features.parse_hepmc` reads HepMC2 and HepMC3 text files, plain or gzipped,
  and takes the format from the file's own header, so it also reads the HepMC2 files that Herwig 7.3
  writes through HepMC3. A header that does not tell raises a `ValueError`.
  `parse_hepmc(path, fmt='hepmc2')` or `fmt='hepmc3'` then forces it, and the example scripts take
  the same as `--fmt`.
- **Unweighted events.** Every event counts once, as in the training samples, and the event
  weights in the file are ignored. A file whose weights are all equal is unweighted and needs
  nothing. With these cards Herwig writes a weight of 1 on every event and Sherpa one constant
  weight, so both are. If the weights differ between events, `parse_hepmc` warns once per file,
  and the weights gentune returns must then be multiplied by the file's own, event by event, and
  renormalized, or the sample unweighted first.

The Sherpa run card, `Sherpa.yaml`, at Sherpa's default tune:

```yaml
EVENT_OUTPUT: HepMC3_GenEvent[events]
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
ALPHAS(MZ): 0.118
ORDER_ALPHAS: 1
SHOWER_GENERATOR: CSS
FRAGMENTATION: Ahadic
AHADIC:
  PT_MAX: 0.68
  ALPHA_L: 3.9
  GAMMA_L: 0.48
  STRANGE_FRACTION: 0.46
  BARYON_FRACTION: 0.17
  ALPHA_G: 0.97
  BETA_L: 0.18
PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
```

with `EVENTS` and `RANDOM_SEED` added. The Herwig input file at Herwig's default tune:

```
read snippets/EECollider.in
cd /Herwig/MatrixElements
insert SubProcess:MatrixElements 0 MEee2gZ2qq
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 91.2
set /Herwig/Shower/AlphaQCDFSR:AlphaIn 0.102337
set /Herwig/Shower/PTCutOff:pTmin 0.654714*GeV
set /Herwig/Hadronization/ClusterFissioner:ClMaxLight 3.528693
set /Herwig/Hadronization/ClusterFissioner:ClPowLight 1.849375
set /Herwig/Hadronization/ClusterFissioner:PSplitLight 0.914156
set /Herwig/Hadronization/HadronSelector:PwtSquark 0.374094
set /Herwig/Hadronization/HadronSelector:PwtDIquark 0.33107
set /Herwig/Hadronization/ClusterDecayer:ClSmrLight 0.78
set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm
set /Herwig/Decays/DecayHandler:LifeTimeOption 0
read snippets/HepMC.in
set /Herwig/Analysis/HepMC:PrintEvent 100000000
set /Herwig/Analysis/HepMC:Filename events.hepmc
saverun r9730 EventGenerator
```

with `Herwig read` on this file and then `Herwig run r9730.run -N <events> --seed <seed>`, as
`generators/gen_stageF.sh` does. For another point, `gentune.axes.card(stage, theta)` writes the
parameter lines in the same form, and the listing of `reweight_own_events.py` prints them for the
defaults. `examples/data/README.md` has both cards as they made the example files.

**The stable-hadron convention is the trap that will get you.** Sherpa keeps hadrons with
`c*tau` above 10 mm as final state particles. Herwig decays K0S, Lambda, Sigma, Xi and Omega by
default. The two generators therefore disagree about what a final state particle is unless you
make them agree. Every sample here uses the Sherpa convention, which in a Herwig input file
means

```
set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm
set /Herwig/Decays/DecayHandler:LifeTimeOption 0
```

Without the first line the particle content differs and the weights are not the ones the model
was trained to give. The second is Herwig's default and is set only to be explicit.

The model then reads each event in its own thrust frame as a set of particles with six features.
If your events are built differently the weights are for a different question. The whole
construction is in `gentune/features.py`, which is deliberately one short file.

- every final state particle of the whole event except neutrinos, both hemispheres. The photons
  radiated by the beams are among them, as in training; only the observables leave them out
- `z` is the energy fraction of the whole event, `cos_theta` is signed against the thrust axis,
  so the hemisphere is carried by the sign, and `phi` is the azimuth around that axis
- at most 100 particles, keeping the hardest, padded and masked
- the per-particle standardization shipped with the model, never your own sample's statistics,
  because normalizing against your own mean removes exactly the between-run differences the
  model reads

The thrust axis of the features is that of an iteration, which the heads were trained with. It
ends on a local maximum in a few percent of Z-pole events. That changes nothing in the weights,
since the training samples were built the same way, but an event shape computed in this frame
differs from one with the exact axis (`generators/README.md`).

## Generator parameters that do not do what their name suggests

Found by reading the code and by scanning, and worth knowing whatever you use these models for.

- `ClusterFissioner:FissionPwtSquark` is inert unless `Fission` is set to `new`. Varying it in
  the default configuration changes nothing.
- `PartonSplitter:SplitPwtSquark`, default 0.824135, was tuned to 7 TeV proton collision
  minimum bias data and never to electron-positron annihilation. It is not an
  electron-positron parameter and is not varied here. The strange weight that is varied,
  `pwtsquark`, is `HadronSelector:PwtSquark`, the weight of a strange pair when a cluster decays
  to hadrons.
- `ClusterDecayer:ClSmrLight`, default 0.78, has never been fitted by any published Herwig
  tune. The one Bayesian study to float it prefers 0.675. It is varied here with a wide range
  for exactly that reason, and it turns out to be weak.
- In Sherpa's Ahadic the cluster decay weight uses `GAMMA_L/KT_0^2` and nothing else, so
  `GAMMA_L` and `KT_0` are one direction and not two. `KT_0` is held.
- `ALPHA_G` multiplies a weight `z^a + (1-z)^a` which is flat at `a = 1` exactly and not
  approximately, so it looks inert from the code and is not. Measured against a flat line it is
  the strongest single parameter for multiplicity in the Sherpa box.
- Sherpa's factorization scale weights are inert for electron-positron collisions. The
  seven-point columns are bitwise duplicates, so treating them as independent inflates a scale
  band by the square root of two.

## What is here

Three trained stages ship. `STAGES.md` gives each one's box, its training design and its measured
closure, and `gentune.axes` names the parameters, in the order the head stores them. The release
names differ from the paper's stage letters, as the last column says: Stages D and E of the paper
are released as `E` and `F`.

| stage | generator | parameters | what it varies | in the paper |
|---|---|---|---|---|
| `E` | Sherpa 3 | 8 | the shower coupling and seven Ahadic hadronization parameters, among them the strange and the baryon fraction. Trained on 352 runs, closure measured at 24 held-out points | Stage D |
| `F` | Herwig 7.3 | 8 | the shower coupling, the shower cutoff and six cluster hadronization parameters, among them the diquark weight that sets baryon production. Trained on 368 runs, closure measured at 24 held-out points | Stage E |
| `MIX17` | both | 17 | the `E` box, the `F` box, and the mixing fraction, 0 for Sherpa alone and 1 for Herwig alone. Reference sample of 1,149,840 events, closure measured at the 48 held-out points of both generators | the seventeen-parameter mixture |

`E`, `F` and `MIX17` are the networks of the paper. The networks of the paper's Stages A, B and C
(one, two and three parameters) are not here, and neither are those of an earlier draft, which the
git history keeps.

Parameter names, ranges, defaults, the settings they stand for and the traps just above are
printed by

```
python examples/parameter_scan.py --stage F --axis list
python examples/reweight_own_events.py --stage F
```

and `--axis` takes a name as well as an index. The names are short labels that follow the
generator's own, with `S:` and `H:` for the two blocks of `MIX17`. A label is not a setting:
`gentune.axes.SETTING` gives the full setting of each, as the training cards write it,
`gentune.axes.DEFAULT` the value the generator ships with, and `gentune.axes.card(stage, theta)`
the card lines for a point. `gentune.axes.verify` recomputes each box from the design CSV that
trained the stage and refuses to label anything if the two disagree, which is what makes the
names a measurement rather than a memory.

In this folder, a few megabytes per stage:

| file | what it is |
|---|---|
| `models/<stage>_head.npz` | the parameter network, the box, the training design, the calibration |
| `models/<stage>_trunk.npz` | the event-side network, so you can embed your own events |
| `models/<stage>_selftest.npz` | 2000 reference events, their place in the reference sample (`reference_index`), and the logits to reproduce |
| `models/MANIFEST.json` | shapes, boxes, checksums, and the verification each stage passed |
| `models/MIX17_wifi.npz` | the coefficients `w` of the learning band of `MIX17` and their covariance `C`, described above |
| `gentune/` | the reader, numpy only: `head`, `trunk`, `features`, `axes`, `mixture`, `bands`, `maxent` and `beam_photons`, and the self-test. `pyproject.toml` installs it |
| `examples/` | `reweight_own_events.py`, `parameter_scan.py`, `generator_band.py`, `learning_band.py` and `reproduce_paper.py`, with `paper_numbers.json`, the paper's numbers it compares with |
| `examples/data/` | the two example HepMC files, what gentune must give for them (`expected.npz`), and how they were made; `convention_factors.json`, the factors of a comparison with a measurement (Bands, above) |
| `tests/` | `test_maxent.py` and `test_mixture.py`, numpy only, and `compare_directlib.py`, which needs torch and the paper's `analysis/directlib.py` |
| `STAGES.md`, `closure_<stage>.csv`, `closure_summary.json`, `accuracy_law.json`, `accuracy_law_points.csv` | the closure of every stage at its held-out points, and the fit of the accuracy law (Accuracy, below) |
| `IMPOSE.md`, `REFERENCE.md` | imposing a calculation in numpy, and the reference sample of `MIX17` with the paper's calculation |
| `head_direction_response.py` | a check of each head: does it respond more along the parameter directions its training design explored least? Its docstring says how to read the answer. `whitened=0` in its output means that the training fed the parameters to the network without rescaling them by the spread of the design, an option that no released head uses |
| `tools_build_reference.py`, `tools_build_targets.py` | the scripts that built the two deposited files (`REFERENCE.md`). `python tools_build_targets.py SRC OUT` rebuilds the targets file from a calculation, for example `SRC` = `../output/thrust_targets_nlo_point.npz` in a full checkout |
| `generators/` | the generator scripts and parameter designs that produced the training samples |

The full reference sample, which is what the weights of the reference act on and what the paper's
own figures use, is hundreds of megabytes to a few gigabytes per stage. That of `MIX17`, 865 MB,
and the targets file of the paper's thrust calculation are in `deposit/` at the top of the
repository, with `build_reference.json`, the record of the build of the reference sample, and
`SHA256SUMS`, the sha256 of the three files. GitHub refuses files over 100 MB, so the reference
sample is stored there in 19 parts. `REFERENCE.md` says what the files hold and how they were built
and checked. From `analysis/release`, join the parts, copy the targets file and check both:

```
cat ../../deposit/reference_MIX17.npz.part-* > reference_MIX17.npz
cp ../../deposit/targets_thrust_nlo.npz .
grep -v build_reference ../../deposit/SHA256SUMS | shasum -a 256 -c
```

`shasum` must print OK for both files (on Linux, write `sha256sum -c` in place of
`shasum -a 256 -c`). `SHA256SUMS` also lists `build_reference.json`, which stays in `deposit/`, so
`grep -v` leaves its line out. The two files are then in `analysis/release`, where the commands
above look for them, and git ignores them. The joined file takes another 865 MB of disk. If a line
says FAILED, do not use that file: run the three lines again, and if it fails again, clone the
repository again.

The reference samples of the other stages are available on request. You do not need one to
reweight your own events, which is what `embed` and `logit` are for.

`generators/` is not needed to use a head either. It is there so the samples can be rebuilt and
so the parameter point behind any run id is on the record. `generators/README.md` has the
container recipe, both generators were run from it end to end, and the Herwig build it pins was
checked to give byte-identical events to the one the released sample was generated with.

## Accuracy

`STAGES.md` carries the closure of every released head against the generator on parameter points
held out of training, `closure_<stage>.csv` has it point by point, and `accuracy_law.json` has the
fit described below in machine-readable form. Every held-out run has 400,000 events. The six
observables leave out the photons radiated by the beams, as everywhere in the paper. The 48
held-out points of `MIX17` are the 24 of each generator, at the fraction 0 or 1, where the head is
one generator alone, so its closure does not depend on how the predictions are mixed. The paper's
closure widths of these models are in its Table 1, as the mean and standard deviation over three
ensembles.

A single worst-case percentage is a bad way to state this, because it is set by one point and
says nothing about where in the box that point sat. What the data supports is a floor plus a
weak dependence on how far you are asking the weights to move, measured as the displacement
between the target mean and the pooled reference mean in units of the statistical error on
that mean.

    relative error  =  floor  +  slope x displacement

| stage | floor | slope per 100 sigma | scatter | measurements | points above the fit |
|---|---|---|---|---|---|
| E, Sherpa, 8 parameters | 0.12% | -0.10% | 0.07% | 96 | 6 |
| F, Herwig, 8 parameters | 0.11% | 0.06% | 0.11% | 96 | 7 |
| MIX17, the two-generator mixture, 17 parameters | 0.15% | 0.03% | 0.13% | 192 | 14 |

Fitted robustly over four observables at the held points of each stage, with soft down-weighting
rather than rejection, so the points above the fit, more than three scatter units above it, are
identified by the fit rather than by choosing them. The displacements run up to 123 sigma for
Sherpa, 389 for Herwig and 384 for the mixture. The 384 measurements behind the three fits ship as
`accuracy_law_points.csv`, so the fit can be checked rather than taken on trust.

**The four observables are one minus thrust, total multiplicity, total broadening and the heavy
jet mass. The fit does not cover the two flavour rates, the baryon rate and the strange
fraction.** The baryon rate reaches 1.36 percent on `E`, 2.05 on `F` and 1.15 on `MIX17`, with a
standard deviation over the held points of 0.47, 0.70 and 0.48 percent. The strange fraction stays
within 0.6 percent on every stage. So a flavour rate has to be read off `STAGES.md` and
`closure_<stage>.csv` directly rather than predicted from the floor and the slope.

The points above the fit are named in `accuracy_law.json`. On every stage the largest is on one
minus thrust. On `F` it is the Herwig held point 8215, which closes to 1.11 percent where the fit
predicts 0.16. On `MIX17` it is the same point, at 1.20 percent against 0.18, and on `E` the Sherpa
held point 7219, at 0.61 percent against 0.12. These are the points the fit puts more than three
scatter units above the law.

Three things to keep in mind when you use these numbers.

**How hard you push.** On every stage the slope moves the error by at most 0.10 percent over 100
sigma of displacement. Compare it with the floor and the scatter of the stage before you reweight
far from the reference.

**A tenth of a percent is not small on a large sample.** At the held points a pull of one is 0.15
percent on one minus thrust, 0.05 percent on the total multiplicity and 0.17 percent on the baryon
rate, and the worst pulls in `STAGES.md` reach 9.7 on `E` (the baryon rate) and 13.7 on `F` (the
total multiplicity). The model error, not the event count, is what limits these weights there, and
generating more events does not reduce it.

**Seventeen parameters against eight.** The mixture's floor is 0.15 percent, against 0.12 and 0.11
for the two generators alone.

### The calibration constant

Every head carries one scalar, the calibration constant T (a temperature) that divides the logit
before it is exponentiated. All three ship at T = 1.000. Each is the value the training's own
procedure picks: the closure width on one half of the events of every held-out run, brought as close
to one as a fixed grid of T allows, here on the observables without the photons radiated by the
beams. `models/MANIFEST.json` and each head file record it.

## Citation

If you use these models, please cite the paper. `CITATION.bib` has the entry.
