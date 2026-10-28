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

## Install

From a checkout, which is what the rest of this file assumes:

```
git clone https://github.com/benleo12/BDistcrete
cd BDistcrete/analysis/release
pip install numpy
```

Every command below runs from `analysis/release`, and the scripts in `examples/` import
`gentune` from the folder next to them, so nothing else needs installing. To use `gentune` from
anywhere, `pip install analysis/release` from the top of the checkout installs it from there.

The package alone also installs with pip, without a checkout:

```
pip install 'git+https://github.com/benleo12/BDistcrete@v2.0.1#subdirectory=analysis/release'
```

This installs `gentune` and numpy and nothing else. The networks in `models/`, the example files
in `examples/data/` and the scripts in `examples/` are not part of the package, so they still come
from a checkout, and the installed package is pointed at them by path:

```
python -m gentune.selftest path/to/analysis/release/models path/to/analysis/release/examples/data
```

Without the second argument the second stage of the self-test is skipped, because by default it
looks for the example files next to the installed package, where they are not.

## Start here

```
python -m gentune.selftest models
```

It has two stages and takes about ten seconds on a laptop, longer on the first run. If it does
not print PASS for the stage you intend to use, stop.

The first stage runs the trunk and the head over the 2000 reference events that every released
stage ships, and compares with the logits that the shipped files produce for them. The stored
events hold the six per-particle columns already constructed, so this stage exercises the two
networks and the final column mapping but not the HepMC parsing or the thrust frame that produced
those columns. The logits it compares against are the ones the shipped files produce. The
training run's own logits are stored alongside as `logit_probe_from_export`, with the largest
difference between the two as `precision_gap`, so that gap is on the record rather than hidden.

The line of `MIX17` prints `N_eff/N=0.073` and a note under it. Its probe pair moves only the
fraction, from 0.5 to 0.9, from an even mixture of the two generators to mostly Herwig, so most
of the reference events lose their weight. That is expected and is not a failure. The other
stages move one parameter and keep 0.996 to 0.999.

The second stage covers the rest of the path. It reads the two example files in `examples/data/`,
30 Herwig 7.3.0 events in HepMC2 and 40 Sherpa 3.0.4 events in HepMC3, both at the generators'
default tunes and made with the training cards. It runs them through the parsing, the thrust
frame, the trunk and the `MIX17` head, and compares the features and the weights of one parameter
pair per file with `examples/data/expected.npz`. The comparison allows for the thrust axis, which
comes from an iteration that can end on the other of two nearly tied axes when the last bit of a
dot product differs between machines. So the columns that do not depend on the axis must agree in
every event to 1e-5, at most 5 percent of the events may move their axis, and every other event
must reproduce its log weight to 0.01. `examples/data/README.md` says how the files were made and
where these tolerances come from. If the example files are not there, the second stage says
which are missing and is skipped, and the first stage's result stands.

`E`, `F` and `MIX17` were trained on per-particle features stored as float16, and their trunk
files say so (`feature_precision`). `Trunk` rounds your features the same way before the network
sees them, which reproduces the training inputs. Read in float32 instead, the reweighted means of
`MIX17` would move by up to 0.17 percent, about its accuracy floor, and those of `E` and `F` by
0.02 percent (`analysis/precision_gap.py`).

Then look at what a head does:

```
python examples/parameter_scan.py --stage E --axis 0
```

and weight a sample, here the Herwig example file, from the default tune to a larger diquark
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
below), `per_event` with one column per name in `observables`.

`--from` is the point the sample was generated at and `--to` the point you want. Each is either
one number per parameter, in the order the listing prints, or a preset followed by the
parameters to change, as `name=value`. `sherpa-default` and `herwig-default` are the generators'
own defaults, the values a card that sets nothing gets, and `default` is the paper's central
point (below). For `MIX17` a sample made by one generator sits at an edge of the fraction, 0 for
Sherpa and 1 for Herwig, where the other generator's block drops out of the weights exactly. The
presets put that block at the centre of its box, where the training runs had it.

With `--list`, or without `--hepmc`, it prints the parameters, their ranges and defaults, and the
defaults as card lines (`python examples/reweight_own_events.py --stage MIX17 --list`). A point
with the wrong number of values, or outside the box, is refused with the name, value and range of
each parameter that is wrong.

## The two things you can ask for, which are not the same

```python
from gentune import Head, Trunk, features

head  = Head('models/E_head.npz')
trunk = Trunk('models/E_trunk.npz')

F, M, _ = features.features_from_events(features.parse_hepmc('sample.hepmc.gz'))
A = trunk.embed(features.trunk_features(F), M)

w = head.ratio_weights(A, theta_to, theta_from)   # your sample, generated at theta_from
w = head.weights(A, theta)                        # the reference sample only
```

`ratio_weights` is almost always the one you want. The classifier learned the log-ratio of each
parameter point to a common reference, which is the pooled union of all the training runs. The
difference of two of those logits is the log-ratio between the two points with the reference
cancelled exactly, so it applies to a sample generated at either point and needs no reference
events at all. Pass the parameters your sample was actually generated at as `theta_from`.

`weights` reweights the reference sample itself. It is correct only on reference events, which
means the events shipped with the export or events you generated from the same pooled
configuration. Using it on a sample from a single parameter point is a mistake that produces
plausible looking and wrong numbers. The 2000 events of every `models/<stage>_selftest.npz` are
reference events, drawn from that stage's reference sample, and `reference_index` gives the place
of each one in it, so `weights` is correct on them. Their event vectors are stored as `AE`.

Both return weights normalized to sum to one over the events you passed in, in the order you
passed them: `w[i]` belongs to the event of row `i` of `A`. `parse_hepmc` returns the events in
the order of the file, with the event number of each in `.numbers`, so for a HepMC file `w[i]` is
the weight of its i-th event, numbered `events.numbers[i]` there. Use `head.n_eff(w)` to see how
many events are effectively carrying the result. When that falls well below the event count the
weights are still correct, and the errors on anything computed from them grow accordingly.

## Two generators at once: the mixture head

A mixture head does not reweight within one generator. It reweights across a family that
contains both, so that "which generator" stops being a discrete choice you cannot interpolate
and becomes a number you can put a prior on. `MIX17` is the multiplicative mixture of the paper,

    q(Phi; theta_S, theta_H, f)  proportional to  q_S(Phi; theta_S)^(1 - f) q_H(Phi; theta_H)^f

so its log ratio to the reference is `(1 - f) l_S + f l_H`, linear in the fraction, where `l_S`
and `l_H` are the log ratios of the two generators. The paper calls the fraction xi. The
parameter vector is the two generators' parameters followed by the fraction,
`theta = (*theta_S, *theta_H, f)`, with `f` in [0, 1] and `f = 0` meaning Sherpa alone. Use it
exactly like a single-generator head:

```python
from gentune import load_head
head = load_head('models/MIX17_head.npz')     # the right class for any export
print(head.nS, head.nH)                       # how many parameters each side takes
w = head.ratio_weights(A, theta_to, theta_from)
```

`split()` divides a parameter vector into its three pieces, and the block sizes are read from
the exported networks rather than assumed, so the two generators need not have the same number
of parameters.

The edges are exact rather than approximate. At `f = 0` the log ratio is `l_S` itself and its
derivative with respect to `theta_H` vanishes identically, and the same holds at `f = 1` for
`theta_S`. So a fraction of zero is Sherpa alone, not a numerical approach to it, and a prior
that puts weight at the endpoints is not fighting the parameterization. The reader also takes
the additive mixture `(1 - f) q_S + f q_H` of earlier releases (`head_kind` `mixture`), whose
log ratio is `log[(1 - f) exp(l_S) + f exp(l_H)]`.

Two consequences worth knowing. A mixture head inherits the accuracy of both sides, so the
weaker of the two sets the floor, and `STAGES.md` reports it measured rather than inferred. And
at `f` near an endpoint the events of the silent generator carry little weight, so the
effective sample size falls even though the weights are correct, which `n_eff()` will show you.

## The learning band of MIX17

The paper's learning band comes from the WiFi fit of Benevedes and Thaler (arXiv:2506.00113).
Its basis functions are the cross terms of the M = 4 members, the event vector of member `i`
against the parameter vector of member `j`, plus a constant,

    log r(Phi, theta) = w_0 + sum_ij w_ij < a_i(Phi), b_j(theta) > ,

with `b_j(theta) = (1 - f) b_S,j(theta_S) + f b_H,j(theta_H)` for the mixture.
`models/MIX17_wifi.npz` holds the fitted weights `w` (`w_0` first, then `w_ij` row by row), their
covariance `C`, and `w_published`, the mean of the members in the same basis (`w_ij` equal to one
over M on the diagonal and zero elsewhere), which is what the prediction uses. The learning band
of a reweighted mean `<O>` is `sqrt(g C g)`, where `g_k`, the derivative of `<O>` with respect
to `w_k` at `w_published`, is the weighted covariance of `O` with the basis function `k`. With
the calculation imposed the derivative also runs through the re-solved multipliers.
`analysis/wifi_fit.py` makes the fit and `analysis/three_bands.py` propagates it.

Both are methods of the head:

```python
B    = head.wifi_basis(A, theta)                                  # (events, 1 + M^2)
band = head.learning_band(A, theta, O, 'models/MIX17_wifi.npz')   # sqrt(g C g)
```

Column 0 of the basis is the constant and column `k = 1 + i*M + j` is `< a_i(Phi), b_j(theta) >`,
with `i` the member of the event vector and `j` the member of the parameter vector, both counted
from zero. That is the order of `w`, so `B @ w_published` is `head.logit(A, theta)/head.T`.
`learning_band` takes the weights of `weights(A, theta)`, so `A` is reference events, or with
`theta_from` those of `ratio_weights`. `O` may be one value per event or one column per
observable, `mask` restricts the mean to a subset of the events, normalized over it, which gives
the band of a mean inside a window or of a bin of a distribution, and `counts` multiplies the
weights event by event, which with `gentune.bands.resample_counts` is a bootstrap. It is the band
for the generator alone, before a calculation is imposed.

## Bands

The generator band and the learning band of the paper come from two commands. On the paper's
reference sample, `reference_MIX17.npz`, a separate download (below), they give the paper's
numbers, on its observables and in its thrust bins:

```
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper
```

and on the 2000 reference events that ship with the networks, with no download, the same with
statistical errors 24 times larger:

```
python examples/generator_band.py --stage MIX17 --reference
python examples/learning_band.py  --stage MIX17 --reference
```

`generator_band.py` reweights the events to the 200 points of the paper's generator band, a
Latin hypercube over the box with seed 11 and the fraction uniform on [0, 1]. They are built as
`analysis/three_bands.py` builds them and are the paper's points to the last digit
(`gentune.bands.lhs_nodes`). For every observable it prints the mean over the points and the
band, which is the standard deviation of the weighted mean over the points, with its 16th and
84th percentiles. Next to them are the statistics that limit the band. `stat. error` is the median
over the points of the statistical error of the mean at one point, `sqrt(sum_i w_i^2 (O_i -
<O>)^2)`. `--bootstrap N`, 100 by default and 0 to skip it, resamples the events N times with
replacement and reweights every resample to every point with its weights renormalized, which gives
`boot sd`, the bootstrap standard deviation of the mean at a point (the median over the points),
and `boot sd of band`, that of the band itself. A band whose statistical error at a point is as
large as the band, or whose bootstrap standard deviation is more than a third of it, is marked `*`
and named in a warning as not significant, and a second warning says when the median `N_eff`
over the points is below 1000. It also prints the median, 5th percentile and smallest `N_eff/N`
over the points, and how many of them pass `head.support()`. `--nodes` and `--seed` change the
hypercube.

`learning_band.py` evaluates `head.learning_band` at one point, `--at`, by default the paper's
central point. `--at` is one number per parameter, or a preset followed by `name=value` changes,
as `--to` of `reweight_own_events.py`, for example `--at default H:pwtdiquark=0.40`. For each
observable it prints the value at the point, the learning band and the statistical error, and with
`--bootstrap N` the bootstrap standard deviation of the band. The WiFi weights ship for `MIX17`
only, so this script runs on `MIX17` only.

**Where the events come from.** `--reference-file FILE` takes the paper's reference sample and
reweights it with `weights`, which is exact there. `--reference` takes the 2000 reference events
of `models/<stage>_selftest.npz` with `weights`. For `MIX17` they are drawn from the reference
sample of 1,149,840 events, so every statistical error from them is about `sqrt(1149840/2000)`,
24 times, that of the paper. `--hepmc FILE --from POINT` takes a sample of your own and reweights
it with `ratio_weights`. `POINT` is written as for `--at`. On a sample of your own the learning
band is that of the log ratio between `--from` and `--at`, so it vanishes when the two are the
same point. The paper quotes the band only on the reference sample.

**Which stage.** `--stage F` (Herwig) or `--stage E` (Sherpa) gives the band of one generator's
own parameters, the band within that generator. `--stage MIX17` gives the paper's band, which
also sweeps the fraction from 0 to 1 and so includes the choice between Sherpa and Herwig. On a
sample of one generator it does that by reweighting the events toward the other generator, which
costs effective sample size: on the Herwig example file the median `N_eff/N` over the points is
0.21 with `MIX17` and 0.63 with `F`. For a Herwig sample at its default tune the two are

```
python examples/generator_band.py --stage F     --hepmc my_herwig.hepmc.gz --from herwig-default
python examples/generator_band.py --stage MIX17 --hepmc my_herwig.hepmc.gz --from herwig-default
```

and for a Sherpa sample `--stage E` and `--from sherpa-default`.

**The observables.** With `--reference-file` they are the paper's own, read from the file:
`1-T` on the exact thrust axis, `B_T`, `rho_H`, `n_ch` (among the 100 hardest particles, as in the
paper), `n` and `n_baryon` (the whole event). With `--reference` and `--hepmc` they are built from
the per-particle features by `gentune.bands.observables`: `1-T`, `n_ch`, `n` and `n_baryon`, each
over the at most 100 particles the features keep, with `1-T` read off the iterative thrust axis
the features are built on, which puts its mean about 1.2 percent above the exact axis
(`generators/README.md`). These names are the values `--hist` takes, `B_T` and `rho_H` with
`--reference-file` only.

**The bins.** `--bins EDGE ...` gives the band of every bin of the distribution of `--hist`, `1-T`
by default, normalized over the bins given. `--bins paper` gives the paper's: DELPHI's thrust
distribution at 91.2 GeV (Rivet `DELPHI_1996_S3430090`, `d11-x01-y01`) from `1-T` = 0.05 up, the
bins `analysis/three_bands.py` normalizes over, with the edges

    0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20 0.25 0.30 0.35 0.40 0.50

kept as the paper's code reads them, rounding included (`gentune.bands.PAPER_THRUST_BINS`), so
every event falls in the paper's bin. Giving these sixteen numbers to `--bins` gives the same bins
up to that rounding.

**What `--out` holds.** Both scripts write an npz with the per-event observables (`per_event`,
one column per name in `observables`) and the event numbers (`event_number`: the number on the
event's line in the HepMC file, or for reference events the index in the reference sample), next
to the points, the means or values and their errors, `N_eff`, the histograms and the bootstrap.
`generator_band.py --save-weights` adds the weights at every point, one row per point, each row in
the order of the events and summing to one, so the band of any other per-event quantity `x` of
the same events is `(weights @ x).std(ddof=1)`. The pieces are in `gentune.bands` for scripts of
your own, which need not import anything from `examples/`: `lhs_nodes`, `observables`,
`reference_observables`, `mean_and_error`, `bin_index`, `histogram`, `resample_counts`,
`resampled_sums` and `PAPER_THRUST_BINS`.

The paper's central point is both generators at their own defaults with a fraction of one half,
`gentune.axes.default('MIX17')` and the `CENTRAL` of `analysis/symmetry/three_bands.sbatch`:

| block | parameters, in the order of the head |
|---|---|
| Sherpa 3.0.4 | `alphas` 0.118, `pt_max` 0.68, `alpha_l` 3.9, `gamma_l` 0.48, `strange_fraction` 0.46, `baryon_fraction` 0.17, `alpha_g` 0.97, `beta_l` 0.18 |
| Herwig 7.3.0 | `alpha_fsr` 0.102337, `ptmin` 0.654714, `clmax` 3.528693, `clpow` 1.849375, `psplit` 0.914156, `pwtsquark` 0.374094, `pwtdiquark` 0.33107, `clsmr` 0.78 |
| fraction | 0.5 |

**Against the paper.** On the reference sample, the first two commands agree with the paper
(`analysis/output/three_bands_MIXGEOV3S9.json` and `.npz`): the generator band to 1e-9 relative
on the means and 1.2e-8 on the bins, as do the means at each of the 200 points; the learning
band to 1e-5; the values and statistical errors at the central point to 1e-9. The generator band
takes about a minute and the learning band a few seconds, both with 3 GB of memory. The paper's
generator band of `<1-T>` is 0.00438, of `<n_ch>` 0.677 and of `<n_baryon>` 0.284, and its
learning band 0.000286, 0.0594 and 0.00954.

On the 2000 reference events at the central point the two scripts give, against the paper,

| band | observable | 2000 reference events | the paper |
|---|---|---|---|
| generator | `<1 - T>` | 0.00386 ± 0.00040 | 0.00438 |
| generator | `<n_ch>` | 0.595 ± 0.037 | 0.677 |
| generator | `<n_baryon>` | 0.278 ± 0.011 | 0.284 |
| learning | `<1 - T>` | 0.000273 ± 0.000045 | 0.000286 |
| learning | `<n_ch>` | 0.0629 ± 0.0063 | 0.0594 |
| learning | `<n_baryon>` | 0.00985 ± 0.00068 | 0.00954 |

where each ± is the `boot sd of band` column of

```
python examples/generator_band.py --stage MIX17 --reference --bootstrap 300
python examples/learning_band.py  --stage MIX17 --reference --bootstrap 300
```

the standard deviation of the band over 300 bootstrap resamples of the 2000 events. The learning
bands agree within it. The generator bands of one minus thrust and of the charged multiplicity
come out 12 percent below the paper's, 1.3 and 2.2 bootstrap standard deviations, and one minus
thrust here is that of the features, on the iterative axis. So these events check the machinery,
and a band to quote needs the full reference sample. A bin of a distribution holds about a hundred
of the 2000 events, so the band of a bin from them is mostly statistical noise, and
`generator_band.py` marks every bin of `--bins paper` as not significant there.

## Imposing your own calculation

[`IMPOSE.md`](IMPOSE.md) describes how to impose a calculation, its moments and their
covariance, on the weights of the `MIX17` reference sample with `gentune.maxent`, in numpy, both
for the paper's thrust calculation and for one of your own, how to read a loaded calculation, and
what a changed one has to satisfy. It needs the reference sample and the paper's calculation, the
two files below, which [`REFERENCE.md`](REFERENCE.md) describes.
[`examples/reproduce_paper.py`](examples/reproduce_paper.py) reproduces the paper's central
predictions, before and after the calculation is imposed, and its generator band from those two
files, and prints PASS or FAIL against the paper's numbers:

```
python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz
```

It takes about six minutes on 8 cores and 2.5 GB of memory.

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
from every training point, which `in_box` alone does not. Drawn uniformly over the box, about two
points in a thousand fail `support` for `E` and `MIX17`, and fewer for `F`. All 200 points of the
generator band pass. A point outside the box, or with the wrong number of values, raises a
`ValueError` that names the index, the name, the value and the range of each parameter at fault.

## Conventions you have to match

A head gives the weights for samples made the way its training samples were made. Your sample
has to match on the following.

- **The process and the energy.** e+e- -> gamma*/Z -> q qbar at sqrt(s) = 91.2 GeV, the Z pole,
  with the shower and hadronization of the generator. No other energy was trained on.
- **The generators.** Sherpa 3.0.4 and Herwig 7.3.0 (the CVMFS build 7.3.0p1, with ThePEG
  2.3.0), pinned in `generators/cvmfs_env.sh`. `C_1M` was made with an unpinned build of Sherpa
  3 (`analysis/gen_stageC_v2.sh`).
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

- every final state particle of the whole event except neutrinos, both hemispheres
- `z` is the energy fraction of the whole event, `cos_theta` is signed against the thrust axis,
  so the hemisphere is carried by the sign, and `phi` is the azimuth around that axis
- at most 100 particles, keeping the hardest, padded and masked
- the per-particle standardization shipped with the model, never your own sample's statistics,
  because normalizing against your own mean removes exactly the between-run differences the
  model reads

The thrust axis of the features is that of an iteration, which the heads were trained with. It
ends on a local maximum in about 4 percent of Z-pole events. That changes nothing in the weights,
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

## What is here and what is a separate download

Four trained stages ship. `STAGES.md` gives each one's box, its training design and its measured
closure, and `gentune.axes` names the parameters, in the order the head stores them. The release
names differ from the paper's stage letters, as the last column says: Stages D and E of the paper
are released as `E` and `F`.

| stage | generator | parameters | what it varies | in the paper |
|---|---|---|---|---|
| `E` | Sherpa 3 | 8 | the shower coupling and seven Ahadic hadronization parameters, among them the strange and the baryon fraction. Trained on 352 runs, closure measured at 24 held-out points | Stage D |
| `F` | Herwig 7.3 | 8 | the shower coupling, the shower cutoff and six cluster hadronization parameters, among them the diquark weight that sets baryon production. Trained on 368 runs, closure measured at 24 held-out points | Stage E |
| `MIX17` | both | 17 | the `E` box, the `F` box, and the fraction of the sample the Herwig side supplies. Reference sample of 1,149,840 events, closure measured at the 48 held-out points of both generators | the seventeen-parameter mixture of Sec. 4 |
| `C_1M` | Sherpa 3 | 3 | the shower coupling, the strange fraction and the Ahadic cluster transverse momentum scale, trained on a cache of 1,080,000 events | an earlier version only |

`E`, `F` and `MIX17` are the networks of the current paper, released in version 2.0.0 of this
release and unchanged since. The git history keeps the networks of version 1.0.0. `C_1M` is that
of version 1.0.0. The networks of the paper's Stages A, B and C (one, two and three parameters)
are not here.

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

In this repository, a few megabytes per stage:

| file | what it is |
|---|---|
| `models/<stage>_head.npz` | the parameter network, the box, the training design, the calibration |
| `models/<stage>_trunk.npz` | the event-side network, so you can embed your own events |
| `models/<stage>_selftest.npz` | 2000 reference events, their place in the reference sample (`reference_index`), and the logits to reproduce |
| `models/MANIFEST.json` | shapes, boxes, checksums, and the verification each stage passed |
| `models/MIX17_wifi.npz` | the basis weights `w` of the learning band of `MIX17` and their covariance `C`, described above |
| `gentune/` | the reader, numpy only: `head`, `trunk`, `features`, `axes`, `bands` and `maxent`, and the self-test. `pyproject.toml` installs it |
| `examples/` | `reweight_own_events.py`, `parameter_scan.py`, `generator_band.py`, `learning_band.py` and `reproduce_paper.py` |
| `examples/data/` | the two example HepMC files, what gentune must give for them (`expected.npz`), and how they were made |
| `tests/` | `test_maxent.py`, numpy only, and `compare_directlib.py`, which needs torch and the paper's `analysis/directlib.py` |
| `IMPOSE.md`, `REFERENCE.md` | imposing a calculation in numpy, and the reference sample of `MIX17` with the paper's calculation |
| `generators/` | the generator scripts and parameter designs that produced the training samples |

The full reference sample, which is what `weights` acts on and what the paper's own figures use,
is hundreds of megabytes to a few gigabytes per stage, so it is a separate download rather than
carried here. That of `MIX17`, 865 MB, and the targets file of the paper's thrust calculation are
public at https://github.com/benleo12/BDistcrete/releases/tag/data-r8, with `REFERENCE.md` and
their checksums, and will also be archived at Zenodo. `REFERENCE.md` says what they hold and how
they were built and checked. In `analysis/release`, where the commands above look for them:

```
for f in reference_MIX17.npz targets_thrust_nlo.npz REFERENCE.md SHA256SUMS; do
  curl -LO https://github.com/benleo12/BDistcrete/releases/download/data-r8/$f
done
shasum -a 256 -c SHA256SUMS          # sha256sum -c SHA256SUMS on Linux; every line must say OK
```

The reference samples of the other stages are available on request. You do not need one to
reweight your own events, which is what `embed` and `logit` are for.

`generators/` is not needed to use a head either. It is there so the samples can be rebuilt and
so the parameter point behind any run id is on the record. `generators/README.md` has the
container recipe, both generators were run from it end to end, and the Herwig build it pins was
checked to give byte-identical events to the one the released sample was generated with.

## Accuracy

`STAGES.md` carries the closure of every released head against the generator on parameter
points held out of training, `closure_<stage>.csv` has it point by point, and
`accuracy_law.json` has the fit described below in machine-readable form. For `E`, `F` and
`MIX17` every held-out run has 400,000 events. The paper's closure widths of the same networks
are in its Table 1.

A single worst-case percentage is a bad way to state this, because it is set by one point and
says nothing about where in the box that point sat. What the data supports is a floor plus a
weak dependence on how far you are asking the weights to move, measured as the displacement
between the target mean and the pooled reference mean in units of the statistical error on
that mean.

    relative error  =  floor  +  slope x displacement

| stage | floor | slope per 100 sigma | scatter | measurements | points above the law |
|---|---|---|---|---|---|
| E, Sherpa, 8 parameters | 0.12% | -0.10% | 0.07% | 96 | 6 |
| F, Herwig, 8 parameters | 0.11% | 0.08% | 0.11% | 96 | 7 |
| MIX17, the two-generator mixture, 17 parameters | 0.16% | 0.03% | 0.13% | 192 | 12 |

Fitted robustly over four observables at the held points of each stage, with soft
down-weighting rather than rejection, so the points above the law, more than three scatter units
above it, are identified by the fit rather than by choosing them. The displacements run up to 123
sigma for Sherpa, 389 for Herwig and 368 for the mixture. The 384 measurements behind the three
fits ship as `accuracy_law_points.csv`, so the fit can be checked rather than taken on trust.

**The four observables are one minus thrust, total multiplicity, total broadening and the heavy
jet mass. The law does not cover the two flavour rates, the baryon rate and the strange
fraction.** The baryon rate reaches 1.36 percent on `E`, 2.05 on `F` and 1.15 on `MIX17`, with a
root mean square over the held points of 0.47, 0.70 and 0.48 percent, three to seven times the
floor. The strange fraction stays within 0.6 percent on every stage. So a flavour rate has to be
read off `STAGES.md` and `closure_<stage>.csv` directly rather than predicted from the floor and
the slope.

The points above the law are named in `accuracy_law.json`. All but one are event shapes, at three
held points of `E` and four of `F`, and on `MIX17` mostly at the same Herwig points. The largest
is the Herwig held point 8215, at the low end of the cluster fission power: one minus thrust closes
to 0.99 percent there on `F` and to 1.14 percent on `MIX17`, where the law predicts 0.16 and 0.18.
**So an event shape of Herwig can be off by up to one percent at some points of its box, about
five times the floor.**

Three things follow.

**The floor dominates.** On every stage the slope moves the error by less than 0.1 percent over
100 sigma of displacement, no more than the scatter. So the accuracy you get is mostly a property
of the head and not of how hard you push it, which is convenient: one number per stage is close
to the whole story.

**A tenth of a percent is not small on a large sample.** At the held points a pull of one is 0.15
percent on one minus thrust and 0.05 percent on the total multiplicity, so the floors are pulls of
one to three standard deviations there, and the worst pulls in `STAGES.md` reach 9.7 on `E` and 14
on `F`, on the baryon rate and the total multiplicity. The model error, not the event count, is
what limits these weights, and generating more events does not reduce it. If you need a rate to
better than a few tenths of a percent, this is the wrong tool.

**Seventeen parameters cost little over eight.** The mixture's floor is 0.16 percent against 0.12
and 0.11 for the two generators alone, and its slope is as small. Widening the box did not cost an
order of magnitude, which is what makes a continuous family practical rather than a curiosity.

### Why the baryon rate is the worst of the six

Two measured properties of the observable account for most of it, and neither is a limit on the
network's ability to see baryons. They were measured on `C_1M` and its box, and the tables of every
stage show the same ordering.

The parameter box moves the baryon rate furthest. Over the 27 training points of the
three-parameter Sherpa box the generator's own mean baryon rate has a spread of 11.8 percent of
its mean, against 2.8 to 6.2 percent for the four event shapes, and that box contains no baryon
knob at all, so the whole move has to be delivered through the strong coupling. In the Herwig box
the diquark weight alone moves the rate by 96 percent end to end while thrust moves 1.9 percent.
And a baryon count is a wide per-event variable, with a relative spread of 1.04 against 0.29 for
the total multiplicity, so at equal statistical significance its percent error is automatically
about three and a half times larger. That factor is in the shipped tables: at the held points of
`E` and `F` a pull of one is 0.17 percent on the baryon rate and 0.05 percent on the total
multiplicity.

What is left after those two is not a representation problem. A linear fit of the per-event
baryon count on the 96 columns of the event vector reaches a held-out R-squared of 0.91, among
the best of the six, and a rank-one tilt of 2 percent of the head's own logit spread zeroes the
worst Stage C baryon error at a cost of 0.1 percent of the effective sample size. The
remainder sits in the parameter network, and it tracks curvature. Fitting a plane in the three
parameters to the generator's response over the 27-point grid leaves the baryon rate with a
non-planar part of 3.1 percent of its mean against 0.5 to 1.0 percent for the event shapes. The
strange fraction is the clean control: its lever arm is the largest of all six at 14.5 percent,
larger than the baryon rate's, and it still closes better, because its response is nearly planar
at 0.6 percent.

Two mechanisms that sound right were measured and are wrong. Discreteness and the sparse tail of
events carrying several baryons are not the cause: the count is strictly even, the mean of half
the count has the same relative error, and capping the count at two removes only a quarter to
four tenths of the error while leaving the rate still the worst of the observables. Nor is it the
statistics: over 17 Stage C points the mean squared pull is 10.2 against 1 for noise, dropping
the three worst points leaves 0.95 percent against 0.44 for the shapes, and at the same parameter
point two independent generator seeds agree to a mean squared pull of 0.97, so the rate is a
reproducible function of the parameters.

### Away from its held points, `C_1M` reads the flavour rates about three times worse

The numbers in `STAGES.md` are measured at the points held out of training, which for the
three-parameter stage are four interior points. At twelve further points that were used for nothing
at all, the root mean square relative error over the same reference rises from 0.35 to 1.17 percent
for the baryon rate, against 0.15 to 0.22 for the multiplicity, 0.12 to 0.34 for the broadening and
0.20 to 0.51 for one minus thrust. The strange fraction is the exception, 0.71 at the held points
and 0.78 at the fresh ones, for the reason given next. So a baryon rate at an arbitrary point of the
box should be expected at the percent level rather than at the few tenths the held points show,
while the shape observables degrade by a factor of two to three from a lower base. The held points
of `E`, `F` and `MIX17` are spread over their boxes, 16 of the 24 per generator drawn at random, so
their tables already describe a typical point.

### The calibration temperature

Every head carries one scalar, the calibration temperature that divides the logit before it is
exponentiated. `E`, `F` and `MIX17` ship at 1.0, where their calibration statistic is 0.92 to 1.02
for every observable against a null of one. For `C_1M` the training run chose it on one half of the
held runs by bringing a binned closure statistic to its null of one, but over a grid that had no
value between 1.0 and 1.2, and the three-parameter head was assigned 1.2 when the same objective on
a fine grid gives 1.125. The head here ships at 1.125, and the export records the original value and
the reason. At the twelve fresh points every one of the six observables improves, the all-observable
root mean square from 0.83 to 0.67 percent, and at the held points five of six improve. The strange
fraction at the held points is the one number that gets worse, 0.28 to 0.71 percent, and it is also
the one whose held-point value at 1.2 was partly in-sample: the temperature was selected at those
same parameter points with an objective in which the strange fraction's binned agreement carried
weight. At the fresh points, which selected nothing, the strange fraction improves from 1.08 to
0.78. The worst-case entry for this head in `STAGES.md` is therefore the strange fraction at one
held point, 1.10 percent, and it is a more honest number than the 0.72 it replaces.

## Citation

If you use these models, please cite the paper. `CITATION.bib` has the entry.
