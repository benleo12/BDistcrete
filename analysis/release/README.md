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

```
python -m gentune.selftest models
```

Every released stage ships a small bundle of reference events together with the logits that the
shipped files produce for them. The self-test runs the trunk and the head over those events and
compares. It takes a few seconds, and if it does not print PASS for the stage you intend to
use, stop.

Two things about its scope, because it is easy to read more into a PASS than it carries. The
stored events hold the six per-particle columns already constructed, so the self-test exercises
the two networks and the final column mapping but NOT the HepMC parsing or the thrust-frame
construction that produced those columns. Run `examples/reweight_own_events.py` on a file of
your own to exercise that path. And the logits it compares against are the ones the shipped
files produce. The training run's own logits are stored alongside as `logit_probe_from_export`,
with the largest difference between the two as `precision_gap`, so that gap is on the record
rather than hidden.

`E`, `F` and `MIX17` were trained on per-particle features stored as float16, and their trunk
files say so (`feature_precision`). `Trunk` rounds your features the same way before the network
sees them, which reproduces the training inputs. Read in float32 instead, the reweighted means of
`MIX17` would move by up to 0.17 percent, about its accuracy floor, and those of `E` and `F` by
0.02 percent (`analysis/precision_gap.py`).

Then look at what a head does:

```
python examples/parameter_scan.py --stage E --axis 0
```

and weight your own events:

```
python examples/reweight_own_events.py --stage E --hepmc my_sample.hepmc.gz \
    --from <the parameters your sample was generated at> \
    --to   <where you want it>
```

Run it with `--stage E` and nothing else to print the parameters and their ranges.

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
plausible looking and wrong numbers.

Both return weights normalized to sum to one over the events you passed in. Use
`head.n_eff(w)` to see how many events are effectively carrying the result. When that falls
well below the event count the weights are still correct, and the errors on anything computed
from them grow accordingly.

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
from every training point, which `in_box` alone does not.

## Conventions you have to match

The model reads each event in its own thrust frame as a set of particles with six features. If
your events are built differently the weights are for a different question. The whole
construction is in `gentune/features.py`, which is deliberately one short file.

- every final state particle of the whole event except neutrinos, both hemispheres
- `z` is the energy fraction of the whole event, `cos_theta` is signed against the thrust axis,
  so the hemisphere is carried by the sign, and `phi` is the azimuth around that axis
- at most 100 particles, keeping the hardest, padded and masked
- the per-particle standardization shipped with the model, never your own sample's statistics,
  because normalizing against your own mean removes exactly the between-run differences the
  model reads

**The stable-hadron convention is the trap that will get you.** Sherpa keeps hadrons with
`c*tau` above 10 mm as final state particles. Herwig decays K0S, Lambda, Sigma, Xi and Omega by
default. The two generators therefore disagree about what a final state particle is unless you
make them agree. Every sample here uses the Sherpa convention, which in a Herwig input file
means

```
set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm
```

Without that line the particle content differs and the weights are not the ones the model was
trained to give.

## Generator parameters that do not do what their name suggests

Found by reading the code and by scanning, and worth knowing whatever you use these models for.

- `ClusterFissioner:FissionPwtSquark` is inert unless `Fission` is set to `new`. Varying it in
  the default configuration changes nothing.
- `PartonSplitter:SplitPwtSquark`, default 0.824135, was tuned to 7 TeV proton collision
  minimum bias data and never to electron-positron annihilation. It is not an
  electron-positron parameter and is not varied here.
- `ClusterSmearing:ClSmrLight`, default 0.78, has never been fitted by any published Herwig
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
names differ from the paper's stage letters, as the last column says.

| stage | generator | parameters | what it varies | in the paper |
|---|---|---|---|---|
| `E` | Sherpa 3 | 8 | the shower coupling and seven Ahadic hadronization parameters, among them the strange and the baryon fraction. Trained on 352 runs, closure measured at 24 held-out points | Stage D |
| `F` | Herwig 7.3 | 8 | the shower coupling, the shower cutoff and six cluster hadronization parameters, among them the diquark weight that sets baryon production. Trained on 368 runs, closure measured at 24 held-out points | Stage E |
| `MIX17` | both | 17 | the `E` box, the `F` box, and the fraction of the sample the Herwig side supplies. Reference sample of 1,149,840 events, closure measured at the 48 held-out points of both generators | the seventeen-parameter mixture of Sec. 4 |
| `C_1M` | Sherpa 3 | 3 | the shower coupling, the strange fraction and the Ahadic cluster transverse momentum scale, trained on a cache of 1,080,000 events | an earlier version only |

`E`, `F` and `MIX17` are the networks of the current paper (version 2.0.0 of this release). The
git history keeps the networks of version 1.0.0. `C_1M` is that of version 1.0.0. The networks of the paper's Stages A, B and C (one, two and three
parameters) are not here.

Parameter names, ranges and the traps just above are printed by

```
python examples/parameter_scan.py --stage F --axis list
python examples/reweight_own_events.py --stage F
```

and `--axis` takes a name as well as an index. The names are the generator's own, so they can be
pasted into a Sherpa run card or a Herwig input file and found. `gentune.axes.verify` recomputes
each box from the design CSV that trained the stage and refuses to label anything if the two
disagree, which is what makes the names a measurement rather than a memory.

In this repository, a few megabytes per stage:

| file | what it is |
|---|---|
| `models/<stage>_head.npz` | the parameter network, the box, the training design, the calibration |
| `models/<stage>_trunk.npz` | the event-side network, so you can embed your own events |
| `models/<stage>_selftest.npz` | reference events and the logits to reproduce |
| `models/MANIFEST.json` | shapes, boxes, checksums, and the verification each stage passed |
| `models/MIX17_wifi.npz` | the basis weights `w` of the learning band of `MIX17` and their covariance `C`, described above |
| `generators/` | the generator scripts and parameter designs that produced the training samples |

The full reference sample, which is what `weights` acts on and what the paper's own figures use,
is hundreds of megabytes to a few gigabytes per stage, so it is deposited separately rather than
carried here. It goes to Zenodo when the paper is published, and the DOI is added to this file
and to the paper at that point. You do not need it to reweight your own events, which is what
`embed` and `logit` are for.

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
