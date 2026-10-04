# Continuous generator-parameter reweighting

This is the released code and trained models from *Beyond Discrete Variations: Continuous
Generator Uncertainties Anchored to Precision QCD*. It answers one question, per event, without
running a generator:

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
files produce on a float32 reader, not the ones the training run itself recorded: stages trained
through the float16 feature cache differ from their training run by a few parts in a thousand in
the logit, which is stored alongside as `logit_probe_from_export` so the gap is on the record
rather than hidden. The effect on anything anyone uses is smaller than the statistical error of
the reference sample.

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
and becomes a number you can put a prior on.

    q(Phi; theta_S, theta_H, f) = (1 - f) q_S(Phi; theta_S) + f q_H(Phi; theta_H)

Its parameter vector is the two generators' parameters followed by the fraction,
`theta = (*theta_S, *theta_H, f)`, with `f` in [0, 1] and `f = 0` meaning the first generator
alone. Use it exactly like a single-generator head:

```python
from gentune import MixtureHead
head = MixtureHead('models/<tag>_head.npz')
print(head.nS, head.nH)          # how many parameters each side takes
w = head.ratio_weights(A, theta_to, theta_from)
```

`split()` divides a parameter vector into its three pieces, and the block sizes are read from
the exported networks rather than assumed, so the two generators need not have the same number
of parameters.

The edges are exact rather than approximate. Writing the logit as
`log[(1-f) exp(l_S) + f exp(l_H)]` makes the derivative with respect to `theta_H` vanish
identically at `f = 0`, and with respect to `theta_S` at `f = 1`. So a fraction of zero is the
first generator alone, not a numerical approach to it, and a prior that puts weight at the
endpoints is not fighting the parameterization.

Two consequences worth knowing. A mixture head inherits the accuracy of both sides, so the
weaker of the two sets the floor, and `STAGES.md` reports it measured rather than inferred. And
at `f` near an endpoint the events of the silent generator carry little weight, so the
effective sample size falls even though the weights are correct, which `n_eff()` will show you.

## Being inside the box is not the same as having been sampled

```python
head.check(theta)          # raises outside the box, warns outside the sampled region
head.support(theta)        # score, threshold, ok, nearest
```

`box` is the range of each parameter. `support` is the question of whether the head ever saw a
point like yours, and the two come apart. The Sherpa stage varies its eight parameters
independently, so its box is filled. The Herwig stage varies eight parameters along a
six-dimensional locus, because the published retunes that justify its ranges move the shower
coupling and the shower cutoff together, and the cluster fission mass and power together.
Sampling those as an independent rectangle would admit corners no tune has ever realised. The
consequence for you is that large parts of the Herwig bounding box were never generated, and
`in_box` alone will not tell you so. `check` will.

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
closure, and `gentune.axes` names the parameters, in the order the head stores them.

| stage | generator | parameters | what it varies |
|---|---|---|---|
| `C_1M` | Sherpa 3 | 3 | the shower coupling, the strange fraction and the Ahadic cluster transverse momentum scale. The paper's anchor section uses it as the three-parameter cross-check of its fit, trained on a cache of 1,080,000 events |
| `E` | Sherpa 3 | 8 | the shower coupling and seven Ahadic hadronization parameters, among them the strange and the baryon fraction |
| `F` | Herwig 7.3 | 8 | the shower coupling, the shower cutoff and six cluster hadronization parameters, among them the diquark weight that sets baryon production |
| `MIX17` | both | 17 | the `E` box, the `F` box, and the fraction of the sample the Herwig side supplies. This is the model the paper's anchor section anchors and fits, on its reference sample of 1,149,888 events |

The paper's ladder heads, the one, two and three-parameter models behind its closure table, are
not here. The three-parameter model that is here is the separate, larger training the anchor
section compares with. The fit of the strong coupling itself uses `MIX17`.

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
`accuracy_law.json` has the fit described below in machine-readable form.

A single worst-case percentage is a bad way to state this, because it is set by one point and
says nothing about where in the box that point sat. What the data supports is a floor plus a
weak dependence on how far you are asking the weights to move, measured as the displacement
between the target mean and the pooled reference mean in units of the statistical error on
that mean.

    relative error  =  floor  +  slope x displacement

| stage | floor | slope per 100 sigma | scatter | measurements | exceptions |
|---|---|---|---|---|---|
| E, Sherpa, 8 parameters | 0.24% | 0.61% | 0.22% | 32 | none |
| F, Herwig, 8 parameters | 0.39% | 0.28% | 0.29% | 32 | none |
| MIX17, the two-generator mixture, 17 parameters | 0.28% | -0.22% | 0.15% | 40 | one, below |

Fitted robustly over four observables at the held points of each stage, with soft down-weighting
rather than rejection, so the exceptions are identified by the fit rather than by choosing them.
The displacements covered run from 1.0 to 33 sigma for Sherpa, 5.8 to 85 for Herwig and 1.9 to
29 for the mixture. The 104 points behind the three fits ship as `accuracy_law_points.csv`, so
the fit can be checked rather than taken on trust.

**The four observables are one minus thrust, total multiplicity, total broadening and the heavy
jet mass. The law does not cover the two flavour rates, the baryon rate and the strange
fraction, and both exceed its floor by a factor of two to nine.** The baryon rate reaches 1.16
percent on stage E against a floor of 0.24, 1.95 on stage F against 0.39, and 2.01 on the
mixture against 0.28. The strange fraction reaches 0.53, 0.60 and 0.85 against the same floors.
On every stage the baryon rate has the largest relative error of the six. So the law describes
the shape observables, and a flavour rate has to be read off `STAGES.md` and
`closure_<stage>.csv` directly rather than predicted from the floor and the slope. In pull units
the ranking is different, and the total multiplicity is the worse of the two on stages F and
MIX17.

### Why the baryon rate is the worst of the six

Two measured properties of the observable account for most of it, and neither is a limit on the
network's ability to see baryons.

The parameter box moves the baryon rate furthest. Over the 27 training points of the
three-parameter Sherpa box the generator's own mean baryon rate has a spread of 11.8 percent of
its mean, against 2.8 to 6.2 percent for the four event shapes, and that box contains no baryon
knob at all, so the whole move has to be delivered through the strong coupling. In the Herwig box
the diquark weight alone moves the rate by 96 percent end to end while thrust moves 1.9 percent.
And a baryon count is a wide per-event variable, with a relative spread of 1.04 against 0.29 for
the total multiplicity, so at equal statistical significance its percent error is automatically
about three and a half times larger. That factor is in the shipped tables and is the same on
every stage: the relative error per unit pull is 0.37 to 0.38 percent per sigma for the baryon
rate against 0.11 for the multiplicity on all four.

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

### Away from the held points the flavour rates are about three times worse

The numbers in `STAGES.md` are measured at the points held out of training, which for the
three-parameter stage are five interior points. At twelve further points that were used for
nothing at all, the root mean square relative error over the same reference rises from 0.35 to
1.17 percent for the baryon rate, against 0.15 to 0.22 for the multiplicity, 0.12 to 0.34 for the
broadening and 0.20 to 0.51 for one minus thrust. The strange fraction is the exception, 0.71 at
the held points and 0.78 at the fresh ones, for the reason given next. So a baryon rate at an
arbitrary point of the box should be expected at the percent level rather than at the few tenths
the held points show, while the shape observables degrade by a factor of two to three from a
lower base.

### The calibration temperature, and why the strange fraction reads worse at the held points

Every head carries one scalar, the calibration temperature that divides the logit before it is
exponentiated. The training run chose it on one half of the held runs by bringing a binned
closure statistic to its null of one, but over a grid that had no value between 1.0 and 1.2, and
the three-parameter head was assigned 1.2 when the same objective on a fine grid gives 1.125.
The head here ships at 1.125, and the export records the original value and the reason. At the
twelve fresh points every one of the six observables improves, the all-observable root mean
square from 0.83 to 0.67 percent, and at the held points five of six improve. The strange
fraction at the held points is the one number that gets worse, 0.28 to 0.71 percent, and it is
also the one whose held-point value at 1.2 was partly in-sample: the temperature was selected at
those same parameter points with an objective in which the strange fraction's binned agreement
carried weight. At the fresh points, which selected nothing, the strange fraction improves from
1.08 to 0.78. The worst-case entry for this head in `STAGES.md` is therefore the strange fraction
at one held point, 1.10 percent, and it is a more honest number than the 0.72 it replaces.

Three things follow.

**The floor dominates.** Over a 50 sigma displacement, further than any held point in the Sherpa
box, the distance term adds 0.21% on top of a 0.25% floor. So the accuracy you get is mostly a
property of the head and not of how hard you push it, which is convenient: one number per stage
is close to the whole story.

**A third of a percent is not small on a large sample.** On the million-event reference the
statistical error on a mean is about 9 parts in ten thousand, so the Sherpa floor of 0.25% is
a three standard deviation pull and the Herwig floor is four. The model error, not the event
count, is what limits these weights, and generating more events does not reduce it. If you need
a rate to better than half a percent, this is the wrong tool.

**Eight parameters cost little over three, and seventeen cost nothing further.** The
three-parameter Sherpa head reaches 1.10% worst case and the eight-parameter one 1.16%, while
its floor is 0.25%. The seventeen-parameter mixture's floor is 0.29%, and its fitted slope is
slightly negative, which at this scatter means no measurable dependence on displacement at all.
Widening the box did not cost an order of magnitude, which is what makes a continuous family
practical rather than a curiosity.

### The documented exception, and the near one

Both are total multiplicity, and both are on the Herwig side.

On the mixture, held point 9934 closes its total multiplicity to 1.15% where the law predicts
0.23%, 6.1 scatter units out, one measurement in forty and the only one beyond three units on
any stage. On the Herwig head, held point 8206 closes to 1.35% where the law predicts 0.50%,
2.9 units out, so the fit no longer lists it, but it is the worst single multiplicity error of
that stage and it sits just under the line. **Treat Herwig multiplicity at the edges of its box
as uncertain at the one to one and a half percent level.**

`support()` will not flag either of them, and it is worth being exact about why, because an
earlier version of this section told you to call it as though it would. `support()` answers
whether a point sits in the region the training design covered. It does not answer whether the
weights will be accurate there, and on this head it does not even rank the risk: all eight
Herwig held points return `ok=True`, because the threshold is the largest score over the
training design and every held point is inside that design by construction, and 8206 is only
sixth of the eight by score while carrying the worst multiplicity error. Across those eight
points the score correlates with the size of that error at 0.26, which is nothing to rely on.
So read `support()` as the question it answers, whether the region was sampled, and read the
accuracy off the closure tables.

What that point is not: it is not a training shortfall, since doubling the budget to 72,000
steps moved the width from 1.232 to 1.235. It is not too few runs, since training the Sherpa
head on 64 runs to match Herwig's count and reference size cost 0.014 in width against
Herwig's deficit of 0.149. It is not extrapolation beyond the data, since the point sits at
position 0.34 inside the range of multiplicities the training runs span. And it is not input
conditioning, since whitening the parameters by the design covariance made every observable
worse in both stages. Each of those was tested rather than argued.

### Why the Herwig floor is above Sherpa's

This was an open question in an earlier version of the release and it is not any more, so here
is what was established and what remains.

The cause was the shape of the Herwig training design. It varied two of its eight parameters at
30 percent of the amplitude of the other six, following published retunes in which the shower
coupling and the cutoff move together, and the head responded 2.14 times more strongly along
those two starved directions where the isotropic Sherpa head responds 0.59 times as strongly.
That is a head fitting noise where the design gave it nothing, and it is measurable on any head
with `head_direction_response.py`.

Four other explanations were tested and rejected. Doubling the training budget on the original
design moved the closure width from 1.232 to 1.235. Training the Sherpa head on 64 runs to match
Herwig's count cost 0.014 in width against Herwig's deficit of 0.149. The failing point sat
inside the range of multiplicities the training runs span, so it was not extrapolation. And
whitening the parameters by the design covariance made every observable worse in both stages.

The fix was to add 48 further Herwig points placed to raise the smallest eigenvalue of the
design covariance, which took its anisotropy from 3.66 to 1.09 and matched the Sherpa box's
1.11. The head shipped here is trained on all 112 points. Against the 64-point head it improves
five of six observables, halves the multiplicity error from 1.31 to 0.65 percent, takes the
worst single error from 3.43 to 1.95 percent and the worst pull from 28.8 to 11.5, and lowers
this law's floor from 0.47 to 0.41 percent and its scatter from 0.49 to 0.32, both measured on
the event shapes before the thrust-axis correction (on the corrected shapes the 112-point head gives
0.39 and 0.29). The 64-point head's numbers are at its own calibration of 1.0, the
112-point head's at the recalibrated 1.025.

Two things did not resolve, and both are stated above rather than here: the baryon rate closes
worse than it did on the narrower design, 1.24 against 0.93 percent in root mean square, and it
has the largest relative error of the six on every stage, for the reasons measured above. And the
direction response fell to 1.20 rather
than below one, so the head still responds somewhat more strongly along the directions its
design covers least, on a design where those directions are now barely distinguishable from the
rest.

One practical note for anyone reproducing this. The 48 added points deliberately sit where no
published tune sits, with a high shower coupling against a low cutoff and the reverse, because a
design has to span what the head will be asked about while a prior says where the answer
probably lies. Folding the prior into the design is what left the first head unable to answer
off-locus questions, and that cannot be undone after the fact. Restricting to the locus remains
available afterwards as a prior.

## Citation

If you use these models, please cite the paper. `CITATION.bib` has the entry.
