# The MIX17 reference sample

`models/MIX17_head.npz` reweights one fixed sample of events, the reference, to any point of its
seventeen-parameter box. The weights only mean something on those events, and so does the
calculation the paper imposes on them. This file describes the deposit that holds them.

| file | bytes | sha256 |
|---|---|---|
| `reference_MIX17.npz` | 865,346,461 | `3a1ff95851ebc078e77e373381457e2e0dcd4d5c05885c8c305029ca264b6fc6` |
| `targets_thrust_nlo.npz` | 30,098 | `7e125ae4e03da92255f0f2ae92b91553a86cd016b334eaf11b01ea3873a835ca` |

Both are in `deposit/` at the top of the repository, with `build_reference.json`, the record of
the build (last section), and `SHA256SUMS`, the sha256 of the three files. GitHub refuses files over
100 MB, so the reference sample is stored there in 19 parts. The commands below run in
`analysis/release` of a checkout. From there, join the parts, copy the targets file and check both:

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

Both files are plain numpy archives. `np.load(path)` opens them, with no `allow_pickle`, no torch
and no other package.

The deposit of an earlier draft had the same events and the same `AE`, but its observables kept
the photons radiated by the beams. `gentune.bands.reference_observables` warns when it reads that
file.

## What the reference is

The reference holds 1,149,840 events of e+e- to hadrons at 91.2 GeV. They are the pooled union
of the 720 training runs of MIX17, 1597 events from each:

- 352 Sherpa 3.0.4 runs at fraction 0, with the Herwig parameters at the centre of their box;
- 368 Herwig 7.3.0p1 runs at fraction 1, with the Sherpa parameters at their centre.

The classifier learned the log-ratio of every parameter point to this pooled sample. At the
fraction 0 the head gives the weights that take these events to Sherpa at theta_S, and at the
fraction 1 those that take them to Herwig at theta_H, and the paper's prediction at a point
(theta_S, theta_H, xi) mixes the two, (1 - xi) <O>_S + xi <O>_H (`gentune.mixture`, README). These
are the weights of the reference, which are correct only on reference events. These are those
events.

The file ships a(Phi), the event-side vector that is all a head reads of an event. It is `AE`,
of shape (4, 1149840, 48): one 48-vector per event and ensemble member. It is bitwise the array
of the trained export (`MIXGEOV3S9_cond.npz`), and every number of the paper was computed from
it. The alternative was the per-particle features: a user would then first have to run them
through `Trunk.embed`, about 1.7e13 multiply-adds, and the result would match the paper only to
float32 rounding, not exactly. The features were still used, to check the event order (see the
last section).

## Reweighting it

```python
import numpy as np
from gentune import load_head, axes, mixture

head = load_head('models/MIX17_head.npz')
r = np.load('reference_MIX17.npz')
A = head.pack(r['AE'], np.float64)               # the event vectors as one float64 matrix, 1.8 GB
tau = r['1_minus_thrust']

theta = axes.default('MIX17')                    # the paper's central point
w = mixture.weights(head, A, theta)              # {'sherpa': w_S, 'herwig': w_H}, each sums to one
print(mixture.mean(w, tau, theta))               # <1 - T> before the calculation: 0.0620683
print(mixture.n_eff(w))                          # N_eff of Sherpa 664,340 and of Herwig 600,619
```

That `theta` is the paper's central point: both generators at their own defaults, with
xi = 1/2. `r['theta_names']` lists the parameters in order, and `gentune.axes` says what each one
is. `IMPOSE.md` says what this takes in time and memory. `head.weights(A, theta)` at the same point
would give the network's own interpolation between the generators, which the paper does not use
(README).

Any per-event array of the file can take the place of `tau`. `r['generator']` selects the events
of one generator. `r['run_id']`, with `r['run_ids']` and `r['run_theta']`, gives the parameter
point each event was generated at.

## Imposing a calculation

The paper then imposes the calculated thrust moments with a maximum-entropy tilt, on each
generator's weights alone. Only the events in the window `lo <= tau < hi` are tilted, and the
weights are normalized over all events:

    w_i  proportional to  w0_i exp(lam . (f(tau_i) - c))   inside the window,
    w_i  proportional to  w0_i                              outside it.

Here w0 are the weights of one generator above, f are the moment functions, and c are the
calculated moments, normalized inside the window. The multipliers solve

    E_w[ (f - c) Theta(window) ] + Pw^2 Sigma lam = 0 ,

with Pw the window fraction under w0. `Sigma = sigma_pert + diag((floor_rel |c|)^2)` makes the
constraint soft. `gentune.maxent` solves this in numpy, `gentune.mixture.impose` does it for each
generator, and `targets_thrust_nlo.npz` is a targets file it reads:

```python
from gentune.maxent import load_targets

t = load_targets('targets_thrust_nlo.npz')
inwin, F = t.functions(tau)                      # the window and the moment functions
wc, info = mixture.impose(w, inwin, F, t)        # the paper's c, sigma_pert and floor, each generator alone
print(mixture.mean(wc, tau, theta))              # the paper's imposed <1 - T>: 0.0626911
wv, _ = mixture.impose(w, inwin, F, t, which="(2.0, 1.0, 'logR', 1.0)")   # one variation
```

For a calculation of your own, pass your window means and covariance as the pair `(c, Sigma)`,
`mixture.impose(w, inwin, F, (c, Sigma), floor_rel=...)`, or build a `gentune.maxent.Targets` and
save it, in a format `load_targets` reads. `F` holds one column per function, evaluated on the
window events, and `c` the calculated values, normalized inside the window. `IMPOSE.md` shows
both, the statistical error and the theory band of an imposed prediction, how to read a loaded
calculation, and why `c` has to be the moments of one distribution.

## The paper from these files

With numpy and `gentune` alone, from `analysis/release`:

```bash
python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper --targets targets_thrust_nlo.npz
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper --targets targets_thrust_nlo.npz
```

The first reproduces the paper's central predictions before and after the calculation, with
their statistical errors, learning bands and theory band, its DELPHI thrust distribution, and its
generator band over the 2000 points, and prints PASS or FAIL against the paper's numbers. The
other two give the generator band and the learning band, on the paper's observables and in the
paper's thrust bins, with the statistical error of each (`README.md`, under Bands); without
`--targets`, those of the generators alone.

The build script of the reference carries its own copy of the same step, `tilt()` in
`tools_build_reference.py`, written out in numpy step for step as the paper's
`directlib.Model.tilt_on`. Its own checks use that copy.

## The photons radiated by the beams

The observables of this file leave out the photons that the incoming e+ and e- radiate along the
beam, as every observable of the paper does. `AE` does not: the head reads every particle, as in
training. The photons are found among the stored particles, which no longer say where they came
from, by the rule of `gentune/beam_photons.py`, a copy of the paper's `analysis/beam_photons.py`
(rule version 3, stated in full in its docstring). In the thrust frame of the features, with phi
measured about the thrust axis:

- **base**: a kept particle that is massless and neutral and lies in the plane of the thrust axis
  and the beam (`||phi| - pi/2| < 1e-7`);
- **(i)**: where the base rule flags nothing, a massless neutral particle on the axis itself
  (`|cos theta| >= 1 - 1e-7`, phi 0 or pi): a hard beam photon alone in its hemisphere sets the
  axis on the beam;
- **(ii)**: a flagged photon on the axis loses its flag when it is the hardest particle of a whole
  event without missing momentum and the rest of its hemisphere carries a transverse momentum
  below 1e-6 of its energy. The axis then starts and stays on that photon, where a beam photon
  would sit at phi 0 or pi, so a photon at phi +-pi/2 there is not one;
- **(iii)**: every other flagged photon implies an angle between the axis and the beam, and one
  that differs by more than 1e-3 from that of the largest group of agreeing photons (where no two
  agree, from that of the hardest) loses its flag.

Over the 340,720,000 events of the paper's generator runs the rule flags 579,296,866 photons in
322,685,107 events. Where it can be wrong, all in Herwig: in a hemisphere with two flagged photons
that disagree, the hardest photon decides, in 656 such pairs where both are above 1 MeV and in 27
where the softer is below, and about 98 and 27 of these are expected wrong; in 106 more events whose
flags all disagree, but not as a pair in one hemisphere, the hardest decides too, with no estimate
of how often wrongly; rule (i) flags about 65 photons that do not come from the beam; 2 events keep
flags that disagree by less than 1e-3; and rule (ii) cannot see a true missing momentum below its
cut of 1e-4 GeV, and leaves out events cut at 100 particles (none of its 181 decisions met one). In
this file, 1,076,397 of the 1,149,840 events hold at least one beam photon.

`python -m gentune.beam_photons FILE.npz` runs the rule on a file with the arrays `particles` and
`mask`, such as `models/MIX17_selftest.npz`, checks it event by event and prints its counts. They
are the events with a beam photon (`with_beam_photon`) and how many hold 0, 1, 2, 3, or 4 and more
(`n_hist`); the photons of the base rule (`base_photons`) and in the end (`photons`); the events
where (i), (ii) and (iii) acted (`axis_rule_events`, `lone_events`, `conflict_events`), with
`tie_events` where the hardest photon decided; and the events cut at 100 particles (`truncated`).

## reference_MIX17.npz

N = 1,149,840. Event i of every per-event array is the event of `AE[:, i]`.

| key | shape, dtype | content |
|---|---|---|
| `AE` | (4, N, 48) float32 | a(Phi) per ensemble member and event, the input of the head (`gentune.mixture.weights`, `head.logit`) |
| `1_minus_thrust` | (N,) float64 | tau = 1 - T, with the exact thrust axis, the beam photons left out |
| `B_total` | (N,) float64 | total jet broadening B_W + B_N, each B = sum of \|p_i x n\| over a hemisphere, divided by 2 sum of \|p_i\| over the event, the beam photons left out |
| `rho_heavy` | (N,) float64 | heavy hemisphere mass M_H^2 / E_vis^2, the beam photons left out |
| `nch` | (N,) int16 | charged particles among the stored particles (at most 100, the hardest) |
| `nbaryon` | (N,) float32 | baryons plus antibaryons in the full event |
| `mult_total` | (N,) float64 | final-state particles in the full event, the beam photons left out |
| `strange` | (N,) float32 | fraction of the stored particles other than the beam photons with 10^-0.36 < m/GeV < 10^-0.20 (the kaons) |
| `tau_parton` | (N,) float32 | 1 - T of the end-of-shower partons, NaN where there are none |
| `n_parton` | (N,) int16 | number of end-of-shower partons |
| `run_id` | (N,) int16 | the training run of each event |
| `generator` | (N,) int8 | 0 Sherpa, 1 Herwig, an index into `generator_names` |
| `generator_names` | (2,) str | `Sherpa 3.0.4`, `Herwig 7.3.0p1` |
| `run_ids` | (720,) int64 | the runs in reference order; event i belongs to run `run_ids[i // nper]` |
| `run_theta` | (720, 17) float64 | the 17 parameters of each run, physical units, columns `theta_names` |
| `run_generator` | (720,) int8 | the generator of each run |
| `theta_names` | (17,) str | `S:alphas` ... `S:beta_l`, `H:alpha_fsr` ... `H:clsmr`, `fraction`, as `gentune.axes.NAMES['MIX17']` |
| `nper` | () int64 | 1597, the events per run |
| `meta` | () str | JSON: the event order, the source files and their sha256, the sha256 of every array, a line on every key, how the paper mixes the generators (`mixture`), the checks of its build (last section) |

The event shapes are dimensionless, and the masses are in GeV. Final state means every stable
particle except neutrinos, with hadrons of c tau above 10 mm kept stable in both generators. Only
622 events reach the limit of 100 stored particles, so `nch` and `strange` see the full event almost
everywhere.

The observables are those of the paper. The thrust histograms and the window use
`1_minus_thrust`, and the means of the paper's bands use it together with `B_total`,
`rho_heavy`, `nch` and `nbaryon`. The keys carry the names the paper's own code reads, so
`directlib.Model` reads the file as its reference bundle.

The order is run after run in `run_ids`, the training order of the run list. Within run r, the
events are `numpy.random.default_rng(50000 + r).choice(80000, 1597, replace=False)` of the run's
80,000 training events, in the order drawn.

To check a file, compare its sha256 with the table above. To check one array, compare
`hashlib.sha256(np.ascontiguousarray(r[key]).tobytes()).hexdigest()` with
`json.loads(str(r['meta']))['arrays'][key]['sha256']`.

## targets_thrust_nlo.npz

This is the paper's NNLL+NLO thrust calculation, with a dispersive power correction. It is
evaluated at the published pair (alpha_s, alpha_0) of Gehrmann, Luisoni and Monni (Eur. Phys. J. C 73
(2013) 2265, log-R), with their renormalon subtraction to order alpha_s^3. Every
number is copied unchanged from `output/thrust_targets_nlo_point.npz`, and the build checks this
bitwise.

| key | shape, dtype | content |
|---|---|---|
| `moment_names` | (14,) str | `logt tau tlogt log2t tau2 tlog2t t2logt log3t tau3 tlog3t t3logt t2log2t log4t tau4` |
| `moment_powers` | (14, 2) int64 | (a, b) of each moment, f(tau) = tau^a (ln tau)^b, with tau clipped at 1e-6 |
| `central` | (14,) float64 | c_k = <f_k(tau)> over the window, normalized inside the window |
| `sigma_pert` | (14, 14) float64 | Sigma_c, the sum of d d^T with d a scale variation minus `central` (see below) |
| `window` | (2,) float64 | [0.05, 1/3), in tau |
| `floor_rel` | () float64 | 0.002, the paper's diagonal floor: Sigma = sigma_pert + diag((floor_rel \|c\|)^2) |
| `alpha_s`, `alpha_0` | () float64 | 0.1137 and 0.524, the point |
| `pwin` | () float64 | the calculated window fraction, 0.4486, which the paper does not impose |
| `scale_variations` | (9, 14) float64 | the moments for each scale and scheme variation, at the same point |
| `scale_labels` | (9,) str | (x_muR, x_V, matching scheme, x_NNLO) of each row |
| `scale_pwin` | (9,) float64 | the window fraction of each row |
| `np_variations` | (3, 14) float64 | the moments with the Milan factor up and down by 20 percent, and with the renormalon subtraction at second order (the central one is at third order) |
| `np_labels` | (3,) str | `milan_up milan_dn sub2` |
| `milan`, `sub_order_central`, `order` | () | 1.49, 3 and 2: the central Milan factor, the subtraction order and the fixed-order order |
| `calc_point` | () str | JSON of the point and its source |
| `meta` | () str | JSON: the source and its sha256, the sha256 of every array, over its dtype, shape and bytes in that order (`sha_array` in `tools_build_targets.py`), a line on every key |

`sigma_pert` sums d d^T over three groups of scale variations:

- the log-R pair with muR times 2 and times 1/2, each at weight 1/2;
- the log-R pair with x_V times 2 and times 1/2, each at weight 1/2;
- the mod-R scheme at central scales, at weight 1.

## How it was made, and what was checked

`reference_MIX17.npz` was built by `tools_build_reference.py`, whose sha256 its `meta` names
(`build_script_sha256`), with `BEAM_PHOTONS=remove`: every source from the paper's run, the
observables without the beam photons. It ran on Symmetry, as `build_reference.py` in a folder of
its own, by `analysis/symmetry/release/refbuild_lmx.sbatch`, with numpy 2.2.4 and no torch, in 141
seconds. `targets_thrust_nlo.npz` is the file of the earlier deposit, kept bitwise, because the
calculation did not change: the build checks that it holds, bitwise, the calculation of
`output/thrust_targets_nlo_point.npz`, and its own `meta` names `tools_build_targets.py`, which
built it, by sha256.

The reference build copies its arrays from these files:

- the training export, `MIXGEOV3S9_cond.npz`, at the calibration constant the paper's run fitted
  on the photon-free observables (`output/temperature/MIXGEOV3S9.json`);
- the paper's reference bundle, `MIXGEOV3S9_ref_v2_slim.npz`, which holds the observables with the
  exact thrust axis and without the beam photons, made with the rule above and recorded with it;
- the run list, `data_stagePURE17_v3/meta.json`, with the runs and their parameters.

Before writing anything, it checks the following.

- **Provenance.** The released `MIX17_head.npz` is bitwise the export in all 92 arrays they share.
  The observables are in the training order of the runs. Three runs, the first, the first Herwig and
  the last, are replayed from their particle files and give the stored reference events bitwise.
- **Event order.** The released trunk runs on the stored per-particle features of 8 events of every
  run, 5760 events. It reproduces `AE` to 4.0e-5, where |a| reaches 15.9. The same events shifted by
  one event miss by 15.8. The resulting logits at the central point, at the fractions 0 and 1,
  differ by 3.7e-5. The charged multiplicity, and without the beam photons the total multiplicity
  and the kaon fraction, recomputed from the particles, equal `nch`, `mult_total` and `strange` on
  all 1,149,840 events.
- **Logits.** At the 2000 `reference_index` events of `MIX17_selftest.npz`, its `AE` equals the
  shipped one exactly. On the shipped `AE`, the released head reproduces three sets of logits: the
  training run's own 18 x 2000 logits exactly, the self-test's logits to 1.3e-5, and the export's
  256 checksum logits to 1.3e-7.
- **The paper's bands.** The stored bands it compares with are those of the paper: the linear
  mixture, the 2000 points of the Latin hypercube with seed 11 over the whole box of the export
  (the same points to 1e-12 of a half width), the central point at the default tunes of both
  generators and xi = 1/2, and the comparison with DELPHI. The `bands` entry of the checks records
  this.
- **The paper.** At the paper's central point, numpy alone on the reference file and the targets
  file reproduces the stored means and DELPHI histograms of the paper's run
  (`output/three_bands_MIXGEOV3S9_lin`), of each generator and of the mixture, before and after the
  calculation is imposed on each generator alone: the means to 2.6e-9 and the histograms to 1.2e-8,
  relative. The tilt acts on 535,993 window events and converges in 3 Newton steps for Sherpa and 3
  for Herwig. The paper's torch code ran the head in float32 and this check runs it in float64,
  which accounts for the residual.

After writing, the script read the file back, checked every array against its sha256 in `meta`,
and ran the paper check again on what it read. Every array was then compared with the copy of the
same file that the paper's run had refreshed when it removed the beam photons from the
observables, and all were equal. The full record of the build is in that `meta` and in
`build_reference.json`, which `deposit/` holds next to the two files.

The `meta` of `targets_thrust_nlo.npz` holds no checks. `tools_build_targets.py` checks that the
moments, their variations and `sigma_pert` are a bitwise copy of its source and that `sigma_pert` is
the sum given above.

`examples/reproduce_paper.py` checks the two deposited files together, on every number of the
paper's central point and generator band (`README.md`, Against the paper).
