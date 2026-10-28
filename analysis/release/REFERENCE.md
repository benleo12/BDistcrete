# The MIX17 reference sample

`models/MIX17_head.npz` reweights one fixed sample of events, the reference, to any point of its
seventeen-parameter box. The weights only mean something on those events, and so does the
calculation the paper imposes on them. This file describes the deposit that holds them.

| file | bytes | sha256 |
|---|---|---|
| `reference_MIX17.npz` | 865,353,418 | `a22b834d357f5fb954d07346436ce7fd85c6cd6450d918f27961cd17fdf73bfe` |
| `targets_thrust_nlo.npz` | 33,524 | `765f6d755c1a56fe039f173571371b6be63e30c15061f76f46444e49405beb3b` |

Both are public, with this file and `SHA256SUMS`, as the assets of the GitHub release
https://github.com/benleo12/BDistcrete/releases/tag/data-r8, and will also be archived at Zenodo.
In `analysis/release` of a checkout, where the examples look for them:

```bash
for f in reference_MIX17.npz targets_thrust_nlo.npz REFERENCE.md SHA256SUMS; do
  curl -LO https://github.com/benleo12/BDistcrete/releases/download/data-r8/$f
done
shasum -a 256 -c SHA256SUMS          # sha256sum -c SHA256SUMS on Linux; every line must say OK
```

Both files are plain numpy archives. `np.load(path)` opens them, with no `allow_pickle`, no torch
and no other package.

## What the reference is

The reference holds 1,149,840 events of e+e- to hadrons at 91.2 GeV. They are the pooled union
of the 720 training runs of MIX17, 1597 events from each:

- 352 Sherpa 3.0.4 runs at fraction 0, with the Herwig parameters at the centre of their box;
- 368 Herwig 7.3.0p1 runs at fraction 1, with the Sherpa parameters at their centre.

The classifier learned the log-ratio of every parameter point to this pooled sample, so
`head.weights(AE, theta)` carries it to any `theta` in the box. That is the `weights` of the
README, the one that is correct only on reference events. These are those events.

The file ships a(Phi), the event-side vector that is all a head reads of an event. It is `AE`,
of shape (4, 1149840, 48): one 48-vector per event and ensemble member. It is bitwise the array
of the trained export (`MIXGEOV3S9_cond.npz`), and every number of the paper was computed from
it.

The alternative was the per-particle features, at 0.62 GB instead of 0.83 GB compressed. A user
would then first have to run them through `Trunk.embed`, about 1.7e13 multiply-adds, and the
result would match the paper only to float32 rounding (4e-5 on the vectors), not exactly. The
features were still used, to check the event order (see the last section).

## Reweighting it

```python
import numpy as np
from gentune import load_head

head = load_head('models/MIX17_head.npz')
r = np.load('reference_MIX17.npz')
AE, tau = r['AE'], r['1_minus_thrust']           # AE takes 0.88 GB of memory

theta = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18,                      # Sherpa
         0.102337, 0.654714, 3.528693, 1.849375, 0.914156, 0.374094, 0.33107, 0.78,  # Herwig
         0.5]                                                                  # fraction
w = head.weights(AE, theta)                      # sums to one over the 1,149,840 events
print(w @ tau, head.n_eff(w))                    # <1 - T> before the calculation 0.0648321, N_eff 641,406
```

That `theta` is the paper's central point: both generators at their own defaults, with equal
weight. A head evaluation over the whole sample needs about 3 GB of memory and a few seconds.
`r['theta_names']` lists the parameters in order, and `gentune.axes` says what each one is.

Any per-event array of the file can take the place of `tau`. `r['generator']` selects the events
of one generator. `r['run_id']`, with `r['run_ids']` and `r['run_theta']`, gives the parameter
point each event was generated at.

## Imposing a calculation

The paper then imposes the calculated thrust moments with a maximum-entropy tilt. Only the
events in the window `lo <= tau < hi` are tilted, and the weights are normalized over all events:

    w_i  proportional to  w0_i exp(lam . (f(tau_i) - c))   inside the window,
    w_i  proportional to  w0_i                              outside it.

Here w0 are the weights above, f are the moment functions, and c are the calculated moments,
normalized inside the window. The multipliers solve

    E_w[ (f - c) Theta(window) ] + Pw^2 Sigma lam = 0 ,

with Pw the window fraction under w0. `Sigma = sigma_pert + diag((floor_rel |c|)^2)` makes the
constraint soft. `gentune.maxent` solves this in numpy, and `targets_thrust_nlo.npz` is a targets
file it reads:

```python
from gentune.maxent import load_targets

t = load_targets('targets_thrust_nlo.npz')
inwin, F = t.functions(tau)                      # the window and the moment functions
f = head.logit(AE, theta)/head.T                 # log w0, any normalization
lw, lam, info = t.impose(f, inwin, F)            # the paper's c, sigma_pert and floor
w = np.exp(lw)
print(w @ tau)                                   # the paper's imposed <1 - T>: 0.0651102
lw_up, _, _ = t.impose(f, inwin, F, which="(2.0, 1.0, 'logR', 1.0)")   # one variation
```

For a calculation of your own, pass your window means and covariance to
`gentune.maxent.impose(f, inwin, F, c, Sigma, floor_rel=...)`. Alternatively, build a
`gentune.maxent.Targets` and save it, in a format `load_targets` reads. `F` holds one column per
function, evaluated on the window events, and `c` the calculated values, normalized inside the
window. `IMPOSE.md` shows both, how to read a loaded calculation, and why `c` has to be the
moments of one distribution.

## The paper from these files

With numpy and `gentune` alone, from `analysis/release`:

```bash
python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/learning_band.py  --stage MIX17 --reference-file reference_MIX17.npz --bins paper
```

The first reproduces the paper's central predictions before and after the calculation, its
DELPHI thrust distribution and its generator band over the 200 points, and prints PASS or FAIL
against the paper's numbers. The other two give the generator band and the learning band of the
generators alone, on the paper's observables and in the paper's thrust bins, with the
statistical error of each (`README.md`, under Bands).

The build script carries its own implementation of the same step, `tilt()` in
`tools_build_reference.py`, written independently from the paper's `directlib.Model.tilt_on`.
That is the one the checks below use.

## reference_MIX17.npz

N = 1,149,840. Event i of every per-event array is the event of `AE[:, i]`.

| key | shape, dtype | content |
|---|---|---|
| `AE` | (4, N, 48) float32 | a(Phi) per ensemble member and event, the input of `head.weights` and `head.logit` |
| `1_minus_thrust` | (N,) float64 | tau = 1 - T, with the exact thrust axis |
| `B_total` | (N,) float64 | total jet broadening B_W + B_N, each B = sum \|p_i x n\| / (2 sum \|p_i\|) over a hemisphere |
| `rho_heavy` | (N,) float64 | heavy hemisphere mass M_H^2 / E_vis^2 |
| `nch` | (N,) int16 | charged particles among the stored particles (at most 100, the hardest) |
| `nbaryon` | (N,) float32 | baryons plus antibaryons in the full event |
| `mult_total` | (N,) float64 | final-state particles in the full event |
| `strange` | (N,) float32 | fraction of the stored particles with 10^-0.36 < m/GeV < 10^-0.20 (the kaons) |
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
| `meta` | () str | JSON: the event order, the source files and their sha256, the sha256 of every array, a line on every key, the checks below |

The event shapes are dimensionless, and the masses are in GeV. Final state means every stable
particle except neutrinos, with hadrons of c tau above 10 mm kept stable in both generators.
Only 622 events reach the limit of 100 stored particles, so `nch` and `strange` see the full
event almost everywhere.

The observables are those of the paper. The thrust histograms and the window use
`1_minus_thrust`, and the means of the paper's bands use it together with `B_total`,
`rho_heavy`, `nch` and `nbaryon`. The keys carry the names the paper's own code reads, so
`directlib.Model` takes the file as its reference bundle unchanged.

The order is run after run in `run_ids`, the training order of the run list. Within run r, the
events are `numpy.random.default_rng(50000 + r).choice(80000, 1597, replace=False)` of the run's
80,000 training events, in the order drawn.

To check a download, compare its sha256 with the table above. To check one array, compare
`hashlib.sha256(np.ascontiguousarray(r[key]).tobytes()).hexdigest()` with
`json.loads(str(r['meta']))['arrays'][key]['sha256']`.

## targets_thrust_nlo.npz

This is the paper's NNLL+NLO thrust calculation, with a dispersive power correction. It is
evaluated at the pair (alpha_s, alpha_0) of its own fit to ALEPH in 0.05 < tau < 0.33. Every
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
| `alpha_s`, `alpha_0` | () float64 | 0.113502 and 0.431222, the point |
| `pwin` | () float64 | the calculated window fraction, 0.4392, which the paper does not impose |
| `scale_variations` | (9, 14) float64 | the moments for each scale and scheme variation, at the same point |
| `scale_labels` | (9,) str | (x_muR, x_V, matching scheme, x_NNLO) of each row |
| `scale_pwin` | (9,) float64 | the window fraction of each row |
| `np_variations` | (3, 14) float64 | the moments with the Milan factor up and down by 20 percent, and with the third-order renormalon subtraction |
| `np_labels` | (3,) str | `milan_up milan_dn sub3` |
| `milan`, `sub_order_central`, `order` | () | 1.49, 2 and 2: the central Milan factor, the subtraction order and the fixed-order order |
| `chain_summary` | () str | JSON of the fit that fixed the point |
| `meta` | () str | JSON: the source and its sha256, the sha256 of every array, a line on every key |

`sigma_pert` sums d d^T over three groups of scale variations:

- the log-R pair with muR times 2 and times 1/2, each at weight 1/2;
- the log-R pair with x_V times 2 and times 1/2, each at weight 1/2;
- the mod-R scheme at central scales, at weight 1.

## How it was made, and what was checked

`tools_build_reference.py` is the exact script that built both files. It ran on Symmetry, in
`/mnt/beegfs/bassi/reweighting/release_ref` as `build_reference.py`, with numpy 2.2.4 and no
torch. It copies its arrays from these files:

- the training export, `MIXGEOV3S9_cond.npz`;
- the paper's reference bundle, `MIXGEOV3S9_ref_v2_slim.npz`, which holds the observables with
  the exact thrust axis;
- the run list, `data_stagePURE17_v3/meta.json`, with the runs and their parameters;
- the calculation, `thrust_targets_nlo_point.npz`.

Before writing anything, it checks the following.

- **Provenance.** The released `MIX17_head.npz` is bitwise the export in all 92 arrays they
  share. The observables are in the training order of the runs. Three runs, the first, the
  first Herwig and the last, are replayed from their particle files and give the stored
  reference events bitwise.
- **Event order.** The released trunk runs on the stored per-particle features of 8 events of
  every run, 5760 events. It reproduces `AE` to 4.0e-5, where |a| reaches 15.9. The same events
  shifted by one event miss by 15.8. The resulting logits at the central point differ by
  1.8e-5. The charged multiplicity and the kaon fraction, recomputed from the particles, equal
  `nch` and `strange` on all 1,149,840 events.
- **Logits.** At the 2000 `reference_index` events of `MIX17_selftest.npz`, its `AE` equals the
  shipped one exactly. On the shipped `AE`, the released head reproduces three sets of logits:
  the training run's own 18 x 2000 logits exactly, the self-test's logits to 1.3e-5, and the
  export's 256 checksum logits to 1.3e-7.
- **The paper.** At the paper's central point, numpy alone on the two files reproduces the stored
  bands, `output/three_bands_MIXGEOV3S9`. The seven means of the paper are the window mean and
  fraction, <1 - T>, <B_T>, <rho_H>, <n_ch> and <n_baryon>. They agree to 1.0e-9, and the
  DELPHI thrust histogram to 6.9e-9, relative, both before and after the calculation is imposed.
  The tilt acts on 545,623 window events and converges in 4 Newton steps. The paper's torch code
  ran the head in float32 and this check runs it in float64, which accounts for the residual.

After writing, the script reads both files back and checks every array against its sha256 in
`meta`. It then repeats the paper check on what it read. The full record is in each file's
`meta` and in `build_reference.json`, next to the files on Symmetry.
