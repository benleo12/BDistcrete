# BDistcrete

Code for *Combined Uncertainties from Event Generators and Precision QCD* (B. Assi and J. Thaler),
earlier circulated as *Beyond Discrete Variations*. Every section, figure and table number below
is that of the paper. The LaTeX source of the paper is kept in a separate repository
(benleo12/Reweighting-generator-parameters, private until publication). Each block of commands
below starts at the top of a checkout, `git clone https://github.com/benleo12/BDistcrete`.

## What the paper does

A generator prediction depends on the values of the generator parameters and on the choice of
generator, and a weight learned from finite samples has an uncertainty of its own. The paper
obtains these from one stored event sample of e+e- → hadrons at the Z pole. As an application, it
then imposes a precision calculation on the same events.

- **Classifier weight.** As in DCTR, a classifier that takes the generator parameters as an input
  reweights the sample to any point of a parameter box. The paper writes the weight as the
  exponential of an inner product of stored functions of the event with functions of the
  parameters, the factorized network. A new point then costs one matrix-vector product, and the
  derivative of any prediction is a covariance over the same events. Sherpa and Herwig enter as a
  multiplicative mixture, a weighted geometric mean of the two densities with a continuous mixing
  fraction ξ, so there are seventeen parameters: eight of Sherpa, eight of Herwig and ξ. This is a
  physics choice: most of the difference between two parton-shower generators is expected to lie in
  their Sudakov factors, and the geometric mean interpolates the exponents of these factors.
- **Learning band.** Each model is trained as an ensemble of four members, from different starting
  values and on resamples of the training runs. A fit of the log ratio as a combination of the
  members' products, on events set aside from training (WiFi ensembles, Benevedes and Thaler,
  arXiv:2506.00113), gives the covariance of the classifier weight. The prediction is the ensemble
  average.
- **Generator band.** Each parameter, ξ included, is flat on its range, which makes the prior a
  standard Gaussian in the variables z of Eq. (4.1). The generator band of a prediction is its
  standard deviation under this prior.
- **Linear covariance and 400 variations.** The derivatives at one point give the covariance of any
  set of predictions at linear order. The paper compares it with 400 variations, each an exact
  reweighting of the sample to one draw from the prior, for the thrust and charged-multiplicity
  distributions and seven means.
- **Application and theory band.** A maximum-entropy weight imposes fourteen moments of an NNLL+NLO
  thrust calculation with a dispersive power correction on the same events, inside the window
  0.05 ≤ τ < 1/3. The scale, scheme and dispersive-model variations of the calculation give a
  theory band that keeps the sign of each variation. Here the generator band comes from the
  predictions at 200 points of the box.

The paper performs no fit of the strong coupling. The two parameters of the calculation are fixed
inputs, α_s(M_Z) = 0.1137 and α_0 = 0.524, the result of the published log-R fit of Gehrmann,
Luisoni and Monni (`analysis/output/calc_point.json`). The predictions are compared with DELPHI.
Section 3 tests the classifier weights against fresh generator runs for one to seventeen
parameters, and the learning band for seventeen parameters. Section 4 writes the prior on the box,
derives the linear covariance and compares it with 400 variations. Section 5 imposes the calculation
and shows the three bands of the thrust distribution and of seven means.

## Quick start: the released networks, numpy only

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
cd analysis/release
python -m gentune.selftest models
python examples/reweight_own_events.py --stage MIX17 --hepmc examples/data/herwig_default.hepmc.gz \
    --from herwig-default --to herwig-default H:pwtdiquark=0.40 --out weights.npz
python examples/generator_band.py --stage MIX17 --reference
python examples/learning_band.py --stage MIX17 --reference
```

The self-test checks the networks, and the reading of HepMC files, against stored outputs in
about ten seconds, longer on the first run. The second command writes one weight per event that
takes the Herwig example file from Herwig's default tune to a larger diquark weight, in the order
of the file and summing to one, and prints the effective sample size. For your own sample, pass
its file and the point it was generated at. The last two give the generator band of the paper,
over its 200 points of the box, and the learning band at its central point, on the 2000 reference
events that ship with the networks, with the statistical error of each. Those errors are about 24
times the paper's.

With the paper's reference sample and calculation, the two public files of Data (below),
downloaded into `analysis/release`, the same scripts, in the same environment, give the paper's bands on its observables
and in its thrust bins, and `reproduce_paper.py` its central predictions and generator band,
before and after the calculation is imposed, checked against the paper's numbers:

```bash
cd analysis/release
python examples/generator_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/learning_band.py --stage MIX17 --reference-file reference_MIX17.npz --bins paper
python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz
```

[`analysis/release/README.md`](analysis/release/README.md) describes all of it, including what
your sample has to match,
[`analysis/release/IMPOSE.md`](analysis/release/IMPOSE.md) how to impose a calculation in numpy,
[`analysis/release/REFERENCE.md`](analysis/release/REFERENCE.md) the reference sample and the
calculation, and
[`analysis/release/examples/reproduce_paper.py`](analysis/release/examples/reproduce_paper.py)
what it checks.

## Quick start: the numbers and figures of the paper

```bash
conda env create -f environment.yml && conda activate bdistcrete
cd analysis
python phase2tools/make_numbers.py output
python figs_r8.py
python plot_envelope.py output/envelope_test_MIXGEOV3S9 tune full
```

`make_numbers.py` reads the result files in `analysis/output/` and
`analysis/data_stagePURE17/meta.json`, and writes
`analysis/phase2tools/numbers.tex`, 624 macros identical to the paper's `numbers.tex`, and
`numbers.json` next to it. `figs_r8.py` draws nine data figures, Figs. 3 to 7 and 11 to 14, into
`output/r8_<name>.pdf`, and with names it draws only those figures, for example
`python figs_r8.py bands obs`. The `aleph` figure prints a few divide-by-zero warnings, which do no
harm. `plot_envelope.py` draws Figs. 8 to 10 into the same `output/` folder, and also per-box figures
that are not in the paper. Figs. 1 and 2 are drawn in TikZ in the paper source. The figures are
typeset with LaTeX, which `environment.yml` does not install; a full TeX Live has everything they
need.

## What can be redone from where

1. **From this repository alone, on a laptop.** Every quoted number and every figure, from the
   result files in `analysis/output/`. The thrust calculation at the fixed point and its comparison
   with ALEPH and DELPHI (`targets_nlo_point.py`, `calc_bins.py`, `calc_vs_delphi.py`, at the point
   of `output/calc_point.json`, with `RIVET_REF=rivet_ref`; the commands are in `docs/REPRODUCE.md`).
   The collection of the scan of the number of terms (`kscan_collect.py`). The self-test of the
   released networks, the reweighting of your own events, and the generator and learning bands on
   the 2000 reference events that ship with them (`analysis/release/`).
2. **With the public reference sample of the mixture** (Data, below), with numpy alone. The
   central predictions and the generator band of Sec. 5, before and after the calculation is
   imposed, and the learning band of the generators alone, on the paper's reference events. They
   agree with the paper to a few 1e-8 relative or better, the learning band to 1e-5. Also the scale
   and scheme displacements of the calculation (`analysis/release/IMPOSE.md`). The paper's signed
   theory band is built in `analysis/three_bands.py`.
3. **With the network exports and event samples of the paper** ([`docs/DATA.md`](docs/DATA.md)).
   The three bands, the learning band and its coverage, the comparison with 400 variations
   (`envelope_test.py`), the inputs of Figs. 3 to 5, the strict tests, the cost benchmark and the
   packaging of the released networks. These ran on the Symmetry cluster at Perimeter, with the
   batch scripts in `analysis/symmetry/`.
4. **From the generators.** Every sample can be regenerated with the scripts and designs in `gen/`.
   Training needs the samples and a GPU, one NVIDIA H200 per job.

The toy model of Fig. 13 (App. B.1) needs no data, because `toy_final.py` generates its own events.
Its training ran on a GPU (`symmetry/toy_final.sbatch`), and `TOY_K=48 TOY_SEED=0 python toy_final.py`
runs one case directly.

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) gives the command and the inputs of every figure, table
and number.

## The pipeline of the paper

All scripts are in `analysis/` and run from there.

| what | where |
|---|---|
| training, both networks, every model | `r2_ladder.py` (`CondPFN`, `GeoMixturePFN`, `ConcatPFN`, `ConcatGeoMixture`), `stage_mixture.py`, `symmetry/train_p2.sbatch`, `symmetry/train_mixgeo.sbatch` |
| closure on the standard and the strict test | `ladder_widths_big.py`, `eval_valid.py`, `symmetry/score_p2_array.sbatch`, `symmetry/strict_closure.sbatch` |
| the learning band and its coverage | `wifi_embed.py`, `wifi_fit.py`, `wifi_diag.py`, `wifi_coverage.py`, `symmetry/wifi_cov.sbatch` |
| the calculation at the fixed point | `output/calc_point.json`, `targets_nlo_point.py`, `calc_bins.py`, `calc_vs_delphi.py`, which import the chain of `thrust_chain.py` |
| the comparison with 400 variations | `envelope_test.py`, `symmetry/envelope_test.sbatch`, `plot_envelope.py` |
| the maximum-entropy weights and the three bands | `directlib.py`, `three_bands.py`, `theory_variations.py`, `band_checks.py` |
| the toy, the scans and the cost | `toy_final.py`, `kscan_submit.sh`, `kscan_collect.py`, `widebox_submit.sh`, `widebox_collect.py`, `bench_cost_final.py` |
| every quoted number, every data figure | `phase2tools/make_numbers.py`, `figs_r8.py`, `plot_envelope.py` |
| the released networks | `package_release.py`, `make_stage_table.py`, `precision_gap.py`, `closure_vs_distance.py`, `fit_accuracy_law.py`, run by `symmetry/package_r8.sbatch`, `table_r8.sbatch` and `accuracy_r8.sbatch` |

## Contents

| path | what it holds |
|---|---|
| `analysis/` | every script of the paper, run from this directory |
| `analysis/output/` | the result files the numbers and figures are made from. `figs_r8.py` and `plot_envelope.py` write the figures here |
| `analysis/phase2tools/` | `make_numbers.py` and its output, `numbers.tex` and `numbers.json` |
| `analysis/symmetry/` | the batch scripts of the paper, as run on the Symmetry cluster |
| `analysis/release/` | the released networks (numpy only), the `gentune` reader, the examples and example HepMC files, the generator scripts and designs, the closure tables |
| `gen/` | the scripts and designs that generated the samples of the paper |
| `analysis/lepcov/`, `analysis/rivet_ref/` | the LEP measurements (HEPData tables and the YODA files of Rivet) and their reader |
| `analysis/ares_recovered/`, `analysis/eerad3/` | our drivers and patches for ARES and EERAD3, and the ARES tables. The EERAD3 coefficients are `output/eerad_*.npy` |
| `analysis/perlmutter/`, `analysis/farm/` | batch scripts of the earlier draft |
| `docs/REPRODUCE.md` | every figure, table and quoted number, with the command and the inputs that produce it |
| `docs/DATA.md` | the network exports and event samples, which are too large for git, and the download of the public ones |
| `tools/get_data.sh` | downloads the data archives of the earlier draft, for now only `A0_theory_logs` (`docs/DATA.md`) |
| `CLAUDE.md` | orientation for Claude Code: the method, how it relates to DCTR, the map of the code |
| `environment.yml`, `environment-tf.yml` | the analysis environment, and the TensorFlow one that only the energyflow cross-check needs |

## Data

The network exports and event samples are too large for git. [`docs/DATA.md`](docs/DATA.md)
lists the files the paper needs and what reads each one.

The reference sample of the mixture, `reference_MIX17.npz` (865 MB, the 1,149,840 events the
released `MIX17` network reweights, with the paper's per-event observables), and the paper's
thrust calculation, `targets_thrust_nlo.npz`, are public, as the assets of the GitHub release
https://github.com/benleo12/BDistcrete/releases/tag/data-r8, with `REFERENCE.md`, which describes
them, and `SHA256SUMS`. They will also be archived at Zenodo. To download and check them in
`analysis/release`, where the commands above run:

```bash
cd analysis/release
for f in reference_MIX17.npz targets_thrust_nlo.npz REFERENCE.md SHA256SUMS; do
  curl -fLO https://github.com/benleo12/BDistcrete/releases/download/data-r8/$f
done
shasum -a 256 -c SHA256SUMS
```

Every line must say OK. On Linux, use `sha256sum -c SHA256SUMS`.

Only the other exports and samples of the paper are available on request, and they will be
archived with the paper at Zenodo. The result files made from them are in `analysis/output/`, so
every number and figure can be redone without them.

## Batch scripts

The scripts in `analysis/symmetry/` are the ones that ran on Symmetry, one H200 per training job.
Each one changes to `${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}`, the cluster's copy
of `analysis/`, and takes its parent as the folder that holds `gen/` and the Python environment
`env/torch`. To run them elsewhere, set `MAXENT_DIR` to your `analysis/` folder and submit from
there:

```bash
cd analysis
export MAXENT_DIR=$PWD
TAG=MIXGEOV3S9 sbatch --export=ALL symmetry/three_bands.sbatch
```

The logs go to `logs/` in the folder you submit from. The `#SBATCH` lines name the partitions of
Symmetry, so adapt them to your cluster. The scripts set the settings of the paper that
`r2_ladder.py` does not use by default (four bootstrap members, a fifth of each run set aside,
SiLU). The number of terms and of steps come from `K` and `STEPS`, so pass `K=48 STEPS=324000` as
in `docs/REPRODUCE.md`.
`gen/README.md` describes the generation scripts.

## The released networks

`analysis/release/` holds the released networks: `E` (Stage D, Sherpa), `F` (Stage E, Herwig) and
`MIX17` (the seventeen-parameter mixture of Sec. 3.3, its first ensemble, used in Secs. 4 and 5),
with the coefficients of its learning band and their covariance in `MIX17_wifi.npz`. `C_1M` is a
three-parameter network of the earlier draft. They need only numpy, and
`python -m gentune.selftest models`, run in `analysis/release/`, checks them and the reading of two
example HepMC files, one per generator, against stored outputs. The reference sample of `MIX17` and
the paper's calculation are the release `data-r8` (Data).
[`analysis/release/README.md`](analysis/release/README.md) describes how to use them.

`pip install 'git+https://github.com/benleo12/BDistcrete#subdirectory=analysis/release'` installs
`gentune` alone. The networks and examples still come from a checkout.

## Notes on exact reproduction

- The run number of every sample is its generator seed. The one exception is run 32077, generated
  with seed 5032077 (`gen/designs/seed_overrides.csv`). The four extensions of a held-out run use
  the seeds run + k × 10^6, k = 1 to 4.
- Every event shape is computed with the exact thrust axis (`compute_efps.py`). The network inputs
  use the axis of an iterative search (`extract_particles_full_flavor.py`), as App. A says.
- The scripts that read the LEP measurements look for them in a local Rivet installation unless
  `RIVET_REF=rivet_ref` points them to the copies in `analysis/rivet_ref/`.
- External codes are not redistributed: Sherpa 3.0.4 and Herwig 7.3.0 (pinned CVMFS builds, see
  `analysis/release/generators/README.md`), ARES (GPL-3, upstream commit e75c88c with the edits in
  `analysis/ares_recovered/ares_edits.txt`) and EERAD3 1.0 (with `analysis/eerad3/eerad3_blk.patch`).

## Earlier draft with an α_s fit (not in the paper)

The earlier draft fitted α_s(M_Z) and α_0 to the LEP thrust distributions with the seventeen generator parameters profiled, on networks trained to an
earlier standard. The paper has no such fit, and every one of its results comes from one training
standard. The code of the earlier draft stays in the repository: `make_sec6_numbers.py`, the
profile and fit scripts (`profile_*.py`, `bands_exact.py`, `surface_rows.py`, `pseudo_rows*.py`),
the mixture trained on 208 runs (tags `MIXGEO`, `MIXGEOpm`, `MIXGEO8`, `MIXGEO16`, `MIXGEO32`,
`run_mixgeo_local.sh`), its figures (`figs_pro.py`,
`fig_three_bands*.py`, `fig_anchored_fit.py`) and the batch scripts in `analysis/perlmutter/` and
`analysis/farm/`. Its result files are in `analysis/output/`, and its data are the release
`data-v1` (`docs/DATA.md`). Its seed study is superseded by the three ensembles of every
model in Table 1. The instructions for it are in the git history, for example
`git show 1d66811:docs/REPRODUCE.md`.

## License

The code is released under the MIT License (`LICENSE`). ARES and EERAD3 are not included and keep
their own licenses.

## Citation

See `analysis/release/CITATION.bib`.
