# BDistcrete

Code for *Combined Uncertainties from Event Generators and Precision QCD* (B. Assi and J. Thaler),
earlier circulated as *Beyond Discrete Variations*. This README describes draft r8 of the paper,
and every section, figure and table number below is that of draft r8. The LaTeX source of the
paper is kept in a separate repository synced from Overleaf
(benleo12/Reweighting-generator-parameters, private until publication).

## What the paper does

An event-generator prediction has three kinds of uncertainty. The paper obtains all three from one
stored event sample of e+e- → hadrons at the Z pole and shows each one separately and combined.

- **Generator band.** A classifier that takes the generator parameters as an input reweights the
  sample to any point of a parameter box. The estimator is that of DCTR. Its output is written as
  an inner product of stored functions of the event with functions of the parameters, the
  factorized network. Sherpa and Herwig enter as a multiplicative mixture with a continuous mixing
  fraction ξ, so the family has seventeen parameters: eight of Sherpa, eight of Herwig and ξ. The
  generator band is the spread of a prediction over 200 points of the box.
- **Learning band.** Each model is trained as four members, from different starting values and on
  resamples of the training runs. A fit of the log ratio as a combination of the members' products,
  on events set aside from training (WiFi ensembles, Benevedes and Thaler, arXiv:2506.00113), gives
  the covariance of the learned weight. The prediction is the mean of members.
- **Theory band.** A maximum-entropy weight imposes fourteen moments of an NNLL+NLO thrust
  calculation with a dispersive power correction on the same events, inside the window
  0.05 ≤ τ < 1/3. The scale, scheme and dispersive-model variations of the calculation give a
  theory band that keeps the sign of each variation.

The paper performs no fit of the strong coupling. The two parameters of the calculation are fixed
inputs, α_s(M_Z) = 0.1135 and α_0 = 0.43, the values at which the calculation alone describes the
ALEPH thrust distribution in the window. The predictions are compared with DELPHI. Section 3 tests
the classifier weights and the learning band against fresh generator runs for one to seventeen
parameters. Section 4 shows the three bands of the thrust distribution and of seven means.

## Quick start

```bash
conda env create -f environment.yml && conda activate bdistcrete
cd analysis
python phase2tools/make_numbers.py output    # every quoted number
python figs_r8.py                            # the ten data figures, Figs. 3 to 12
```

`make_numbers.py` reads the result files in `analysis/output/` and
`analysis/data_stagePURE17/meta.json`, and writes
`analysis/phase2tools/numbers.tex`, 575 macros identical to the paper's `numbers.tex`, and
`numbers.json` next to it. `figs_r8.py` writes `output/r8_<name>.pdf`, and with names it draws
only those figures, for example `python figs_r8.py bands obs`. Figs. 1 and 2 are drawn in TikZ in
the paper source. The figures are typeset with LaTeX, which `environment.yml` does not install. To
reweight your own events with the released networks, see `analysis/release/README.md`.

## What can be redone from where

1. **From this repository alone, on a laptop.** Every quoted number and every figure, from the
   result files in `analysis/output/`. The thrust calculation at the fixed point and its comparison
   with ALEPH and DELPHI (`thrust_chain.py`, `targets_nlo_point.py`, `calc_bins.py`,
   `calc_vs_delphi.py`, with `RIVET_REF=rivet_ref`). The collection of the term-count scan
   (`kscan_collect.py`). The self-test of the released networks and the reweighting of your own
   events (`analysis/release/`).
2. **With the network exports and event samples of draft r8** ([`docs/DATA.md`](docs/DATA.md)).
   The three bands, the learning band and its coverage, the inputs of Figs. 3 and 5, the strict
   tests, the cost benchmark and the packaging of the released networks. These ran on the Symmetry
   cluster at Perimeter, with the batch scripts in `analysis/symmetry/`.
3. **From the generators.** Every sample can be regenerated with the scripts and designs in `gen/`.
   Training needs the samples and a GPU, one NVIDIA H200 per job.

The toy of Fig. 4 needs no data, because `toy_final.py` generates its own events. Its training ran
on a GPU (`symmetry/toy_final.sbatch`).

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) gives the command and the inputs of every figure, table
and number.

## The pipeline of draft r8

All scripts are in `analysis/` and run from there.

| what | where |
|---|---|
| training, both networks, every model | `r2_ladder.py` (`CondPFN`, `GeoMixturePFN`, `ConcatPFN`, `ConcatGeoMixture`), `stage_mixture.py`, `symmetry/train_p2.sbatch`, `symmetry/train_mixgeo.sbatch` |
| closure on the standard and the strict test | `ladder_widths_big.py`, `eval_valid.py`, `symmetry/score_p2_array.sbatch`, `symmetry/strict_closure.sbatch` |
| the learning band and its coverage | `wifi_embed.py`, `wifi_fit.py`, `wifi_diag.py`, `wifi_coverage.py`, `symmetry/wifi_cov.sbatch` |
| the calculation at the fixed point | `thrust_chain.py`, `targets_nlo_point.py`, `calc_bins.py`, `calc_vs_delphi.py` |
| the maximum-entropy weights and the three bands | `directlib.py`, `three_bands.py`, `theory_variations.py`, `band_checks.py` |
| the toy, the scans and the cost | `toy_final.py`, `kscan_submit.sh`, `kscan_collect.py`, `widebox_submit.sh`, `widebox_collect.py`, `bench_cost_final.py` |
| every quoted number, every data figure | `phase2tools/make_numbers.py`, `figs_r8.py` |
| the released networks | `package_release.py`, `make_stage_table.py`, `precision_gap.py`, `closure_vs_distance.py`, `fit_accuracy_law.py`, run by `symmetry/package_r8.sbatch`, `table_r8.sbatch` and `accuracy_r8.sbatch` |

## Contents

| path | what it holds |
|---|---|
| `analysis/` | every script of the paper, run from this directory |
| `analysis/output/` | the result files the numbers and figures are made from. `figs_r8.py` writes the figures here |
| `analysis/phase2tools/` | `make_numbers.py` and its output, `numbers.tex` and `numbers.json` |
| `analysis/symmetry/` | the batch scripts of draft r8, as run on the Symmetry cluster |
| `analysis/release/` | the released networks (numpy only), the `gentune` reader, the generator scripts and designs, the closure tables |
| `gen/` | the scripts and designs that generated the samples of draft r8 |
| `analysis/lepcov/`, `analysis/rivet_ref/` | the LEP measurements (HEPData tables and the YODA files of Rivet) and their reader |
| `analysis/ares_recovered/`, `analysis/eerad3/` | our drivers and patches for ARES and EERAD3, and the tables they produced |
| `analysis/perlmutter/`, `analysis/farm/` | batch scripts of the earlier draft |
| `docs/REPRODUCE.md` | every figure, table and quoted number, with the command and the inputs that produce it |
| `docs/DATA.md` | the network exports and event samples, which are too large for git |
| `tools/get_data.sh` | downloads the data archives of the earlier draft |
| `CLAUDE.md` | orientation for Claude Code: the method, how it relates to DCTR, the map of the code |
| `environment.yml`, `environment-tf.yml` | the analysis environment, and the TensorFlow one that only the energyflow cross-check needs |

## Data

The network exports and event samples are too large for git, and those of draft r8 are not yet in
a release archive. [`docs/DATA.md`](docs/DATA.md) lists the files the paper needs and what reads
each one. They will be archived with the paper at Zenodo, and until then they are available on
request. The result files made from them are in `analysis/output/`, so every number and figure
can be redone without them.

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

## Release v2.0.0

The tag `v2.0.0` marks the code of draft r8. `analysis/release/` holds the released networks at
version 2.0.0 (`analysis/release/VERSION`): `E` (Stage D, Sherpa), `F` (Stage E, Herwig) and
`MIX17` (the seventeen-parameter mixture of Sec. 4, its first set of members), with the basis
weights of its learning band in `MIX17_wifi.npz`. `C_1M` is a three-parameter network of the
earlier draft. They need only numpy, and `python -m gentune.selftest models`, run in
`analysis/release/`, checks them.

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
earlier standard. Draft r8 has no such fit, and every one of its results comes from one training
standard. The code of the earlier draft stays in the repository: `make_sec6_numbers.py`, the
profile and fit scripts (`profile_*.py`, `bands_exact.py`, `surface_rows.py`, `pseudo_rows*.py`),
the mixture trained on 208 runs (tags `MIXGEO`, `MIXGEOpm`, `MIXGEO8`, `MIXGEO16`, `MIXGEO32`,
`run_mixgeo_local.sh`), its figures (`figs_pro.py`,
`fig_three_bands*.py`, `fig_anchored_fit.py`) and the batch scripts in `analysis/perlmutter/` and
`analysis/farm/`. Its result files are in `analysis/output/`, and its data are the release
`data-v1` (`docs/DATA.md`). Its seed study is superseded by the three sets of members of every
model in Table 1. The instructions for it are in the git history, for example
`git show 1d66811:docs/REPRODUCE.md`.

## License

The code is released under the MIT License (`LICENSE`). ARES and EERAD3 are not included and keep
their own licenses.

## Citation

See `analysis/release/CITATION.bib`.
