# BDistcrete

Code for *Combined Uncertainties from Event Generators and Precision QCD* (B. Assi and J. Thaler),
earlier circulated as *Beyond Discrete Variations: Continuous Generator Uncertainties Anchored to
Precision QCD*. The LaTeX source of the paper is kept in a separate repository synced from Overleaf
(benleo12/Reweighting-generator-parameters, private until publication).

An event-generator prediction has three kinds of uncertainty, and this code puts all three on one
stored event sample. A classifier conditioned on the generator parameters reweights the sample to
any point of a parameter box, so the generator becomes a continuous, differentiable function of its
hadronization and shower parameters, including the choice between Sherpa and Herwig as the fraction
of a multiplicative mixture (the generator uncertainty). A fit of the classifier's own ensemble of
trainings on held-out events, following Benevedes and Thaler (arXiv:2506.00113), gives the central
weight and its covariance (the learning uncertainty). A maximum-entropy reweighting then imposes
the windowed thrust moments of an NNLL+NLO calculation on the same events and propagates its scale
and scheme variations (the theory uncertainty). The demonstration of the paper shows the three
bands of the thrust distribution and of other observables, separately and combined, against LEP
data. The sections below marked "earlier draft" document the strong-coupling fit of the earlier
version, whose code remains in the repository.

## The revision of October 2026 in short

| what | where |
|---|---|
| the multiplicative mixture head and the ensemble-size switch | `analysis/r2_ladder.py` (`GeoMixturePFN`, `LADDER_ENS`, `LADDER_EMB_STORE`), `analysis/stage_mixture.py` (geometric head is the default) |
| the pure-run sample of both generators | `analysis/make_pure_dataset.py`, `analysis/data_stagePURE17/meta.json` |
| the learning uncertainty: basis, fit, covariance, diagnostics, ensemble-size scan | `analysis/wifi_embed.py`, `wifi_fit.py`, `wifi_diag.py`, `wifi_scan_table.py` |
| the NNLL+NLO calculation at a fixed point | `THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python thrust_chain.py`, `analysis/targets_nlo_point.py` |
| the three bands and their figures | `analysis/three_bands.py`, `fig_three_bands.py`, `fig_three_bands_obs.py`, results in `analysis/output/three_bands_*.json` |
| the training jobs | `analysis/perlmutter/train_mixgeo_ens.sbatch` (any ensemble size), `analysis/run_mixgeo_local.sh` (one laptop GPU) |

`docs/REPRODUCE.md` opens with the commands of this revision.

## Contents

| path | what it holds |
|---|---|
| `analysis/` | every script that produces a figure, table or number of the paper, run from this directory |
| `analysis/output/` | the small result files the figures and numbers are made from, and the paper's figures |
| `analysis/release/` | the trained models (`C_1M`, `E`, `F`, `MIX17`) with a numpy-only reader, `gentune`, the generator cards and run scripts, examples and closure tables |
| `analysis/perlmutter/` | the batch scripts as run at NERSC |
| `analysis/farm/` | the GPU farm of the term-count scan and the wide coupling range, with the script versions it ran |
| `analysis/ares_recovered/`, `analysis/eerad3/` | our drivers and patches for ARES and EERAD3, and the tables they produced |
| `analysis/rivet_ref/` | the LEP thrust measurements (HEPData values in the YODA format of Rivet) |
| `docs/REPRODUCE.md` | every figure, table and quoted number, with the command and the inputs that produce it |
| `docs/DATA.md` | the event samples and network exports, which are too large for git, as downloadable archives |
| `tools/get_data.sh` | downloads the archives of `docs/DATA.md`, checks their SHA256 and unpacks them into `analysis/` |
| `CLAUDE.md` | orientation for Claude Code: the method, how it relates to DCTR, the map of the code |
| `environment.yml`, `environment-tf.yml` | the analysis environment, and the TensorFlow one that only `crosscheck_ef.py` needs |

## Quick start

```bash
conda env create -f environment.yml && conda activate bdistcrete
cd analysis
python make_sec6_numbers.py                                   # every number of Sec. 6, from output/
python fig_anchored_fit.py output/profile_MIX17ext_central.json   # Fig. 10
python fig_thrust_vs_aleph.py                                  # Fig. 11 and the fit of App. D
python figs_pro.py toy anymap head widebox                     # Figs. 4, 6, 7 and 8
python fig_cost.py                                             # Fig. 9, from output/bench/
python method_figs.py                                          # Fig. 2
```

The figure scripts draw their labels with LaTeX, which `environment.yml` does not install, so a
TeX distribution is needed for the figures. To reweight your own events with the released models,
see `analysis/release/README.md`.

## What can be reproduced from where

1. **From this repository alone, on a laptop:** every number of Sec. 6 (`make_sec6_numbers.py`), the
   thrust calculation and its fit to ALEPH (App. D), and every figure made by code except Fig. 5,
   which needs a Stage C event file (Figs. 1 and 3 are drawn in the LaTeX source).
2. **With the network exports and reference samples (`tools/get_data.sh`,
   [`docs/DATA.md`](docs/DATA.md)):** the closure tests of Sec. 5, the anchoring and the fits of
   Sec. 6. The fits ran on NERSC CPU nodes (`analysis/perlmutter/`).
3. **From the generators:** the run cards and seeds of every sample are included, and
   `docs/DATA.md` lists the versions. Training ran on GPUs.

## Data

The exports and event samples are too large for git. [`docs/DATA.md`](docs/DATA.md) lists them as
archives of under 2 GB each, published with the GitHub release `data-v1`, and says what each one
holds. `tools/get_data.sh` downloads archives by group or by name, checks their SHA256 and unpacks
them into `analysis/` with the paths the scripts use. From the repository root:

```bash
tools/get_data.sh sec6              # the archives of one group in the table below
tools/get_data.sh B1_exports_ABC    # or single archives, by name
tools/get_data.sh core              # A0 and B1 to B10 (9.6 GB): every row below except App. A, Sec. 5.8 and retraining
tools/get_data.sh --list            # every archive with its size, and the groups
```

The script also links the reference names the scripts open to the slim reference files
(`docs/DATA.md`, "Reference files"). Run `tools/get_data.sh --unlink` before retraining. The
archives each cross-check needs, and the group that downloads them:

| cross-check | archives | group |
|---|---|---|
| every number of Sec. 6 from the stored results, Figs. 2, 4 and 6 to 10, the fit of App. D | none | |
| the per-job files of the toy merges of Sec. 5.1 and of the Stage E term-count merge, the Sec. 6 grid profiles, the ARES tables | none, they ship with the repository | |
| the fixed-order cumulants of App. D, rebuilt from the raw EERAD3 logs | `A0_theory_logs` | `appD` |
| Table 1, Sec. 5.2 rates and energy fraction, Fig. 5, App. B temperatures and single networks | `B1_exports_ABC`, `B2_held_ABC` | `table1` |
| Fig. 6 and the multiplicity tails (Sec. 5.3) | `B1_exports_ABC`, `B2_held_ABC`, `B3_held_C_valid` | `sec5` |
| offsets at 80k and 800k events and the accuracy law (Sec. 5.3) | `B1_exports_ABC`, `B2_held_ABC`, `B4_held_C_floor` | `sec5` |
| Table 2, the seven-parameter mixture | `B5_mix7` | `table2` |
| the Sec. 6 fit and App. E | `B6_sec6_MIX17` | `sec6` |
| Table 3 | `B6_sec6_MIX17`, `B8_E`, `B9_F`, `B10_held_8param` | `table3` |
| the Sec. 6 three-parameter family | `B7_C1M` | `family` |
| App. A, the exact-axis widths and the closure of the 54-run network | `B1_exports_ABC`, `B7_C1M`, `B12_appA_v2` | `appA` |
| Sec. 5.8 cost | `B6_sec6_MIX17`, `B11_bench` | `cost` |
| retraining a network | its training archives (`C1_train_A` to `C9b_train_MIX17`) and the archive with its held-out runs, listed under "Retraining" in `docs/DATA.md` | `train-A`, `train-B`, `train-C`, `train-C1M`, `widebox`, `train-DM2`, `train-E`, `train-F`, `train-MIX17` |

`docs/REPRODUCE.md` gives the archives per section as well.

## Notes on exact reproduction

- The closure tests of Secs. 5.2 to 5.6 used an earlier thrust-axis finder
  (`compute_efps_old_axis.py`), and so did the temperature fits of App. B (`docs/DATA.md`, "Notes
  on exact reproduction"). Everything else uses the exact axis (`compute_efps.py`). The network
  inputs are unchanged between the two.
- The temperature grid of `r2_ladder.py` was refined after the ladder networks were trained. The
  ladder uses temperature one, as stated in the paper, so a retrained Stage C needs
  `python apply_T.py C 1.0`.
- Four Stage B runs lost their seeds and cannot be regenerated bit for bit.
- External codes are not redistributed: Sherpa 3.0.4 and a Sherpa 3.1 development build, Herwig
  7.2.3 and 7.3.0, Rivet 3.1.10, ARES (GPL-3, upstream commit e75c88c with the edits in
  `ares_recovered/ares_edits.txt`) and EERAD3 1.0 (with `eerad3/eerad3_blk.patch`).
- Batch scripts use `$SCRATCH/BDistcrete/analysis` and the placeholder `<your-account>`.

## License

The code is released under the MIT License (`LICENSE`). ARES and EERAD3 are not included and keep their own licenses.

## Citation

See `analysis/release/CITATION.bib`.
