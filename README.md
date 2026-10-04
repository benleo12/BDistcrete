# BDistcrete

Code for *Beyond Discrete Variations: Continuous Generator Uncertainties Anchored to Precision QCD*
(B. Assi and J. Thaler).

A classifier conditioned on the generator parameters reweights one stored event sample to any point
of a parameter box, so the generator becomes a continuous, differentiable function of its
hadronization and shower parameters, including the choice between Sherpa and Herwig through a
mixing fraction. A maximum-entropy reweighting then anchors the sample to windowed thrust moments of
an NNLL+NNLO calculation, and a fit to LEP data with the generator parameters as nuisance parameters
determines the strong coupling.

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
| `docs/DATA.md` | the event samples and network exports, which are too large for git |
| `CLAUDE.md` | orientation for Claude Code: the method, how it relates to DCTR, the map of the code |
| `environment.yml`, `environment-tf.yml` | the analysis environment, and the TensorFlow one that only `crosscheck_ef.py` needs |

## Quick start

```bash
conda env create -f environment.yml && conda activate bdistcrete
cd analysis
python make_sec6_numbers.py                                   # every number of Sec. 6, from output/
python fig_anchored_fit.py output/profile_MIX17ext_central.json   # Fig. 9
python fig_thrust_vs_aleph.py                                  # Fig. 10 and the fit of App. D
python figs_pro.py toy anymap head widebox                     # Figs. 4, 6, 7 and 8
python method_figs.py                                          # Fig. 2
```

To reweight your own events with the released models, see `analysis/release/README.md`.

## What can be reproduced from where

1. **From this repository alone, on a laptop:** every number of Sec. 6 (`make_sec6_numbers.py`), the
   thrust calculation and its fit to ALEPH (App. D), and every figure made by code except Fig. 5,
   which needs a Stage C event file (Figs. 1 and 3 are drawn in the LaTeX source).
2. **With the network exports and reference samples (`docs/DATA.md`):** the closure tests of Sec. 5,
   the anchoring and the fits of Sec. 6. The fits ran on NERSC CPU nodes (`analysis/perlmutter/`).
3. **From the generators:** the run cards and seeds of every sample are included (`docs/DATA.md`
   lists the versions); training ran on GPUs.

## Notes on exact reproduction

- The closure tests of Secs. 5.2 to 5.6 used an earlier thrust-axis finder
  (`compute_efps_old_axis.py`); everything else uses the exact axis (`compute_efps.py`). The
  network inputs are unchanged between the two.
- The temperature grid of `r2_ladder.py` was refined after the ladder networks were trained; the
  ladder uses temperature one, as stated in the paper.
- Four Stage B runs lost their seeds and cannot be regenerated bit for bit.
- External codes are not redistributed: Sherpa 3.0.4 and a Sherpa 3.1 development build, Herwig
  7.2.3 and 7.3.0, Rivet 3.1.10, ARES (GPL-3, with the changes in `ares_recovered/ares_edits.txt`)
  and EERAD3 1.0 (with `eerad3/eerad3_blk.patch`).
- Batch scripts use `$SCRATCH/BDistcrete/analysis` and the placeholder `<your-account>`.

## Citation

See `analysis/release/CITATION.bib`.
