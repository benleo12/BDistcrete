# Reproducing the paper

This file maps every figure, table and quoted number of the paper to the command and the inputs
that make it. The paper's numbers come from one complete run of the scripts of `analysis/` on the
Symmetry cluster at Perimeter, in November 2026, and the commands below are those of that run. It
scored every trained network again and propagated every uncertainty, with three choices that run
through everything:

- **The two generators mix linearly.** A prediction at the fraction xi of Herwig is
  (1 - xi) <O>_S + xi <O>_H, with xi flat on [0, 1]. The trained mixture is evaluated only at
  xi = 0 (Sherpa) and xi = 1 (Herwig), each generator's weights are normalized over the reference
  events, and a calculation is imposed on each generator alone before the two predictions are
  mixed. The scripts take this as `MIX=linear`, and their outputs end in `_lin`.
- **The photons radiated by the beams are left out of every observable**, by the rule of
  `beam_photons.py` (`analysis/release/REFERENCE.md` describes it). The network inputs keep them.
  Every data folder the run reads is a photon-free copy (`_nb`) of the published one, and every
  reference bundle it reads is listed with that rule in `output/bundle_rule.json`.
- **Every distribution is the fraction of all events in each bin over its width**, and every mean
  is over all events. DELPHI's thrust distribution is the table from charged and neutral particles,
  `d11-x01-y02`, used as published.

The calibration constant of every network was fitted again on these observables by the
training's own procedure, and every script checks that an export carries its refit (`T_DIR`).
**(data)** marks a step that needs the network exports or event samples of `docs/DATA.md`, and
**(GPU)** a step that ran on a GPU.

The result files of that run that the paper reads are in `analysis/output/`: its json files, its
small npz files, `numbers/numbers_lin.tex` and the figures in `figs/`. Its large npz files, the
draws of the comparison with 2000 variations and the bases of the learning bands, are not in git
(`docs/DATA.md`).

## Before you start

- Create the environment with `conda env create -f ../environment.yml`. The figures are typeset
  with LaTeX (`usetex`, with `amsmath` and `amssymb`), which the environment does not install; a
  full TeX Live has everything they need. The comparison with the energyflow PFN needs a second
  environment, `conda env create -f ../environment-tf.yml`.
- The scripts that read the LEP measurements (`calc_bins.py`, `calc_vs_delphi.py` and everything
  that imports `directlib.py`) look for them in a local Rivet installation. Set
  `export RIVET_REF=rivet_ref` to use the copies that ship in `analysis/rivet_ref/`.
- The run's batch scripts are in `symmetry/rerun/` (the scoring side), `symmetry/` (the
  propagation side) and `symmetry/release/` (the deposit). Each runs only in the run's folder,
  `RUN=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code`, which it names in its first lines and in
  its `#SBATCH -o` line, with the Python of `/mnt/beegfs/bassi/reweighting/env/torch`; elsewhere,
  change those lines. Their `#SBATCH` lines name the partitions of Symmetry. `symmetry/rerun/sync_run.sh`
  copied the scripts into RUN: every file of `symmetry/rerun/files_score.txt` and
  `symmetry/files_prop.txt`, with their md5 compared on both sides.
- In RUN, `data_stagePURE17_v3`, `data_stagePURE17_v3_h400` and the other data folders, and
  `../gen/data_abc`, `data_abc_blind` and `data_widebox`, are links to their photon-free copies.
  `output/models` holds the trained networks of the published run and the photon-free reference
  bundles, which the rule's step wrote and listed in `output/bundle_rule.json` with the md5 of the
  `beam_photons.py` that made them. With `BEAM_PHOTONS=remove`, which every job sets, a script stops
  on a bundle that is not listed.
- `output/published/` in RUN holds read-only links, which `symmetry/rerun/setup_lmx.py` makes, to
  files of the published run in `/mnt/beegfs/bassi/reweighting/maxent_code/output`: its reference
  bundles, which keep the particles of every event, its training records, the basis functions of
  its learning bands and its truncation count. The run reads them and writes nothing there.
- The comparisons with a measurement multiply each generator's prediction, after the calculation
  is imposed and before the mixing, by its factor to the hadron level of that measurement:
  `CONV_FACTORS=lmx_runs/conventions/factors.json`, the file `analysis/conventions/factors.json`,
  which `analysis/conventions/README.md` says how to make. The release ships a copy,
  `analysis/release/examples/data/convention_factors.json`.
- A run writes its result files and exports under the names given below, and the first submission
  moves every earlier file of those names aside into `output/published_copies` (`LIST.tsv` lists
  them). Copy the shipped files first if you want to compare with them.

## The photon-free data

The event shapes of every run, trained on or held out, and of every reference bundle were made
again without the photons radiated by the beams, once, before the run. The particle files keep
them, as the networks need, and the rule of `beam_photons.py` finds them among the stored features
of each event. 1 - T, B_T and rho_H then become the value with the photons minus the change their
removal makes on momenta rebuilt from the features, and the total multiplicity loses the flagged
photons. The copies are the `_nb` folders, with the same files under the same names, and the
refreshed reference bundles of `output/models` and `output/` (the four bases of the learning bands,
`wifi_<tag>_basis.npz`, are in `output/`), each recorded in `output/bundle_rule.json` with its
sha256 and the md5 of the `beam_photons.py` that made it. The kaon fraction, the multiplicity of
the stored particles and the x_p spectrum are counted without the photons when a scoring reads the
particles (`score_obs.py`). The scripts of this step are in `analysis/photon_free_data/`, whose
`README.md` gives their order, inputs and outputs.

## Validation samples

The main closure results in Tables 1 and 2 and the rank scan in Figure 5 use
**200,000 comparison events per held-out parameter point**, the comparison half
of the extended 400,000-event runs. Their existing filenames and macro names
retain `strict` or `Strict`, and the rank scan reads `scan_strict`.
The split is redrawn after pooling the original run and four independent-seed
extensions, so it includes some original selection/calibration events. The
selected checkpoint and calibration factor are held fixed for this comparison.

Other checks answer different questions and retain their original samples:

- Checkpoint selection and calibration use the selection half of the original
  80,000-event runs. Their historical `standard` results use the other 40,000 events.
- The additional Stage C points in Figure 3, the wide-coupling comparison, the
  DCTR singular-value truncation check and the WiFi fitted-combination comparison
  use 40,000 comparison events per point. They are distinct diagnostics, not a
  second set of the main closure results.
- The learning-band mean comparisons in Figure 4 use all 400,000 events at each
  held-out point, including the 40,000 events used for selection and calibration.
  They do not use the 200,000-event histogram-comparison half.

The numerical results, filenames, JSON keys and macro names are unchanged. The
November 14 cleanup removes the duplicated 40,000-event main tables and rank-scan
panel, and drops the old illustrative closure figure from the paper. Its cache
and command remain available below.

## Figures

Figure numbers below follow the November 14 manuscript. The existing propagation
plots (Figs. 8, 9 and 11 to 16) come from `lmx_figs_prop`, the scoring plots
(Figs. 3 to 5 and 10) from `lmx_figs_score`, and Fig. 6 from `lmx_figs_cost`.
Figure 7 is drawn directly from those existing outputs with the command below.
The paper's `figs/r8_<name>.pdf` are copies of `output/figs/r8_<name>_lin.pdf`,
or of `r8_<name>.pdf` where the figure has no `_lin` version. Figure 7 uses
`before_after_distributions.pdf`.

Data plots use 10-point axis labels, ticks, panel titles and legend text on
5.95-inch PDF canvases, included at the full paper text width. Legends use short
keys; definitions and explanations are given in the captions.

| figure | what it shows | drawn from (`output/`) | inputs made by | drawn by |
|---|---|---|---|---|
| 1 | overview of the method | | TikZ in `main.tex` | |
| 2 | the factorized network | | TikZ in `main.tex` | |
| 3 | Stage C across its box, with 40,000 comparison events per point | `valid_map_C_P2augfullfac0_report.json`, `xp_null.json` | `symmetry/rerun/stagec_lmx.sbatch` (`eval_valid.py`, `xp_null.py`) **(data)** | `figs_r8.py map`, which reads the `mean` of `xp_null.json` for the dotted tick |
| 4 | mean residuals with statistical and statistical-plus-learning denominators, separately for Sherpa and Herwig, using 400,000 events per point | `wifi_coverage_MIXGEOV3S9.json`, `..._MIXGEOV3S9s1.json`, `..._MIXGEOV3S9s2.json` | `symmetry/rerun/wifi_cov_lmx.sbatch` with `TAG` each of the three **(data)** | `figs_r8.py coverage` |
| 5 | closure against the number of terms per member, using 200,000 comparison events per point | `rank_scan_final.json`, `scan_strict` only | the K-scan scorings of the scoring side, then `symmetry/rerun/collect_lmx.sbatch` (`kscan_collect.py`) **(data)** | `figs_r8.py kscan`, one panel for Stages A to D |
| 6 | the cost at one parameter point | `bench_final_cpu_4_lin.json`, `bench_final_cuda_12_lin.json` | `symmetry/bench_final_cpu.sbatch` on a whole CPU node and `symmetry/bench_final_gpu.sbatch` on one H200 (`bench_cost_final.py`, `BENCH_MIX=linear`) **(data, GPU)** | `figs_r8.py cost` with `COST_SUFFIX=_lin` |
| 7 | thrust and charged multiplicity before and after imposing the calculation, with ratio panels and the same generator uncertainty prescription | `envelope_test_MIXGEOV3S9_full_lin.npz`, `envelope_test_MIXGEOV3S9_tune_lin.npz`, `three_bands_MIXGEOV3S9_lin.npz` | existing propagation outputs, no new runs | `plot_before_after_distributions.py`, command below |
| 8 | conditional thrust shape, before and after imposing the moments | `three_bands_MIXGEOV3S9_lin.npz`, `theory_variations_MIX_lin.json`, `calc_bins_nlo_point.json` | as Fig. 15, and `calc_bins.py` (`lmx_calc_bins`) | `figs_r8.py window` (`r8_window_shape_lin.pdf`) |
| 9 | event fraction in the same interval, compared with the calculation and DELPHI | as Fig. 8 | as Fig. 8 | `figs_r8.py window` (`r8_window_fraction_lin.pdf`) |
| 10 | the toy model (App. B) | `toy_final_K48_s0.npz`, `toy_final_K48_s0.json` | `symmetry/rerun/toy_final_lmx.sbatch` (`toy_final.py`, which generates its own events), nine trainings on one H200 **(GPU)** | `figs_r8.py toy` |
| 11 | the calculation against ALEPH and DELPHI (App. D) | `calc_bins_nlo_point.json` | `calc_bins.py` (`lmx_calc_bins`) | `figs_r8.py aleph` |
| 12 | thrust and charged multiplicity over 2000 variations, compared with linear propagation (App. F) | `envelope_test_MIXGEOV3S9_tune_lin.npz`, `envelope_test_MIXGEOV3S9_full_lin.npz` | `symmetry/envelope_test.sbatch`, as in "The comparison with 2000 variations" below **(data)** | `plot_envelope.py output/envelope_test_MIXGEOV3S9` with `SUFFIX=_lin` |
| 13 | bin correlations compared with linear propagation (App. F) | as Fig. 12 | as Fig. 12 | as Fig. 12 |
| 14 | thrust propagation with the calculation imposed (App. F) | as Fig. 12 | as Fig. 12 | as Fig. 12 |
| 15 | full-range DELPHI comparison and uncertainty components (App. F) | `three_bands_MIXGEOV3S9_lin.npz`, `three_bands_MIXGEOV3S9_lin.json`, `theory_variations_MIX_lin.json` | `symmetry/three_bands.sbatch` with `TAG=MIXGEOV3S9 BOX=full`, and `symmetry/theory_variations.sbatch` **(data)** | `figs_r8.py bands` |
| 16 | uncertainty components of the seven integrated quantities (App. F.4) | `three_bands_MIXGEOV3S9_lin.json` | as Fig. 15 | `figs_r8.py obs` |

The November 13 and 14 presentation changes use the existing result files. The helpers in
`polish_paper_figures.py` separate the learning checks by generator, split the shape
and event-fraction comparisons, and draw the full uncertainty budget with every
positive component visible. No network, event sample or fitted moment is changed.
To reproduce these plots from `analysis/`:

```bash
R8_OUT=output/figs \
BANDS_NPZ=output/three_bands_MIXGEOV3S9_lin.npz \
BANDS_JSON=output/three_bands_MIXGEOV3S9_lin.json \
VARS_JSON=output/theory_variations_MIX_lin.json \
CALC_BINS=output/calc_bins_nlo_point.json \
python figs_r8.py coverage kscan window obs
```

The `window` command also retains the old combined `r8_window_lin.pdf` for
compatibility. The two new figures state their distinct particle definitions.
The DELPHI event-fraction error uses the statistical and systematic components
cached in `calc_bins_nlo_point.json`: statistics in quadrature, systematics fully
correlated. This avoids dependence on a local YAML reader or a fallback to total
bin errors. The plot JSON records the error prescription and plotted values.

For Figure 7, run from `analysis/`:

```bash
python plot_before_after_distributions.py \
  --variations output/envelope_test_MIXGEOV3S9_full_lin.npz \
  --nominal output/envelope_test_MIXGEOV3S9_tune_lin.npz \
  --published-check output/three_bands_MIXGEOV3S9_lin.npz \
  --output output/figs/before_after_distributions.pdf \
  --record output/figs/before_after_distributions.json
```

Both before/after groups use the generator standard deviation from the same 2000 `Y_theta`
variations in the full-range file. The `tune` archive supplies only the
already-evaluated nominal predictions at the default settings, which the
script checks against `three_bands`; its narrower parameter ranges are not
used for these bands. The script also checks the thrust generator bands
against the published arrays and writes its checks beside the figure.
Each ratio panel divides the displayed nominal curve and its plus/minus-SD band
by the fixed nominal prediction before imposing the calculation. These are the
same uncertainties in relative units, not a separately propagated random ratio.
No event generation, training or new moment fit is needed.

The archived `fig_closure_mix.npz` was made by
`symmetry/rerun/closure_mix_lmx.sbatch` (`lmx_closmix`, **data**) at the point of
median 40,000-event closure width for each generator. `figs_r8.py closure_mix`
still draws it as `r8_closure_mix.pdf`, but it is neither a paper figure nor part
of the default `figs_r8.py` command list.

## Tables

| table | what it shows | from (`output/` unless stated) | made by |
|---|---|---|---|
| 1 | closure widths of every model, both networks, three ensembles, using 200,000 comparison events per point | Stages A to C: `p2augfullstrict_<S>_<net>_s<k>.json`, with `p2augstrict_<S>_<net>_s<k>.json` where the training uses that prefix. Stages D and E: `p2v3fullstrict_{E,F}_<net>_s<k>.json`. Mixture: `p2v3fullstrict_MIX_{fac,cat}_s<k>.json`. Here `fac` is factorized and `cat` is conventional DCTR | the scoring side (below) **(data, GPU)**, then `make_numbers.py` (`\Acc{Aug,Dense}*Strict*`) |
| 2 | closure widths per observable, Stages D and E and the mixture, using 200,000 comparison events per point | as Table 1 | `make_numbers.py` (`\ObsDense*Strict`) |
| 3 | parameters of Stages A to C | the designs `gen/designs/stage{A,B,C}_design_aug.csv` (App. A.1) | typed in `main.tex` |
| 4 | parameters of Stages D and E | the designs `analysis/release/generators/stage{E,F}_design_v3all.csv` and the boxes in `analysis/release/STAGES.md` | typed in `main.tex` |
| 5 | loss and closure width of the ensemble average and of the fitted combination, using 40,000 comparison events per point | `wifi_MIXGEOV3S9{,s1,s2}_diag.json`, `wifi_MIXGEOV3S9{,s1,s2}_fit.json` | `symmetry/rerun/wifi_cov_lmx.sbatch` **(data)**, then `make_numbers.py` (`\Wifi{Loss,Width}*`) |

The macro generator still reads both historical sample sizes and writes both
`Std` and `Strict` families. Only the `Strict` closure results appear in Tables 1
and 2. Its stored 40,000-event Gaussian null reference is retained as a diagnostic,
without the former example in Table 1's caption.

## Quoted numbers

Every quoted number of the paper is a macro of `numbers.tex`, which `phase2tools/make_numbers.py`
writes from the result files of the run:

```bash
SUFFIX=_lin INDIR=$RUN/output OUTDIR=$RUN/output/numbers python -B -u phase2tools/make_numbers.py
```

It writes `output/numbers/numbers_lin.tex` and `numbers_lin.json`, which records every input with
its md5. It refuses, writing nothing: an input made at a reduced size (fewer than 2000 points or
draws, fewer than 1000 further draws on the full envelope box, a test file), a missing input, a
benchmark file whose device or threads are not those of its name, an input of the scoring side
that `output/scoring_check.json` does not list with its md5, an input that resolves outside RUN or
into `output/published_copies`, and any macro of its list that it did not write. Only the four
`\Cost*` macros of the GPU benchmark may be missing, while no GPU benchmark exists. The job
`lmx_figs_cost` writes the file again with all of them, and that `numbers_lin.tex` is the paper's
`numbers.tex`. The macros come in families:

| macros | where in the paper | read from (`output/`) | made by |
|---|---|---|---|
| `\Acc{Aug,Dense}*Strict*`, `\AccFinal*Strict*`, `\ObsDense*Strict`, `\Mix{Sherpa,Herwig}PointsStrict` | Tables 1 and 2, Secs. 3.3 and 3.6 | the files of Table 1 | the scoring side |
| `\Gauss*` | retained null-reference diagnostic, no longer in Table 1's caption | `gauss_ref_lin.json`, the closure width of two 40,000-event Gaussian samples with the paper's estimator | `gauss_ref.py` (`lmx_gauss`) |
| `\Kappa*` | App. A.3 | the refitted calibration constant of every training of Table 1 | the scoring side |
| `\LcRunsThreeStrictAll`, `\LcMembersStrictAll` | App. A.3 | `p2v3fullstrict_MIX_fac_s<k>.json`, `widths_v2_MIXGEOV3M8S9h400.json`, and for the sentence on more runs, events or steps the other `widths_v2_<tag>{,h400}{,r415}.json` | the scoring side (`strict_closure_lmx.sbatch`, `rescore_v4_lmx.sbatch`) |
| `\Calib*` | archived null diagnostic, no longer quoted | `p2augfull_{A,C}_fac_s0.json`, `p2aug_B_fac_s0.json` | the scoring side |
| `\Toy*` | App. B.1 | `toy_final_K<K>_s<k>.json` | `symmetry/rerun/toy_final_lmx.sbatch` |
| `\Map*`, `\XpNullMean` | Sec. 3.4, Fig. 3 | `valid_map_C_P2augfullfac0_report.json`, `xp_null.json` | `symmetry/rerun/stagec_lmx.sbatch` **(data)** |
| `\Single*` | archived member comparison, no longer quoted | `single_members_C_P2augfullfac0.json` | `symmetry/rerun/stagec_lmx.sbatch` **(data)** |
| `\Wide*` | Sec. 3.4 | `p2wb18{single,pool,local9100,...,local9105}_W_fac_s0.json` | the scoring side **(data)**. `figs_r8.py widebox` draws these results, which the paper quotes without a figure |
| `\Cov*`, `\CovEight*` | Secs. 3.5 and 6, Fig. 4 | `wifi_coverage_MIXGEOV3S9{,s1,s2}.json`, `wifi_coverage_MIXGEOV3M8S9.json` | `symmetry/rerun/wifi_cov_lmx.sbatch` **(data)** |
| `\SvdRank*` | Sec. 3.6 | `p2trunc_{A,B,C,E,MIX}_cat_s0.json` | the scoring side, `specs/svd.txt` **(data, GPU)** |
| `\TruncMaxPercent` | App. A.2 | `truncation_fraction.json`, counted on the shapes files of the published run, which count every particle as the network input does | `symmetry/rerun/trunc_lmx.sbatch` **(data)** |
| `\PfnParams*`, `\PfnAuc*` | archived implementation comparison, no longer quoted | `crosscheck_seeds_summary.json`: network parameter counts and AUCs | `symmetry/pfn_lmx.sbatch` (`lmx_pfn`) **(data)** |
| `\Cost*` | Secs. 1, 3.6, 4.2 and 6, Fig. 6 | `bench_final_cpu_4_lin.json`, `bench_final_cuda_12_lin.json` | as Fig. 6. `\Cost{Fac,Late,Cat}{Cpu,Gpu}` time both generators' log-ratio outputs and a backward pass with supplied output derivatives. `\CostW*` time those outputs alone, before exponentiation and normalization, and `\CostVar*` scale that cost to the variations of Sec. 4.3. `\CostRestCpu` includes the remaining fit calculation, with maximum-entropy weights and the fit statistic and their derivative evaluation |
| `\CalcAlphas`, `\CalcAlphaZero` | Sec. 5.2, App. D.2 | `three_bands_MIXGEOV3S9_lin.json` | as Fig. 15, from the fixed point of the calculation below |
| `\CalcShift` | App. D | `calc_bins_nlo_point.json`, `power_correction`: the shift a_tau P of the distribution at the fixed point and central scales | `calc_bins.py` (`lmx_calc_bins`) |
| `\Tb*` | Secs. 1, 2.3, 5.2 to 5.5 and 6, Apps. D.2, E.2 and F.3 | `three_bands_MIXGEOV3S9{,s1,s2}_lin.{json,npz}`, `theory_variations_MIX_lin.json`, `calc_vs_delphi_MIXGEOV3S9_lin.json`, `band_checks_MIXGEOV3S9_lin.json` | the three bands of Sec. 5 below **(data)** |
| `\Env*` | Secs. 4.3 and 6, Apps. E and F | `envelope_test_MIXGEOV3S9_<box>_lin.{json,npz}` for the boxes `full`, `tune`, `tune_sherpa`, `tune_herwig`, `full_sherpa` and `full_herwig`; `\EnvNdraw` and `\EnvNref` are the draws of every box and the further draws of the full box | "The comparison with 2000 variations" below **(data)** |

The explanation of Table 5 by the reference events of the other generator rests on
`output/wifi_gensplit_MIXGEOV3S9.json`, which `symmetry/rerun/wifi_cov_lmx.sbatch`
writes with `TAG=MIXGEOV3S9` **(data)**. The archived parameter-count comparison
also uses `output/param_count.json`, written by `symmetry/rerun/trunc_lmx.sbatch`.
Generated macro families for these archived diagnostics remain in the result files;
they do not add results to the current manuscript.

## Trained models

The trainings ran before this run, with the commands of this section; the run did not train any
network again. It scored the trained networks again on the photon-free observables (next
section), and its results replace the scorings these commands wrote. Every model of Table 1 is
trained three times, the three ensembles, with seeds 0, 1 and 2 (`LADDER_SEED_OFFSET` 0, 1000 and
2000). Each training writes its historical 40,000-event comparison result. The separate scoring
called `strict` evaluates the saved model with 200,000 comparison events per point, the results
used in Tables 1 and 2. These commands run from `analysis/`
with `export MAXENT_DIR=$PWD`, submitted with `sbatch --export=ALL`; the batch scripts set the
training settings of the paper (four bootstrap members, a fifth of each run set aside, SiLU), and
`K` and `STEPS` must be passed as below. `train_p2.sbatch` requires both, and
`train_mixgeo.sbatch` defaults to 36000 steps. `r2_ladder.py` alone defaults to other settings.

**Stages A to C.** The densified designs are `gen/designs/stage{A,B,C}_design_aug.csv` and the runs
are in `gen/data_abc`. The strict test reads `data_abc_h400`, made by `MAXENT_DIR=$PWD python make_abc_strict.py`
from `gen/data_abc` and the extensions in `gen/data_abc_ext/` **(data)**.

```bash
export MAXENT_DIR=$PWD K=48 STEPS=324000 DESIGN_TAG=augfull ABC_DESIGN=../gen/designs
export SPECS="A:fac:0 A:fac:1 A:fac:2 A:cat:0 A:cat:1 A:cat:2 B:fac:0 B:fac:1 B:fac:2 B:cat:0 B:cat:1 B:cat:2 C:fac:0 C:fac:1 C:fac:2 C:cat:0 C:cat:1 C:cat:2"
J=$(sbatch --parsable --array=0-17 --export=ALL symmetry/train_p2.sbatch)
sbatch --array=0-17 --dependency=aftercorr:$J --export=ALL symmetry/score_p2_array.sbatch
```

This writes the exports `output/models/<S>_P2augfull<net><k>_*`. Five trainings of Table 1 (B
factorized, seeds 0 to 2, and C factorized and C with the DCTR network, seed 2) are those of a run
with `DESIGN_TAG=aug`, which had early stopping switched on but trained the full 324000 steps;
the scoring side scores them under the names `p2aug_*`.

**Stages D and E, and the DCTR network of the mixture.** The designs are
`stageE_design_v3all.csv` and `stageF_design_v3all.csv` in `analysis/`, and the runs are those of
`data_stagePURE17_v3` **(data)**.

```bash
export MAXENT_DIR=$PWD K=48 STEPS=324000 DESIGN_TAG=v3full DM_DATA=data_stagePURE17_v3 \
       E_DATA=data_stagePURE17_v3 E_CSV=stageE_design_v3all.csv F_DATA=data_stagePURE17_v3 F_CSV=stageF_design_v3all.csv
export SPECS="E:fac:0 E:fac:1 E:fac:2 E:cat:0 E:cat:1 E:cat:2 F:fac:0 F:fac:1 F:fac:2 F:cat:0 F:cat:1 F:cat:2 MIX:cat:0 MIX:cat:1 MIX:cat:2"
J=$(sbatch --parsable --array=0-14 --export=ALL symmetry/train_p2.sbatch)
sbatch --array=0-14 --dependency=aftercorr:$J --export=ALL symmetry/score_p2_array.sbatch
```

The strict test reads `data_stagePURE17_v3_h400`. This writes the exports
`output/models/{E,F}_P2v3full<net><k>_*` and `output/models/MIXP2v3fullcat<k>_*`.

**The factorized mixture.** `symmetry/train_mixgeo.sbatch` trains it, builds its reference bundle
(`rebuild_ref_lowmem.py`), scores it on the standard test (`ladder_widths_big.py`) and embeds the
basis of its learning band (`wifi_embed.py`). With `CHAIN=1` a training longer than one queue slot
continues in a second job.

```bash
export MAXENT_DIR=$PWD
for k in 0 1 2; do
  T=MIXGEOV3S9; [ $k = 0 ] || T=MIXGEOV3S9s$k
  A="ALL,TAG=$T,DATA=data_stagePURE17_v3,STEPS=324000,CHAIN=1,LADDER_SEED_OFFSET=$((1000*k))"
  J=$(sbatch --parsable --export=$A symmetry/train_mixgeo.sbatch)
  sbatch --dependency=afterany:$J --export=$A symmetry/train_mixgeo.sbatch
done
```

This writes the exports `output/models/<tag>_*` and `output/wifi_<tag>_basis.npz`. The strict test
of Table 1 reads `output/models/MIXP2v3fullfac<k>_trunk.pt` and `_cond.npz`, links to the files of
`MIXGEOV3S9`, `MIXGEOV3S9s1` and `MIXGEOV3S9s2`. `MMAP_BASE` sets where a training keeps the memory
map of its training set (see the header of `train_mixgeo.sbatch`).

**The learning curve of Sec. 3.1.** The other mixtures are trained the same way: `MIXGEOV3E20S9`,
`MIXGEOV3E160S9` and `MIXGEOV3E320S9` (2 × 10^4 to 3.2 × 10^5 events per run), `MIXGEOV3S27F`
(972000 steps), `MIXGEOV4S9` and `MIXGEOV4L` (2768 runs) and `MIXGEOV3M8S9` (eight members,
`ENS=8`). Their data sets are built by `symmetry/build_dataset.sbatch` (`make_dataset_v3.py`). The
row of 208 runs is `train_p2.sbatch` with `DESIGN_TAG=origfull` on `data_stagePURE17_orig`, with
its strict test on the data set of `make_orig_h400.py` and the 48 held-out points of the 720-run
set on those of `make_orig48.py`.

**The number of terms (Fig. 5) and the wide coupling scan (Sec. 3.4).** `bash kscan_submit.sh`
**(data, GPU)** trains Stages A to C and the Sherpa stage at every K of the scan, and
`mkdir -p ../gen/tasks ../gen/logs && bash widebox_submit.sh` **(data, GPU)** generates the runs of
the wide coupling scan into `gen/data_widebox` and trains its three references. The paper's
networks of the wide scan were trained with `STEPS=18000` and the tags `wb18single`, `wb18pool` and
`wb18local<run>`, in place of the 324000 steps and the `wb` tags written in the script. Both
scripts change to `MAXENT_DIR`, by default the Symmetry copy of the published run
`/mnt/beegfs/bassi/reweighting/maxent_code`, and submit the batch scripts by name from there, as they
sit in the analysis folder on Symmetry; elsewhere, first run `export MAXENT_DIR=$PWD` and
`cp symmetry/*.sbatch .` in `analysis/`.

**The data sets of the mixture.** `data_stagePURE17_v3` holds the 720 training runs and the 48
held-out points at 8 × 10^4 events, and `data_stagePURE17_v3_h400` the same with the held-out runs
at 4 × 10^5 events. Both are built from the generated runs by `make_dataset_v3.py`, which
`symmetry/build_dataset.sbatch` runs once for every entry of `SPECS`:

```bash
export MAXENT_DIR=$PWD SPECS="data_stagePURE17_v3:release,v3:0:0 data_stagePURE17_v3_h400:release,v3:0:4"
sbatch --export=ALL symmetry/build_dataset.sbatch
```

`make_dataset_v3.py` and `make_abc_strict.py` read and write under `MAXENT_DIR` and its parent, by
default the Symmetry copy of the published run. The samples themselves come from `gen/`
(`gen/README.md`).

## The scoring side

Every trained model of the closure tests, the learning curve, the calibration and the learning
band is scored again with the code that scored it, on the photon-free observables, with its
calibration constant refitted by the training's own procedure: the closure width on one half of
the events of every held-out run, brought as close to one as `r2_ladder.T_GRID` allows. The
learning-curve mixtures whose training runs are gone are refitted by `recalibrate_T.py` with the
same objective, and `apply_T.py --lmx` writes the new value into the exports that the later steps
read. Every result carries a fingerprint of the run, the code and the data record, and
`collect_lmx.sbatch` writes `output/scoring_check.json`, which `make_numbers.py` requires, only when
every result passes. On the login node of Symmetry:

```bash
export MAXENT_DIR=/mnt/beegfs/bassi/reweighting/lmx_runs/maxent_code
cd $MAXENT_DIR
LMX_GPU_APPROVED=1 LMX_PARTITION=h200preq LMX_GRES=gpu:h200:1 LMX_CPUS=12 LMX_MEM=128G NPAR_CAT=1 \
  bash symmetry/rerun/submit_scoring_lmx.sh
```

`submit_scoring_lmx.sh` checks RUN (`setup_lmx.py`), the particle caches and the bundle record
(`lmx_fingerprint.py precheck`), moves the earlier files of the outputs' names aside, writes the
start mark `output/.lmx_rerun_start`, and submits 19 jobs with their dependencies. The DCTR
scorings, SvdRank and the toy model go to the GPU partition, `-p h200preq --gres=gpu:h200:1 -c 12
--mem=128G`, which the variables on the command line give; everything else runs on CPU nodes,
`-p preq`. Without those variables every job runs on CPU nodes, and the DCTR scorings then take
weeks rather than hours. The jobs, all in `symmetry/rerun/`:

| job | script | what it writes in `output/` |
|---|---|---|
| `lmx_stdfac`, `lmx_stdcat` | `score_p2_array_lmx.sbatch` over `specs/std_fac.txt` and `specs/std_cat.txt` | the historical 40,000-event comparison and refitted temperature of every factorized and DCTR model, the K scan and the wide scan: `p2augfull_*`, `p2aug_*`, `p2v3full_*`, `stage_mixture_MIXGEOV3S9{,s1,s2}.json`, `p2origfull_*`, `p2augK*`, `p2v3K*`, `p2wb18*`; `temperature/<model>.json` |
| `lmx_strictfac`, `lmx_strictcat` | the same over `specs/strict_fac.txt` and `specs/strict_cat.txt` | the 200,000-event comparisons at the refitted temperatures, used for Tables 1 and 2 and Figure 5 |
| `lmx_svd` | the same over `specs/svd.txt` | `p2trunc_*`, the separate 40,000-event DCTR truncation diagnostic |
| `lmx_recal` | `recal_lc_lmx.sbatch` | the refits of the learning-curve mixtures without training runs (`recalibrate_T.py`) |
| `lmx_trunc` | `trunc_lmx.sbatch` | `truncation_fraction.json`, `param_count.json` |
| `lmx_toy` | `toy_final_lmx.sbatch` | the nine trainings of the toy model |
| `lmx_applyT`, `lmx_applyTrel` | `apply_T_lmx.sbatch`, `apply_T_rel_lmx.sbatch` | the exports with their refitted temperature, of the mixtures and Stage C, and of Stages D and E for the release |
| `lmx_widths`, `lmx_rescorev4` | `strict_closure_lmx.sbatch`, `rescore_v4_lmx.sbatch` | `widths_v2_<tag>{,h400}{,r415}.json` of the learning curve |
| `lmx_stagec` | `stagec_lmx.sbatch` | `valid_map_C_P2augfullfac0_report.json`, `single_members_C_P2augfullfac0.json`, `xp_null.json` |
| `lmx_wifiS9`, `lmx_wifiS9s1`, `lmx_wifiS9s2`, `lmx_wifiM8` | `wifi_cov_lmx.sbatch` with `TAG` | `wifi_<tag>_basis_rebuilt.npz`, the basis of the learning band with the held-out observables without the beam photons; `wifi_<tag>_fit.{npz,json}`, `wifi_<tag>_diag.json`, `wifi_coverage_<tag>.json`, `wifi_gensplit_MIXGEOV3S9.json` |
| `lmx_closmix` | `closure_mix_lmx.sbatch` | `fig_closure_mix.npz`, retained for the archived illustration, not a current paper figure |
| `lmx_collect` | `collect_lmx.sbatch` | `rank_scan_final.json`, `widebox_wb18.json`, then `scoring_check.json`; it ends `COLLECT DONE` |

How to submit again after a time-out or a failure, keeping every result that carries the
fingerprint of the run (`RESUME=1`), is in the header of `submit_scoring_lmx.sh`.

## The propagation side

Everything below runs in RUN, in bash, with this environment. Every job takes the environment of
the shell that submits it, so the block first clears every variable a script would read as a test
setting or a file override (the `unset` lines), and it exports `L`, `RUN` and `PY`, which the jobs
submitted with `--wrap '...'` expand themselves. Its second line goes into RUN and makes the
folders the jobs write their logs, numbers and figures into; it prints STOP when RUN is not there.

```bash
L=/mnt/beegfs/bassi/reweighting/lmx_runs; RUN=$L/maxent_code
cd $RUN && mkdir -p logs output/numbers output/figs || echo "STOP: no $RUN"
unset NODE_LIMIT DRAW_LIMIT NODES NDRAW OUT EXPORT REF WFIT SEED
unset TAG BOX NREF NBOOT RES_TOL MC_NP TEST_DATA TUNE_SCALE XI_NODES CENTRAL WIFI_CENTRAL NO_WIFI CONVENTIONS \
      THRUST_ORDER CHAIN_SUFFIX FIXED_POINT FO_NORM NP_SUB_ORDER ARES_EXT DATA_COV HEPDATA_DIR MAXENT_CODE TARGETS_GRID \
      NPZ CALC_POINT SUFFIX INDIR OUTDIR ENV_DIR LC_DIR THREE_TAG WIFI_TAGS COVERAGE_TAGS MAP_FILE NUMBERS_TEST \
      BANDS_NPZ BANDS_JSON VARS_JSON CALC_BINS COST_SUFFIX BENCH_DIR FIG_CONV FIG_OUT R8_OUT PNG HERO_FROM MIXCL_IN \
      MAP_IN XP_NULL RANK_IN COV_IN TOY_NPZ TOY_JSON DM_DATA BEAM_PHOTONS BEAM_PHOTONS_ALLOW_MIXED PHOTON_FREE_DIR_RE \
      POBS_DIR TORCH_THREADS OMP_NUM_THREADS LMX_PARTITION LMX_GRES LMX_GPU_APPROVED \
      CROSSCHECK_DATA CROSSCHECK_PAIR CROSSCHECK_SEED CROSSCHECK_OUT CROSSCHECK_PLAIN CROSSCHECK_DEVICE EF_VENV PYTHONPATH \
      GAUSS_REPS
unset $(compgen -e | grep -E '^(BENCH|LADDER|CONCAT|WIDEBOX|SBATCH|SALLOC|SRUN|STAGE[A-Z]*)_')
SB=/cm/shared/apps/slurm/current/bin/sbatch; PY=/mnt/beegfs/bassi/reweighting/env/torch/bin/python
export L RUN PY MAXENT_DIR=$RUN TORCH_ENV=/mnt/beegfs/bassi/reweighting/env/torch PYTHONDONTWRITEBYTECODE=1
export CONV_FACTORS=$L/conventions/factors.json MIX=linear T_DIR=$RUN/output/temperature XDG_CACHE_HOME=$L/xdg-cache
export BEAM_PHOTONS=remove
export FIGENV="PYTHONPATH=$L/pylib MPLCONFIGDIR=$L/mplconfig TEXMFVAR=$L/texmf-var TEXMFCONFIG=$L/texmf-config XDG_CACHE_HOME=$L/xdg-cache RIVET_REF=$RUN/rivet_ref PYTHONDONTWRITEBYTECODE=1"
PRE=       # :<id>:<id>... of lmx_applyT and lmx_wifiS9{,s1,s2}, as submit_scoring_lmx.sh printed them; empty once they have ended
DEP=${PRE:+--dependency=afterok$PRE}
```

`$L/pylib` holds matplotlib and PyYAML for the figure and calculation steps, which the torch
environment lacks, and scikit-learn for the PyTorch part of the PFN cross-check. The envelope test,
the three bands and the cost benchmark read the scoring side's refitted exports and learning fits,
so they wait for `lmx_applyT` and `lmx_wifiS9{,s1,s2}` (`$DEP`).

### The comparison with 2000 variations

`envelope_test.py` tests the linear covariance of Sec. 4.2 against 2000 variations, each an exact
reweighting of the reference sample to one draw from the prior of Sec. 4.1, first without the
calculation (Sec. 4.3) and then with it imposed (Sec. 5.3). `BOX` sets the ranges: `full` (the
trained box, xi over [0, 1]), `tune` (each range a third of its width, centred on the default
tunes and cut symmetrically where the edge of the box is nearer, so the default stays the centre,
and xi over [0, 1]), and `tune_sherpa`, `tune_herwig`, `full_sherpa` and `full_herwig` (one
generator alone, xi fixed at 0 or 1, over its tune-sized or its full ranges). On the full box, `NREF=1000` adds 1000 further draws, which measure the sampling
noise of the 2000. It reads the export, reference bundle and learning fit of `MIXGEOV3S9` and the
calculation of `output/thrust_targets_nlo_point.npz` **(data)**.

```bash
for b in tune full tune_sherpa tune_herwig full_sherpa full_herwig; do
  NR=0; [ $b = full ] && NR=1000
  TAG=MIXGEOV3S9 BOX=$b NREF=$NR $SB --parsable -p preq -J lmx_env_$b -c 8 -t 6:00:00 $DEP --export=ALL symmetry/envelope_test.sbatch
done
```

It writes `output/envelope_test_MIXGEOV3S9_<box>_lin.json`, with the bands, their ratios and
bootstrap uncertainties and the comparison of the correlations, and `.npz`, with the draws and the
covariances.

### The three bands of Sec. 5

```bash
for t in MIXGEOV3S9 MIXGEOV3S9s1 MIXGEOV3S9s2; do
  TAG=$t BOX=full $SB --parsable -p preq -J lmx_3b_$t -c 8 -t 6:00:00 $DEP --export=ALL symmetry/three_bands.sbatch
done
TAG=MIXGEOV3S9 BOX=full $SB --parsable -p preq -J lmx_bandchk -c 8 -t 6:00:00 $DEP --export=ALL symmetry/band_checks.sbatch
$SB --parsable -p preq -J lmx_thvars -c 4 -t 1:00:00 $DEP --export=ALL symmetry/theory_variations.sbatch
```

`three_bands.py` imposes the calculation at the central point, the default tunes of both
generators and xi = 1/2 (`CENTRAL` in `three_bands.sbatch`), and gives the statistical error, the
learning band from the WiFi fit and the theory band of every prediction, and the generator band
over 2000 points of a Latin hypercube over the full box with seed 11, before and after the
calculation. It writes `output/three_bands_<tag>_lin.{json,npz}`; `band_checks.py` writes
`output/band_checks_MIXGEOV3S9_lin.json` and `theory_variations.py`
`output/theory_variations_MIX_lin.json`.

### The calculation at its fixed point, against the data

```bash
RIVET_REF=rivet_ref THRUST_ORDER=2 python targets_nlo_point.py
env $FIGENV OUTDIR=$RUN/output $PY -B -u calc_bins.py
env $FIGENV CONV_FACTORS=$CONV_FACTORS NPZ=$RUN/output/three_bands_MIXGEOV3S9_lin.npz OUTDIR=$RUN/output $PY -B -u calc_vs_delphi.py MIXGEOV3S9_lin
```

The first wrote `output/thrust_targets_nlo_point.npz` before the run, which takes it as a fixed
input. The second, the job `lmx_calc_bins`, writes `output/calc_bins_nlo_point.json` for the shape,
event-fraction and direct-calculation comparisons in Figs. 8, 9 and 11,
with the power correction `\CalcShift`; the third, `lmx_calc_vs_delphi`, waits for the three bands
of `MIXGEOV3S9` and writes `output/calc_vs_delphi_MIXGEOV3S9_lin.json`.

The fixed point α_s(M_Z) = 0.1137, α_0(2 GeV) = 0.524 of Eq. (C.9) is the fit of Gehrmann, Luisoni
and Monni in the log-R scheme (Eur. Phys. J. C 73 (2013) 2265, the table after Eq. (43)), which
`output/calc_point.json` holds and the three scripts read (`CALC_POINT` points them to another
file). The same file sets the subtraction of the power correction to their order α_s³
(`np_sub_order`, Eq. (40) of that paper), since their α_0 was fitted with it. Their fit goes to
NNLL+NNLO and includes bottom-quark masses, while the calculation here is NNLL+NLO and massless.
`targets_nlo_point.py` evaluates the fourteen windowed moments, the window fraction, every scale and
scheme variation and the three variations of the dispersive model at the fixed point, and the
covariance Σ_c of Eq. (5.6). The third variation of the model drops the term of order α_s³ (`sub2`).
These steps run from the repository alone. The resummation tables of ARES (`ares_recovered/`) and
the fixed-order coefficients of EERAD3 (`output/eerad_*.npy`) ship with it. Rebuilding those needs
ARES and EERAD3 themselves (`ares_recovered/ares_edits.txt`, `eerad3/`).

### The independent check of every linear output

`check_linear_mix.py` recomputes the linear mixture from the export, the photon-free bundle, the
calculation, the learning fit and the factors, and checks the six envelope files, the three bands,
the band checks and the theory variations against it. It must end with `CHECK LINEAR MIX PASS`:

```bash
$SB --parsable -p preq -J lmx_check -c 8 --mem=32G -t 2:00:00 -o logs/lmx_check_%j.out --dependency=afterok:<the jobs above> \
  --wrap "cd $RUN && OMP_NUM_THREADS=8 $PY -B -u check_linear_mix.py \
    --reference output/models/MIXGEOV3S9_cond.npz --head output/models/MIXGEOV3S9_cond.npz \
    --observables output/models/MIXGEOV3S9_ref_v2_slim.npz --targets output/thrust_targets_nlo_point.npz \
    --wifi output/wifi_MIXGEOV3S9_fit.npz --factors $CONV_FACTORS \
    --envelope output/envelope_test_MIXGEOV3S9_tune_lin.npz output/envelope_test_MIXGEOV3S9_full_lin.npz \
      output/envelope_test_MIXGEOV3S9_tune_sherpa_lin.npz output/envelope_test_MIXGEOV3S9_tune_herwig_lin.npz \
      output/envelope_test_MIXGEOV3S9_full_sherpa_lin.npz output/envelope_test_MIXGEOV3S9_full_herwig_lin.npz \
    --bands output/three_bands_MIXGEOV3S9_lin --checks output/band_checks_MIXGEOV3S9_lin.json \
    --variations output/theory_variations_MIX_lin.json --out output/check_linear_mix.json"
```

### The cost benchmark

```bash
BENCH_PARTICLES=output/published/models/MIXGEOV3S9_ref.npz $SB --parsable -p preq --exclusive -J lmx_bench_cpu -t 6:00:00 $DEP --export=ALL symmetry/bench_final_cpu.sbatch
BENCH_PARTICLES=output/published/models/MIXGEOV3S9_ref.npz $SB --parsable -p h200preq --gres=gpu:h200:1 -J lmx_bench_gpu $DEP --export=ALL symmetry/bench_final_gpu.sbatch
```

The header of `bench_final_gpu.sbatch` asks for no GPU, so the GPU partition and the GPU are given
on the command line, as here; the script stops when it sees no GPU. Both write
`output/bench_final_cpu_4_lin.json` and `output/bench_final_cuda_12_lin.json` (`BENCH_MIX=linear`,
the default). The benchmark reads the particles of the published reference bundle, since the
network inputs keep every particle.

### The PFN cross-check of App. A.2

```bash
CROSSCHECK_DATA=/mnt/beegfs/bassi/reweighting/laptop_archive_20261017/data_widebox \
  $SB --parsable -p preq -J lmx_pfn --export=ALL symmetry/pfn_lmx.sbatch
```

`pfn_lmx.sbatch` trains four networks with five seeds each, 20 trainings one after another, on
runs 9000 and 9010 of the wide coupling scan. They are the energyflow PFN (TensorFlow,
`crosscheck_ef.py`), and with `crosscheck_torch.py` a plain PyTorch PFN of the energyflow
architecture, our conditional PFN at K = 1 with ReLU and the sum over particles not scaled, as the
published run trained it, and the same K = 1 network built as the paper's networks are: SiLU, and
the sum over particles divided by a scale fixed once, before training, on the first 4096 training
events. This implementation comparison is retained in the result archive; its
variant AUCs are no longer quoted in the paper. The energyflow PFN runs in
the venv `EF_VENV`, by default `lmx_runs/pfn_venv`, which has the versions of
`environment-tf.yml`; the PyTorch networks run in the torch environment, on CPUs. Each training
records its AUC and its parameter count. Only when all 20 have succeeded does
`crosscheck_summary.py` write `output/crosscheck_seeds_summary.json`, with the counts behind
`\PfnParams*` ("the same adjustable coefficients"), the mean AUC of each network (`\PfnAucEf`,
`\PfnAucPlain`, `\PfnAucOursRelu`, `\PfnAucOurs`) and the largest standard deviation of a
network's AUC over its seeds (`\PfnAucSdMax`). The job then ends with `PFN CROSSCHECK DONE`. The data are the files the
published run read, kept in `laptop_archive_20261017/data_widebox` on Symmetry; the job logs their
md5.

### The retained Gaussian null reference

```bash
$SB --parsable -p preq -J lmx_gauss -c 1 --mem=4G -t 1:00:00 -o logs/lmx_gauss_%j.out \
  --wrap "cd $RUN && OMP_NUM_THREADS=1 $PY -B -u gauss_ref.py"
```

`gauss_ref.py` computes the closure width of two samples of 4 × 10^4 events from one Gaussian
density, in 25 bins, with the paper's own estimator (the functions of `r2_ladder.py`), over 20000
pairs of samples with fixed seeds, and again with their means apart by five percent of the
standard deviation. It writes `output/gauss_ref_lin.json` and ends with `GAUSS REF DONE`.
This historical diagnostic remains an input to the macro generator but is no longer displayed
in Table 1's caption. It is not a new 200,000-event Gaussian calculation.

A note on the null of this estimator. The docstring of `width_of()` in `r2_ladder.py` says that
the null value is 1. It is not exactly: each pull is divided by its own multinomial standard
deviation, so with B bins the sum of the squared pulls has expectation B and the square of the
closure width B/(B−1). The width itself lies near 1.01 for 25 bins and near 1.05 for the five
bins of the baryon count. The paper (Sec. 3.2 and App. A.2) states this. `r2_ladder.py` is left
byte for byte as it was, since its checksum identifies the code that produced the stored results.

### The macros and the figures

After all of the above, and after the scoring side's `lmx_collect`:

```bash
$SB --parsable -p preq -J lmx_numbers -c 1 --mem=4G -t 0:20:00 -o logs/lmx_numbers_%j.out --dependency=afterok:<...> \
  --wrap 'cd $RUN && SUFFIX=_lin INDIR=$RUN/output OUTDIR=$RUN/output/numbers $PY -B -u phase2tools/make_numbers.py'
$SB --parsable -p preq -J lmx_figs_prop -c 2 --mem=8G -t 1:00:00 -o logs/lmx_figs_prop_%j.out --dependency=afterok:<...> \
  --wrap 'cd $RUN && export $FIGENV R8_OUT=$RUN/output/figs BANDS_NPZ=output/three_bands_MIXGEOV3S9_lin.npz \
    BANDS_JSON=output/three_bands_MIXGEOV3S9_lin.json VARS_JSON=output/theory_variations_MIX_lin.json \
    CALC_BINS=output/calc_bins_nlo_point.json && bad= &&
  for n in bands window obs aleph; do $PY -B -u figs_r8.py $n || bad="$bad $n"; done;
  SUFFIX=_lin $PY -B -u plot_envelope.py output/envelope_test_MIXGEOV3S9 || bad="$bad envelope";
  if [ -n "$bad" ]; then echo "FIGURES FAILED:$bad"; exit 1; fi; echo "FIGURES DONE: bands window obs aleph envelope"'
$SB --parsable -p preq -J lmx_figs_score -c 2 --mem=8G -t 1:00:00 -o logs/lmx_figs_score_%j.out --dependency=afterok:<...> \
  --wrap 'cd $RUN && export $FIGENV R8_OUT=$RUN/output/figs && bad= &&
  for n in map coverage kscan toy; do $PY -B -u figs_r8.py $n || bad="$bad $n"; done;
  if [ -n "$bad" ]; then echo "FIGURES FAILED:$bad"; exit 1; fi; echo "FIGURES DONE: map coverage kscan toy"'
$SB --parsable -p preq -J lmx_figs_cost -c 2 --mem=8G -t 0:30:00 -o logs/lmx_figs_cost_%j.out --dependency=afterok:<...> \
  --wrap 'cd $RUN && export $FIGENV R8_OUT=$RUN/output/figs && bad= &&
  { SUFFIX=_lin INDIR=$RUN/output OUTDIR=$RUN/output/numbers $PY -B -u phase2tools/make_numbers.py || bad="$bad numbers"; };
  { COST_SUFFIX=_lin BENCH_DIR=output $PY -B -u figs_r8.py cost || bad="$bad cost"; };
  { SUFFIX=_lin BENCH_DIR=output FIG_OUT=$RUN/output/figs $PY -B -u fig_cost.py || bad="$bad fig_cost"; };
  if [ -n "$bad" ]; then echo "COST FAILED:$bad"; exit 1; fi; echo COST DONE'
```

`lmx_numbers` waits for the calculation's bins and its comparison with DELPHI, the independent
check, the CPU benchmark, the Gaussian reference, the PFN job and `lmx_collect`; `lmx_figs_prop`
for the propagation alone, `lmx_figs_score` for `lmx_collect` alone, and `lmx_figs_cost` for both
benchmarks and `lmx_numbers`. Each figure keeps its published axis range when its values fit and
widens only as far as they need. The `numbers_lin.tex` that `lmx_figs_cost` writes, with all
thirteen `\Cost*`, is the paper's `numbers.tex`, and `output/figs` holds its figures (Figures,
above).

## The released networks

The release, `analysis/release/`, was packaged from the same run, after its scoring side, by four
jobs of their own:

```bash
cd $RUN; E="--export=ALL,MAXENT_DIR=$RUN"; H=symmetry/rerun
P=$($SB --parsable -p preq -J lmx_package  --dependency=afterok:<lmx_applyTrel> $E $H/package_lmx.sbatch)
T=$($SB --parsable -p preq -J lmx_table    --dependency=afterok:$P $E $H/table_lmx.sbatch)
$SB --parsable -p preq -J lmx_accuracy --dependency=afterok:$T $E $H/accuracy_lmx.sbatch
$SB --parsable -p preq -J lmx_refbuild --dependency=afterok:<lmx_3b_MIXGEOV3S9>:$P $E symmetry/release/refbuild_lmx.sbatch
```

`package_lmx.sbatch` converts `E_P2v3fullfac0`, `F_P2v3fullfac0` and `MIXGEOV3S9` (the networks of
Stages D and E and the mixture) to the numpy files of `analysis/release/models/`, after checking
that each export carries its refitted temperature, and checks them against the trained networks,
into `release_lmx/`. `table_lmx.sbatch` writes the closure tables at the held-out points
(`make_stage_table.py`, photon-free: `closure_<stage>.csv`, `closure_summary.json` and the numbers
of `STAGES.md`) into `release_lmx_table/` and the float32 gap `output/precision_gap_r8.json`
(`precision_gap.py`); `accuracy_lmx.sbatch` the accuracy law (`closure_vs_distance.py`,
`fit_accuracy_law.py`: `accuracy_law.json`, `accuracy_law_points.csv`). `refbuild_lmx.sbatch` builds
the deposit `reference_MIX17.npz` with `release/tools_build_reference.py` under
`BEAM_PHOTONS=remove`, in `lmx_runs/deposit_lin`, and compares every array with the photon-free copy
of the earlier deposit; `analysis/release/REFERENCE.md` says what it checks. It also writes
`build_reference.json`, the record of the build, and `SHA256SUMS`, which the repository's `deposit/`
holds with the two data files (`docs/DATA.md`). The other files were then copied into
`analysis/release/`. `models/MIX17_wifi.npz` is a copy of the run's
`output/wifi_MIXGEOV3S9_fit.npz`. The targets file of the deposit did not change; when only the
calculation changes, `python release/tools_build_targets.py output/thrust_targets_nlo_point.npz
targets_thrust_nlo.npz` rebuilds it and leaves the reference sample alone.

## Earlier drafts (not in the paper)

An earlier draft fitted α_s(M_Z) and α_0 to the LEP data with the generator parameters profiled,
on networks trained to an earlier standard. The paper has none of this. Its scripts stay in the
repository. Its result files, and those of the other earlier drafts, were in `output/` until the
paper's run replaced them. Commit 98c0421 still holds them, and there `python make_sec6_numbers.py`
recomputes the numbers of that draft from `output/`. The walk-throughs of that draft, section by
section with the archives each one needs, are in the git history:
`git show 1d66811:docs/REPRODUCE.md`. Earlier drafts also
mixed the two generators geometrically and kept the photons radiated by the beams in the
observables; the scripts still do that with `MIX=geometric` and `BEAM_PHOTONS` unset, and write
their outputs without `_lin`.
