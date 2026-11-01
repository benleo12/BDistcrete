# Guide for Claude Code

This repository holds the code and results of *Combined Uncertainties from Event Generators and
Precision QCD* (B. Assi and J. Thaler). The paper's LaTeX source is in the private repository
github.com/benleo12/Reweighting-generator-parameters, which Overleaf pulls from. To check the text
against the code, authors can clone it next to this one with
`git clone https://github.com/benleo12/Reweighting-generator-parameters paper`, and read
`paper/main.tex` and `paper/numbers.tex`. Section, figure, table and equation numbers below follow
the compiled paper. Run every command from `analysis/`. Results quoted below that come from
`numbers.tex` name their macro in parentheses.

## What the paper does

A generator prediction depends on the values of the generator parameters and on the choice of
generator, and a weight learned from finite samples has an uncertainty of its own. The paper
obtains these from one stored event sample of e+e- → hadrons at the Z pole and, as an application,
imposes a precision calculation on the same events.

- **Generator band.** A classifier that takes the generator parameters θ as an input reweights the
  stored sample to any point of a parameter box. Sherpa and Herwig enter as a multiplicative
  mixture with a mixing fraction ξ, so there are seventeen parameters. The generator band is the
  standard deviation of a prediction under a flat prior on the box. At linear order it follows from
  the derivatives at one point (Eq. (4.4)). The application takes it from 200 points (`\TbNodes`) of
  a Latin hypercube (Sec. 5.3, App. D.2).
- **Learning band.** The covariance of the classifier weight, from a fit of the members' products on
  events set aside from training (WiFi ensembles, Benevedes and Thaler, arXiv:2506.00113),
  propagated to every prediction. The prediction itself is the ensemble average.
- **Theory band.** A maximum-entropy weight imposes fourteen windowed moments of an NNLL+NLO thrust
  calculation with a dispersive power correction on the same events. The scale, scheme and
  dispersive-model variations of the calculation are imposed one by one, and their displacements
  are kept with their sign.

The combined band adds the three bands and the statistical error of the finite sample in
quadrature (App. D). No band needs a new generator run or a new pass of the event network.

Sec. 4 writes the flat prior in Gaussian variables, derives the linear covariance of any set of
predictions from the derivatives at one point (Eq. (4.4)) and compares it with 400 variations, each
an exact reweighting of the sample to a draw from the prior (`envelope_test.py`, `plot_envelope.py`,
`\Env*`). Sec. 5.3 repeats the comparison with the calculation imposed.

**There is no α_s fit in the paper.** α_s(M_Z) = 0.1137 (`\CalcAlphas`) and α_0 = 0.524
(`\CalcAlphaZero`) are fixed inputs of the calculation and are not varied. They are the published
fit of Gehrmann, Luisoni and Monni in the log-R scheme (Eur. Phys. J. C 73 (2013) 2265, the table
after Eq. (43)), which `output/calc_point.json` holds. An earlier draft used the point at which the
χ² of the NNLL+NLO calculation alone against ALEPH in the window is smallest (0.1135, 0.43), which
`THRUST_ORDER=2 CHAIN_SUFFIX=_nlo python thrust_chain.py` writes to `output/chain_summary_nlo.json`.
The predictions are compared with DELPHI, which played no part in choosing them.

Sec. 2 builds the method, Sec. 3 tests the classifier weights and the learning band against fresh
generator runs, Sec. 4 writes the prior in Gaussian variables, derives the linear covariance and
compares it with 400 variations, Sec. 5 imposes the thrust calculation and shows the three bands of
the thrust distribution and of seven means, and Apps. A to D give the samples and training, the
expansion of the log density ratio with a toy model, the thrust calculation and the propagation of
the uncertainties with the calculation.

## The classifier weight and how it relates to DCTR

**Estimator.** A classifier separates events generated at θ from events of a reference sample,
with θ as an input, by minimizing the binary cross entropy (Eq. (2.3)). Its output f estimates the
log density ratio, and the weight is e^{f/κ} with a calibration constant κ. This is DCTR
(Andreassen and Nachman, 2019). The paper uses its classifier with its training objective
unchanged and does not claim a new estimator.

**Factorized network** (`CondPFN` in `r2_ladder.py`). The output is an inner product,
f(Φ, θ) = Σ_k a_k(Φ) b_k(θ), k = 1..K (Eq. (2.5)), where Φ is the event.
- a(Φ) is a particle flow network: a per-particle map (100, 100, 256), a sum over particles divided
  by a constant fixed on 4096 reference events, and an event map (100, 100, 100) with K outputs.
- b(θ) is a parameter network of three layers of 100 nodes with K outputs.
- Every node applies SiLU, so b(θ) and the derivatives of the weights in θ are smooth. The
  derivatives are ⟨a(Φ), ∂b/∂θ_p⟩, inner products with the stored a(Φ).
- Every model uses K = 48 and N = 4 members, each trained from its own seed on a bootstrap resample
  of the training runs with a fifth of every run set aside. Training is 324000 Adam steps with a
  cosine step size from 10^-3 to 10^-5. The saved network whose closure width on the selection
  halves is closest to one is kept, and κ is fitted on the same halves (0.975 to 1.050,
  `\KappaMin`, `\KappaMax`). The prediction is the ensemble average divided by κ.
- `r2_ladder.py` alone defaults to other settings (K = 24, 36000 steps, ReLU). The batch scripts
  `symmetry/train_p2.sbatch` and `symmetry/train_mixgeo.sbatch` set `LADDER_ENS=4`,
  `LADDER_ACT=silu`, `LADDER_EMB_SCALE=auto`, `LADDER_LR=1e-3`, `LADDER_FIT_FRAC=0.2` and
  `LADDER_BOOTSTRAP=1`. `LADDER_K` and `LADDER_STEPS` come from `K` and `STEPS`.
  `train_p2.sbatch` requires both, and `train_mixgeo.sbatch` defaults to K = 48 and 36000 steps,
  so pass `K=48 STEPS=324000`.

**Multiplicative mixture** (`GeoMixturePFN` in `r2_ladder.py`, trained by `stage_mixture.py` with
`LADDER_HEAD_KIND=geometric`). The two generators enter as q ∝ q_S^(1−ξ) q_H^ξ (Eq. (2.7)). They
share a(Φ) and each has its own parameter network, so
f = (1 − ξ)⟨a, b_S(θ_S)⟩ + ξ⟨a, b_H(θ_H)⟩ = ⟨a, (1 − ξ) b_S + ξ b_H⟩.
- θ = (θ_S, θ_H, ξ), eight Sherpa parameters, eight Herwig parameters and the fraction
  (`LADDER_MIX_SPLIT` sets the two blocks).
- Nothing is learned in ξ. The derivative in ξ is ℓ_H − ℓ_S exactly, and ξ = 0 and ξ = 1 are one
  generator exactly. No generator produces an interior ξ, so the mixture trains on pure runs only
  (`LADDER_TARGET_PURE=1`), 352 Sherpa and 368 Herwig runs.
- Every weight stays the exponential of an inner product of stored event functions with a vector
  of the parameters, which the WiFi fit and the maximum-entropy weight both need.
- The mixture interpolates between two cluster models and contains no string model. ξ is not an
  error statement on the choice of generator.
- `MixturePFN` is the additive mixture log[(1 − ξ)e^{ℓ_S} + ξe^{ℓ_H}] of the earlier draft
  (`LADDER_HEAD_KIND=mixture`).

**Reference sample** (Sec. 2.3). The reference is the union of all training runs with equal
weight. This bounds the weight by the number of runs M at every training point. At the default
tunes, the largest classifier weight of the mixture is 3.4, 4.3 and 6.4 times the mean for
ξ = 0, 1/2 and 1 (`\TbMaxWeightSherpa`, `\TbMaxWeightMid`, `\TbMaxWeightHerwig`). Over the wide
coupling scan the mean closure widths are 1.27 for a single central run, 1.14 for the union of all
21 runs and 1.03 for the union of the 7 runs nearest each target (`\WideSingleMean`,
`\WidePoolMean`, `\WideLocalMean`). The single run gives 2.18 at α_s^sh = 0.083
(`\WideSingleWorst`). The choice works with any estimator, DCTR included.

**The DCTR network** (`cat` in the code) is trained through `symmetry/train_p2.sbatch` with `cat` as
the network in `SPECS` (`STAGE:NET:SEED`), by `r2_ladder.py`, and by `stage_mixture.py` for the
mixture, on the same runs, members, reference, checkpoint selection, κ and closure test as the
factorized network.
- `ConcatPFN` appends θ to every particle and uses the same layer sizes, with one output
  (`LADDER_HEAD_KIND=concat`).
- `ConcatGeoMixture` puts the multiplicative mixture on it. One network sees each generator's own
  parameters, zero-padded, with a one-hot generator label, and
  log w = (1 − ξ) g(Φ; θ_S, S) + ξ g(Φ; θ_H, H) (`LADDER_HEAD_KIND=concat_geometric`).
- `concat_baseline.py` holds the stand-alone baselines of the earlier draft and `LatePFN`, which
  appends θ after the sum over particles and which `bench_cost_final.py` times.

**The expansion of the log density ratio** (Sec. 2.2, App. B). Expanded about the centroid of the
training points, the log ratio is a sum of products of a function of the event and a function of θ.
With the offset of the pooled reference, there are at most K₂ = 1 + d + d(d+1)/2 functions of the event
through second order, and the first d + 1 should carry most of the accuracy.
- This is a property of the log ratio, so any network that learns it inherits it. The outputs of
  the DCTR network, truncated to their 3, 5, 5, 9 and 16 leading products for
  d = 1, 2, 3, 8 and 17, keep their closure width within 0.02 (`\SvdRankA`, `\SvdRankB`,
  `\SvdRankC`, `\SvdRankSherpa`, `\SvdRankMix`, from `symmetry/score_p2.sbatch` with
  `LADDER_TRUNCATE=1`).
- For the factorized network the expansion says how large K must be, which is the scan of Fig. 6.
- It is not an advantage over DCTR.

**What the factorized form gives** (Sec. 3.6).
- **Cost** (Fig. 7, `bench_cost_final.py`). At one parameter point, the weights of the reference
  events of the mixture and the derivatives of one prediction take 0.057 s with the factorized
  network, 19 s with θ appended after the sum and 23 minutes with the DCTR network on one CPU
  process with four threads (`\CostFacCpu`, `\CostLateCpu`, `\CostCatCpuMin`). On one H200 they take
  5.0 ms, 0.087 s and 5.7 s (`\CostFacGpu`, `\CostLateGpu`, `\CostCatGpu`). The maximum-entropy
  weight and its derivatives take 0.2 s on the CPU for every network (`\CostRestCpu`). The weights
  alone take 0.028 s with the factorized network (`\CostWFacCpu`), so the 400 variations take 11 s,
  1.1 hours and 1.9 days on one CPU process (`\CostVarFacSec`, `\CostVarLateHours`,
  `\CostVarCatDays`). The DCTR and late networks are timed with untrained weights, since the cost
  does not depend on them.
- **Accuracy** (Table 1). Up to eight parameters the two networks are comparable, and neither is
  better at every stage. At seventeen parameters both use the multiplicative mixture, and the
  factorized network is the more accurate, 1.158 against 1.341 on the strict test
  (`\AccDenseMixFacStrict`, `\AccDenseMixCatStrict`).

In short: the estimator is DCTR's. What the paper adds is the lower cost of the factorized form, the
derivatives in closed form that give the linear covariance, and the exact multiplicative mixture,
which makes the choice of generator a continuous parameter. The mixture works on the DCTR network as
well.

## The closure test (Sec. 3.2) and Table 1

For each observable and held-out point, the reweighted reference and a fresh generator run are
histogrammed, both normalized, and each bin's difference is divided by its statistical
uncertainty (the square root of the summed squared weights for the reweighted sample, combined
with that of the fresh run). The closure width is the square root of χ²/ndf, with ndf the number
of bins minus one. It is about one when the two differ only by statistical fluctuations.
- Every fresh run is split into a selection half, which chooses the saved network and fits κ, and
  a comparison half, which the tables use.
- The standard test uses the comparison half of each fresh run, 4 × 10^4 events. The strict test
  extends each run by four further runs at the same point and uses the comparison half of the
  result, 2 × 10^5 events.
- Two halves of one fresh run give 0.99 to 1.03, 0.93 to 1.13 and 0.88 to 1.05 at Stages A, B and C
  (`\Calib{A,B,C}{Min,Max}`).
- The observables are 1 − T, the multiplicity of all kept particles, B_tot, ρ_H, the baryon count
  and the kaon fraction (`1_minus_thrust`, `mult_total`, `B_total`, `rho_heavy`, `nbaryon`,
  `strange` in the code). "master" in the result files is the mean over the observables.
- The code is `width_of`, `pulls` and `make_bins` in `r2_ladder.py`, `ladder_widths_big.py` for
  an exported mixture and `eval_valid.py` for Fig. 4.

Table 1 gives the mean and the standard deviation over the three ensembles
(`\Acc<Set><Stage><Net><Test>[Sd]`, with `Set` `Aug` for Stages A to C and `Dense` for the rest,
and `Stage` `A`, `B`, `C`, `Sherpa`, `Herwig`, `Mix`):

| model | parameters | factorized, standard | factorized, strict | uniform weights | DCTR network, standard | DCTR network, strict |
|---|---|---|---|---|---|---|
| Stage A | 1 | 1.016 ± 0.017 | 1.188 ± 0.013 | 3.0, 5.9 | 1.022 ± 0.028 | 1.175 ± 0.006 |
| Stage B | 2 | 1.014 ± 0.014 | 1.164 ± 0.053 | 6.2, 12.8 | 1.022 ± 0.022 | 1.132 ± 0.004 |
| Stage C | 3 | 0.970 ± 0.002 | 1.110 ± 0.076 | 3.5, 7.5 | 0.979 ± 0.005 | 1.138 ± 0.123 |
| Stage D (Sherpa) | 8 | 1.053 ± 0.006 | 1.095 ± 0.018 | 5.9, 12.4 | 1.060 ± 0.009 | 1.120 ± 0.009 |
| Stage E (Herwig) | 8 | 1.073 ± 0.004 | 1.231 ± 0.013 | 6.6, 14.2 | 1.057 ± 0.008 | 1.233 ± 0.029 |
| mixture | 17 | 1.058 ± 0.013 | 1.158 ± 0.019 | 6.3, 12.7 | 1.115 ± 0.003 | 1.341 ± 0.020 |

The factorized network gives 0.97 to 1.07 on the standard test and 1.10 to 1.23 on the strict test
(`\AccFinalFac*`), against 3.0 to 6.6 for uniform weights (`\AccFinalUniStd*`). The baryon count
is reproduced least accurately: 1.40, 1.71 and 1.53 on the strict test for Stages D and E and the
mixture (`\ObsDense{Sherpa,Herwig,Mix}NbStrict`, Table 2).

## The learning band (Secs. 2.6, 3.5 and 4.2, App. A.2)

- **Basis.** A constant and the N² = 16 products f_ij = ⟨a_i(Φ), b_j(θ)⟩ of the event function of
  member i with the parameter function of member j, with b_j of the mixture form, so 17 coefficients
  (Eq. (2.12)). The ensemble average is ω_ij = δ_ij/(Nκ), ω_0 = 0.
- **Fit.** The fit pairs 4000 of the 16000 set-aside events of every training run with 4000
  events drawn from the set-aside events of all runs, both at the run's θ. The loss of Eq. (2.13) is
  convex, and Newton's method from the ensemble average converges in 8, 7 and 9 iterations for the
  three ensembles (`\WifiNewton{One,Two,Three}`). The covariance is C = V⁻¹UV⁻¹, with U counting each
  reference event once.
- **The fit is not the prediction.** The fitted combination reproduces the fresh runs much worse
  than the ensemble average, closure widths 3.75 to 4.13 against 1.04 to 1.07 (Table 5,
  `\WifiWidthFit*`, `\WifiWidthMean*`), because about half of the reference events paired with each
  run come from the other generator (`wifi_gensplit.py`). The prediction is therefore the ensemble
  average, and the fit gives only its covariance.
- **Band.** σ²_learn = gᵀCg, with g the derivative of the prediction in the coefficients at the
  ensemble average. With the calculation imposed g includes the response of the multipliers
  (Eq. (D.5)).
- **Coverage** (Fig. 5, `wifi_coverage.py`). At the 48 held-out points of the strict test the
  reweighted means of five observables differ from the fresh runs by 0.2 to 0.7 percent
  (`\CovErrMin`, `\CovErrMax`). The root mean square of the pulls is 1.6 to 2.9 without the band
  and 0.38 to 0.81 with it (`\CovZWithout*`, `\CovZWith*`). With eight members it is 1.5 to 2.3
  and 0.27 to 0.38 (`\CovEight*`). The band covers the error and is larger than it.
- **Code.** `wifi_embed.py` (the basis, run at the end of `symmetry/train_mixgeo.sbatch`),
  `wifi_fit.py` (Newton and the sandwich covariance), `wifi_diag.py` (Table 5),
  `wifi_coverage.py`, `wifi_gensplit.py`, and `three_bands.py`, which propagates C.
  `release/models/MIX17_wifi.npz` is a copy of `output/wifi_MIXGEOV3S9_fit.npz`.

## The maximum-entropy weights and the three bands (Sec. 5, Apps. C and D)

- **Maximum-entropy weight.** p* = q e^{λ·m}/Z solves the penalized problem of Eq. (5.4), whose
  penalty uses P_win² Σ_c. The multipliers minimize the convex function of Eq. (D.1) by Newton's
  method. The code is `Model.tilt_on` and `tilt_on_t` in `directlib.py`, and `Model.sigma` adds
  the diagonal term of Eq. (5.6).
- **Constraints.** Fourteen functions h_ab = (τ^a ln^b τ − c_ab) Θ(0.05 ≤ τ < 1/3) with
  1 ≤ a + b ≤ 4 (Eq. (5.5)). The window fraction P_win is left to the generators, and imposing the
  calculation changes it by 0.05 percent (`\TbPwinChangePercent`).
- **Calculation** (App. C). ARES NNLL matched to EERAD3 at O(α_s²) in the log-R scheme, with the
  dispersive shift of `np_shift.py`. `targets_nlo_point.py` (`THRUST_ORDER=2`), at the point of
  `output/calc_point.json`, gives `output/thrust_targets_nlo_point.npz`: the moments, the window
  fraction, the five perturbative variations (μ_R and x_V up and down in log-R, and modified-R at
  central scales) and the three variations of the dispersive model (the Milan factor up and down
  by twenty percent, and dropping the subtraction term of order α_s³), all at the fixed point, and
  Σ_c. `calc_bins.py` bins the calculation in the ALEPH and DELPHI bins (Fig. 14).
- **`three_bands.py`** (`TAG=MIXGEOV3S9 sbatch symmetry/three_bands.sbatch`). The central point is
  the default tunes of both generators with ξ = 1/2 (`CENTRAL` in the batch script), and the
  prediction is the ensemble average (`WIFI_CENTRAL=0`). The generator band is the standard
  deviation over `NODES=200` points of a Latin hypercube (`SEED=11`), with the multipliers solved
  again at each point, before and after imposing. The theory band solves the multipliers again for
  each variation and combines the signed displacements with Eq. (5.7). The learning band is gᵀCg,
  and the statistical error comes from resampling the reference at linear order (App. D.2). The
  observables are ⟨τ⟩_win, P_win, ⟨1 − T⟩, ⟨B_tot⟩, ⟨ρ_H⟩, ⟨n_ch⟩, ⟨n_b⟩ and the thrust
  distribution in the DELPHI bins, normalized over the bins above τ = 0.05, with the bins below
  under keys ending in `_low`.
- `theory_variations.py` gives the signed displacement of every variation and each generator alone
  (ξ = 0, 1) with and without the calculation. `band_checks.py` gives the statistical errors, the
  part of the generator band from ξ and the largest weights. `calc_vs_delphi.py` compares the
  prediction, the calculation and ALEPH with DELPHI in the bins wholly inside the window.
- **Results** (Sec. 5.4). Inside the window the generator band is 0.78 to 8.6 percent per bin for
  the generators alone and 0.12 to 2.5 percent with the calculation imposed
  (`\TbBinGen{Before,After}Win{Min,Max}`). The theory band is 0.49 to 20 percent below the
  prediction and 0.43 to 6.2 percent above it (`\TbBinTheory{Down,Up}Win*`). The learning band is
  0.0069 to 0.22 percent (`\TbBinLearnWin*`). Above τ = 0.2 the prediction lies 6.1 to 11 percent
  below DELPHI (`\TbDeficitTop*`), as the calculation does. Fig. 12 and Sec. 5.5 give the seven means
  (`\Tb<observable>{GenOnly,Imposed,GenBefore,GenAfter,Theory,Learn,Comb}` in `numbers.tex`, of which
  the paper quotes a few).

## Map of the code

| file | role |
|---|---|
| `r2_ladder.py` | the networks (`CondPFN`, `MixturePFN`, `GeoMixturePFN`, `ConcatPFN`, `ConcatGeoMixture`), training, checkpoint selection, κ, the closure statistic, scoring a saved model (`LADDER_SCORE_FROM`), the truncation test (`LADDER_TRUNCATE`) |
| `stage_mixture.py`, `mixture_cfg.py` | the training of the mixture and the configuration of its data sets |
| `ladder_widths_big.py`, `rebuild_ref_lowmem.py` | the closure of an exported mixture, and its reference bundle |
| `eval_valid.py`, `xp_null.py` | Fig. 4 |
| `fig_closure_mix.py` | the inputs of Fig. 3 |
| `make_dataset_v3.py`, `make_abc_strict.py`, `make_orig48.py`, `make_orig_h400.py`, `make_pure_dataset.py` | the data sets |
| `wifi_embed.py`, `wifi_fit.py`, `wifi_diag.py`, `wifi_coverage.py`, `wifi_gensplit.py` | the learning band |
| `directlib.py` | `Model`: the export reader, the weights and their derivatives, the maximum-entropy weight, the χ² to the LEP data |
| `output/calc_point.json`, `thrust_chain.py`, `np_shift.py`, `match_v3.py`, `targets_nlo_point.py`, `calc_bins.py` | the thrust calculation (App. C) |
| `envelope_test.py`, `plot_envelope.py` | the comparison with 400 variations (Secs. 4.3 and 5.3, Figs. 8 to 10) |
| `three_bands.py`, `theory_variations.py`, `band_checks.py`, `calc_vs_delphi.py` | the three bands (Sec. 5) |
| `toy_final.py` | the toy (App. B.1) |
| `kscan_submit.sh`, `kscan_collect.py`, `widebox_submit.sh`, `widebox_collect.py` | the number-of-terms scan and the wide coupling scan |
| `bench_cost_final.py` | the cost at one parameter point (Fig. 7) |
| `single_members.py`, `trunc_frac.py`, `param_count.py`, `run_crosscheck_seeds.sh` | the checks of App. A.2 |
| `phase2tools/make_numbers.py` | every macro of `numbers.tex` |
| `figs_r8.py` | the data figures, Figs. 3 to 7 and 11 to 14 |
| `package_release.py`, `make_stage_table.py`, `precision_gap.py`, `closure_vs_distance.py`, `fit_accuracy_law.py` | the released networks |
| `release/` | the released networks with a numpy-only reader (`gentune`), parameter boxes in `release/STAGES.md`, parameter order in `release/gentune/axes.py`, the generator scripts |
| `symmetry/` | the batch scripts of the paper |
| `compute_efps.py`, `compute_shapes_only.py`, `extract_particles_full_flavor.py` | the event shapes and the particle inputs from HepMC |
| `fit_alpha0.py`, `lepcov/lepdata_cov.py`, `rivet_ref/` | the LEP measurements |

## Names in the paper and in the code

| paper | code |
|---|---|
| Stages A, B, C | `A`, `B`, `C`. Densified designs `gen/designs/stage{A,B,C}_design_aug.csv`, runs in `gen/data_abc`, `DESIGN_TAG` `augfull` (and `aug`), exports `<S>_P2augfull<net><k>`, macros `\Acc/ObsAug*` |
| Stage D (Sherpa, eight parameters) | `E`. Design `stageE_design_v3all.csv`, `DESIGN_TAG` `v3full`, exports `E_P2v3full<net><k>`, macros `*DenseSherpa*`, released as `E` |
| Stage E (Herwig, eight parameters) | `F`. Design `stageF_design_v3all.csv`, exports `F_P2v3full<net><k>`, macros `*DenseHerwig*`, released as `F` |
| the seventeen-parameter mixture, factorized | `MIX`. `MIXGEOV3S9`, `MIXGEOV3S9s1`, `MIXGEOV3S9s2` (linked as `MIXP2v3fullfac0` to `2`), runs in `data_stagePURE17_v3`, macros `*DenseMix*`, released as `MIX17` |
| the seventeen-parameter mixture, DCTR network | `MIXP2v3fullcat0` to `2` |
| the three ensembles | seeds 0, 1 and 2 (`LADDER_SEED_OFFSET` 0, 1000, 2000): `_s<k>` in the result files, `<net><k>` in the exports, `s1` and `s2` in the mixture tags, `One`, `Two`, `Three` in the macros |
| factorized and DCTR network | `fac` and `cat` |
| standard and strict test | the comparison ("report") half of the held-out runs at 8 × 10^4 events, and the same on the `*_h400` data sets (`strict` or `h400` in the file names) |
| selection half | the "stop" split |
| calibration constant κ | the temperature `T` (`temperature` in the result files and exports) |
| mixing fraction ξ | `f`, the last component of θ |
| shower coupling α_s^sh | `ALPHAS(MZ)` in Sherpa, `AlphaIn` in Herwig (`alphas`, `alpha_fsr` in the designs) |
| the wide coupling scan | stage `W`, `DESIGN_TAG` `wb18single`, `wb18pool`, `wb18local<run>` |
| the number-of-terms scan | `DESIGN_TAG` `augK<K>` (Stages A to C) and `v3K<K>` (Stage D) |
| the learning curve of Sec. 3.1 | `MIXGEOV3E20S9`, `MIXGEOV3E160S9`, `MIXGEOV3E320S9`, `MIXGEOV3S27F`, `MIXGEOV4S9`, `MIXGEOV4L`, `MIXGEOV3M8S9`, and the 208-run cells `origfull` (macros `\Lc*`) |
| `P2` in the tags | the rerun of every model at the training standard of the paper |

The macro sets `Grid` and `Orig` in `numbers.tex` belong to other designs of Stages A to C and of
the 208-run set. The paper quotes none of them.

## Re-running

- `docs/REPRODUCE.md` maps every figure, table and number to its command and inputs.
- `python phase2tools/make_numbers.py output` writes `phase2tools/numbers.tex`, 624 macros
  identical to the paper's, `python figs_r8.py` draws Figs. 3 to 7 and 11 to 14 and
  `python plot_envelope.py output/envelope_test_MIXGEOV3S9 tune full` draws Figs. 8 to 10 into
  `output/`. All three read only files in git: the result files in `output/` and, for `make_numbers.py`,
  `data_stagePURE17/meta.json`, the held-out points of the 208-run set. The figures need a LaTeX installation,
  which `environment.yml` does not provide.
- From the repository alone you can also redo the thrust calculation and Fig. 14
  (`targets_nlo_point.py`, `calc_bins.py`, `calc_vs_delphi.py`), the collection of the scan of the
  number of terms and the release self-test. The scripts that read the LEP measurements need
  `RIVET_REF=rivet_ref`, otherwise they look for a local Rivet installation.
- With the two files of the release `data-r8` (`docs/DATA.md`), `release/examples/reproduce_paper.py`
  reproduces the central predictions and the generator band with numpy alone.
- Everything else needs the network exports and event samples of `docs/DATA.md`, which are not in
  git and are available on request until they are archived with the paper.
- The batch scripts in `symmetry/` change to `${MAXENT_DIR:-/mnt/beegfs/bassi/reweighting/maxent_code}`.
  Set `MAXENT_DIR` to your `analysis/` folder and submit from there with `--export=ALL`.
- Training needs the samples and a GPU, and it writes its exports and result files under the
  published names.

## Things to know

- Every event shape is computed with the exact thrust axis (`compute_efps.py`, App. A.1). The
  network inputs use the axis of an iterative search (`extract_particles_full_flavor.py`). Every
  sample of the paper was extracted this way.
- The run number is the generator seed, except run 32077 (seed 5032077,
  `gen/designs/seed_overrides.csv`). The extensions of a held-out run use run + k × 10^6.
- `phase2tools/make_numbers.py` writes `numbers.tex` and `numbers.json` next to itself. The first
  line of `numbers.tex` names `phase2/make_numbers.py`, the script's place on the laptop. Do not
  change the text it writes, because the paper's `numbers.tex` must stay identical to it.
- A few numbers of the paper are typed rather than macros, for example the parameter count of the
  event network in App. A.2 (`param_count.py`) and the shift a_τP = 0.0093 of App. C.2
  (`np_shift.shift(1.0, 0.524, asmz=0.1137)` with `NP_SUB_ORDER=3`).
- `python figs_r8.py map` takes the dotted tick of Fig. 4 from the `mean` of `output/xp_null.json`.
  `XP_NULL` overrides it.
- Hadron rates are where the accuracy runs out. The baryon count closes worst in the larger models
  (Table 2, Fig. 4(b)).
- External codes are not included: Sherpa 3.0.4, Herwig 7.3.0, ARES and EERAD3. The README lists
  the versions, and the patches are in `ares_recovered/` and `eerad3/`.
- `make_dataset_v3.py`, `make_abc_strict.py`, `kscan_submit.sh` and `widebox_submit.sh` take their
  paths from `MAXENT_DIR` and its parent, with the Symmetry copy as default. The two submit scripts
  keep the flat layout of Symmetry, where the batch scripts sit in the analysis folder, and submit
  them by name from there, so elsewhere they need `cp symmetry/*.sbatch .` first.

## Earlier draft with an α_s fit (not in the paper)

- The earlier draft fitted α_s(M_Z) and α_0 to the thrust distributions of ALEPH, DELPHI and OPAL
  and the L3 multiplicity with the seventeen generator parameters profiled (its Sec. 6). The paper
  runs no such fit and takes the two parameters of the calculation from a published fit (Sec. 5.2).
  The only 208-run mixture of the paper's training standard is the `origfull` row of the learning
  curve in `numbers.tex`, which the paper does not quote.
- Its code stays in the repository: `make_sec6_numbers.py`, `profile_column.py`,
  `polish_parallel.py`, `profile_rows.py`, `bands_exact.py`, `surface_rows.py`, `pseudo_rows*.py`,
  `fit_summary_direct.py`, `fig_anchored_fit.py`, the 208-run mixtures `MIXGEO`, `MIXGEOpm`,
  `MIXGEO8`, `MIXGEO16`, `MIXGEO32` with `run_mixgeo_local.sh`, `fig_three_bands*.py`, `figs_pro.py`
  and `wifi_scan_table.py`, and the batch scripts in `perlmutter/` and `farm/`. Its data are the
  release `data-v1` (`docs/DATA.md`), and `release/` keeps its network `C_1M`. Its guide is
  `git show 1d66811:CLAUDE.md`.
- **The 6000 steps** of the macros `\CostFit*` (`COST_STEPS` in `phase2tools/make_numbers.py`) come
  from the forty pseudo-data fits of that draft, which needed a median of 6315 evaluations of the χ²
  and its gradient per fit. The paper does not use these macros.
- Its seed study (`perlmutter/seedstudy.sbatch`, `output/seedstudy/`) is superseded by the three
  ensembles of every model in Table 1.
