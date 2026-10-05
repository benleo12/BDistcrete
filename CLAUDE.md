# Guide for Claude Code

This repository holds the code and results of *Beyond Discrete Variations: Continuous Generator
Uncertainties Anchored to Precision QCD* (B. Assi and J. Thaler). The paper's text is not in the
repository. Ask the user for the PDF when a question needs it. Section, figure and table numbers
below follow the paper. Run every command from `analysis/`.

## What the paper does

Generator uncertainties (parton shower, hadronization, the choice of generator) are usually
estimated from a few discrete variations. We make one stored event sample a continuous,
differentiable function of the generator parameters by reweighting it. A maximum-entropy
reweighting then fixes the thrust distribution of that sample to an NNLL+NNLO calculation. A fit to
LEP data treats every generator parameter as a nuisance parameter and determines the strong
coupling.

## The reweighting and how it relates to DCTR

**Estimator.** A classifier separates events generated at parameters θ from events of a reference
sample, with θ as an input. Its logit estimates the log density ratio, and the weights are its
exponential. This is DCTR (Andreassen and Nachman, 2019). The paper does not claim a new estimator.

**Network** (`CondPFN` in `r2_ladder.py`). The logit is an inner product,
f(Φ, θ) = Σ_k a_k(Φ) b_k(θ), k = 1..K, where Φ is the event (its particles).
- a(Φ) is a particle flow network: a per-particle network, a sum over particles, then a network
  with K outputs.
- b(θ) is a network of the parameters alone.
- Production settings: K = 24 for Stages A to C and the seven-parameter mixture, K = 48 for the
  eight-parameter Sherpa and Herwig models and the seventeen-parameter mixture. Four networks are
  averaged and the logit is divided by a calibration temperature.
- `r2_ladder.py` does not default to the production settings. Set
  `LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3`.

**Mixture head** (`MixturePFN` in `r2_ladder.py`, trained by `stage_mixture.py`). Two generators
share a(Φ) and each has its own parameter network: l_S = ⟨a(Φ), b_S(θ_S)⟩ and
l_H = ⟨a(Φ), b_H(θ_H)⟩. With a mixing fraction ξ, the logit is
log[(1 − ξ) e^{l_S} + ξ e^{l_H}]. Nothing is learned in ξ. ξ = 0 is Sherpa alone, ξ = 1 is
Herwig alone, and the derivative in ξ is exact. This makes the choice of generator a continuous
nuisance parameter. The seventeen-parameter model (MIX17) has 8 Sherpa parameters, 8 Herwig
parameters and ξ.

**Reference sample.** The reference is the union of all training runs, not a single central run.
This keeps the weights bounded over wide parameter ranges. At α_s = 0.083 a single central
reference leaves 13 effective events, and the union leaves 56.9k (Sec. 5.5). The choice works with
any estimator, DCTR included.

**DCTR in this repository** is `concat_baseline.py`.
- `CONCAT_MODE=early`, the default, appends θ to every particle (DCTR proper).
- `CONCAT_MODE=late` appends θ after the sum over particles.
- Both use the same particle network sizes, training recipe and closure test as the factorized
  network.

**The score expansion, and what it does and does not show** (Sec. 2.3, App. C).
- Expanding the log ratio log p(Φ|θ)/p(Φ|θ₀) around θ₀ writes it as a sum of products, each a
  function of θ times a function of Φ.
  - At first order there are d + 1 terms, which carry most of the accuracy.
  - At second order there are at most K₂ = 1 + d + d(d+1)/2.
- This is a property of the log ratio, so any network that learns the ratio inherits it.
  `python dctr_rank.py` prints the effective rank of the trained DCTR networks: 3, 6, 8, 12 and 19
  for d = 1, 2, 3, 8 and 17. These follow the same counts, although nothing in those networks
  imposes a rank.
- For the factorized network the expansion does two things. It says how large K must be, which
  is the term-count scan of Sec. 5.4 and Fig. 7. It also explains why restricting the logit to K
  products costs little accuracy.
- It is not an advantage over DCTR.

**What the factorized form gives**, which is where it differs from DCTR in practice:
- **Cost.** a(Φ) is computed once per event and stored. The weights at a new θ, and their
  derivatives in θ, then cost one K-term inner product per event.
  - Early fusion reruns the per-particle network on every particle of every event at each new θ.
  - Late fusion reruns the network after the sum on every event.
  - `bench_cost.py` measures this on the seventeen-parameter fit. On a Perlmutter CPU node (32
    threads, 1.15M reference events, four networks), one step of the profile fit takes 0.59 s
    with the factorized form and 14.3 s with late fusion. The step is χ² and its gradient in the
    17 nuisance parameters, with the maximum-entropy reweighting re-solved.
  - The early-fusion and GPU timings are in `output/bench_cost_*.json` once they finish.
- **The exact mixture head.** Closure widths (one means agreement within statistics) for
  factorized against DCTR are in Table 4 and `output/concat_baseline_*.json`.
  - They are comparable up to eight parameters: 0.95 against 0.98, 1.20 against 1.08, 1.03
    against 0.99, and 1.06 against 1.03.
  - The factorized network is clearly better on the seventeen-parameter mixture, 1.09 against
    1.38.

## The validation statistic (Sec. 4)

For each observable and each parameter point held out of training, the weighted reference
histogram is compared with a fresh generator run at that point.
- Each bin's difference is divided by its pooled multinomial uncertainty.
- The closure width is the square root of the mean squared deviation, with one degree of freedom
  removed for the normalization. It is about one when the two differ only by statistical
  fluctuations.
- "master" in the result files combines all observables.

The observables are 1 − T, total multiplicity, total broadening, heavy jet mass, baryon count and
strange fraction. The code is `ladder_widths_dedup.py`, `ladder_widths_big.py` and the function
`width_of` in `r2_ladder.py`.

## Anchoring and the fit (Sec. 6)

- **Maximum-entropy reweighting.** At given generator parameters, the reweighted sample is changed
  as little as possible (minimum relative entropy) so that its windowed thrust moments match the
  calculation at (α_s, α₀). The code is `tilt_on_t` in `directlib.py`.
- **The calculation.** NNLL from ARES and NNLO from EERAD3 are matched in `thrust_chain.py`. The
  dispersive nonperturbative shift with parameter α₀ is in `np_shift.py`, the theory covariance
  in `theory_cov.py`, and the targets on the (α_s, α₀) grid in `anchor_targets_grid.py`.
- **The fit.**
  - Data: the ALEPH, DELPHI and OPAL thrust distributions above τ = 0.05 and the L3 charged
    multiplicity.
  - α_s and α₀ are fitted, and the 17 generator parameters are profiled.
  - The code is `profile_rows.py`, `bands_exact.py` and `surface_rows.py`, on top of
    `directlib.py`.
  - Result: α_s(M_Z) = 0.1202 +0.0030 −0.0032 (exp) +0.0116 −0.0013 (pert), α₀ = 0.414, χ² 54.6
    for 59 degrees of freedom.
- **Every number of Sec. 6:** `python make_sec6_numbers.py` recomputes them from `output/`.

## Map of the code

| file | role |
|---|---|
| `r2_ladder.py` | factorized network, training, selection, temperature, closure statistic (Stages A to C) |
| `stage_mixture.py`, `mixture_cfg.py`, `make_mixture_data.py` | mixture data sets and the mixture head |
| `concat_baseline.py` | DCTR baselines (early and late fusion), with their SVD |
| `bench_cost.py` | cost of the three designs on the seventeen-parameter fit |
| `ladder_widths_dedup.py`, `ladder_widths_big.py`, `rate_check_*.py` | closure tables (Tables 1 to 4) |
| `compute_efps.py`, `extract_particles_full*.py` | observables and particle inputs from HepMC (exact thrust axis) |
| `directlib.py`, `fitlib.py` | the maximum-entropy reweighting and the profile fit |
| `thrust_chain.py`, `np_shift.py`, `theory_cov.py` | the thrust calculation and its uncertainties (App. D) |
| `release/` | the released models with a numpy-only reader (`gentune`), parameter boxes in `release/STAGES.md`, parameter order in `release/gentune/axes.py`, generator cards |
| `perlmutter/`, `farm/` | batch scripts as run, with their settings |

## Re-running

- `docs/REPRODUCE.md` maps every figure, table and number to its command, and names the data
  archives each section needs.
- `docs/DATA.md` lists the event samples and network exports, which are too large for git, as
  archives of the GitHub release `data-v1` (A0, B1 to B13, C1 to C9) and says what each holds.
  `tools/get_data.sh <group or archive>` (at the repository root, so `../tools/get_data.sh` from
  `analysis/`) downloads them, checks their SHA256 and unpacks them into `analysis/`. The groups
  follow the paper, for example `sec6`, `table1`, `appA` and `train-E`, and `--list` shows them
  all. The script also links the reference names the scripts open to the slim reference files.
  The README has a table of which archives each cross-check needs.
- From the repository alone you can redo:
  - every figure made by code except Fig. 5
  - every number of Sec. 6 from the stored results
  - the thrust calculation's fit to ALEPH
  - `dctr_rank.py`
- The closure tests, the fit itself and the cost benchmark need the exports.
- Retraining needs the event samples and a GPU. It overwrites the exports in `output/models/`. Run
  `../tools/get_data.sh --unlink` first, so that training does not write through the reference
  links into the slim files.
- The figures need a LaTeX installation, which `environment.yml` does not provide.

## Checks in progress

These were queued on Perlmutter after the paper's numbers were fixed. Their result files are not
in `output/` yet, so do not quote them as results. `docs/REPRODUCE.md` (Sec. 5.8) has the commands.
- **Seed spreads for Table 4.** `perlmutter/seedstudy.sbatch` retrains the factorized, early-fusion
  and late-fusion networks with other training seeds, so that each closure width of Table 4 gets a
  spread over seeds. Results go to `output/seedstudy/`. A seed n trains the four networks with
  seeds n to n + 3, so only multiples of 4 give members independent of the published seeds 0 to 3 (100 to 103 for
  the factorized Stage B).
- **The mixture-head control.** The same script with the designs `mixe` and `mixl` trains early
  and late fusion with the exact mixture head (`CONCAT_HEAD=mixture`) on the seventeen-parameter
  mixture. It tests whether the gap there, 1.09 against 1.38, comes from the factorized form or
  from the mixture head.
- **Late-fusion accuracy.** `perlmutter/concat_late.sbatch` for Stage C and the mixture, writing
  `output/concat_baseline_{C,MIX}_late.json`.
- **Cost on CPU and GPU.** `perlmutter/bench_cpu.sbatch` (4 and 128 threads) and
  `perlmutter/bench_gpu.sbatch`, writing `output/bench_cost_cpu_4.json`, `bench_cost_cpu_128.json`
  and `bench_cost_cuda_32.json`. The 0.59 s and 14.3 s above come from a 32-thread run without the
  early design.

## Things to know

- The closure tests of Secs. 5.2 to 5.6 used an earlier thrust-axis finder,
  `compute_efps_old_axis.py`, and so did the temperature fits of App. B. Everything else uses the
  exact one. The network inputs are the same.
- Four Stage B runs lost their seeds and cannot be regenerated bit for bit.
- External codes are not included: Sherpa, Herwig, Rivet, ARES and EERAD3. The README lists the
  versions, and the patches are in `ares_recovered/` and `eerad3/`.
- Hadron rates are where the accuracy runs out. The baryon count closes worst, at the percent
  level (Secs. 5.3 and 5.7).
