# Reproducing the paper

All commands run from `analysis/`. Result files named below are in `analysis/output/` unless marked
**(data)**, which means they need the network exports or event samples of `docs/DATA.md`. Section,
figure and table numbers follow the paper.

The networks are trained with the production settings, which `r2_ladder.py` does not use by default:

```bash
export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
```

## Section 5.1, the toy model

| item | produced by | from |
|---|---|---|
| Fig. 4 (a), (b) | `figs_pro.py toy` | `toy_fvf_data.json` (`validate_pfn.py`) |
| Fig. 4 (c) | `figs_pro.py toy` | `toy_ab_data.json` (`ab_rank.py`) |
| closure widths 17.9, 2.89, 1.08, flat for K = 2 to 32 | `kscan_toy_ruler.py` via `run_kscan_toy_ruler.sh`, merged by `kscan_toy_ruler_merge.py` | `kscan_toy_ruler.json` |
| 1.05 ± 0.02, 1.08 ± 0.09, N* = 89k, R² = 0.99, intercept 0.86 | `toy_law_ruler.py --mode split|law|null` via `run_toy_law_ruler.sh`, merged by `toy_law_ruler_merge.py` | `kscan_toy_law_ruler.json` |

## Section 5.2, the ladder (Table 1, Fig. 5)

| item | produced by | from |
|---|---|---|
| Stage A, B, C networks | `campaign_gpu.sh` steps 1a to 1c (`r2_ladder.py`, Stage B with `LADDER_SEED_OFFSET=100`) | event samples **(data)** |
| Table 1, Stage A | `r2_ladder.py` | `ladder_final.json` |
| Table 1, Stages B and C, and the per-observable ranges | `ladder_widths_dedup.py B` and `C` (held-out runs 6704 and 6902 dropped, they repeat training runs) | `ladder_dedup_*.json`; exports **(data)** |
| mean baryon and strange rates, \|z\| and percent | `rate_check_dedup.py B` and `C` (prints) | exports **(data)** |
| energy-fraction width 0.85, same-run 0.91 ± 0.13 | mean over runs 6900, 6901, 6903, 6904 of `valid_map.json` (`eval_valid.py`) and `xp_samerun.json` (`audit_xp_samerun.py`) | included |
| Fig. 5 | `figs_pro.py hero` | `C_cond.npz`, `C_ref.npz`, run 6901 **(data)**, `valid_map.json` |

## Section 5.3, validation across the box (Fig. 6)

| item | produced by | from |
|---|---|---|
| Fig. 6 and the twelve-run widths and \|z\| | `eval_valid.py` (`campaign_cpu.sh` step 2), `figs_pro.py anymap` | `valid_map.json` |
| multiplicity tails, 11 to 14 percent | `tail_stress.py` | `tail_stress.json` |
| offsets at 80k and 800k events | `floor_eval_gen.py --out output/floor_eval_final.json` (runs from `gen_floor.sh`) | `floor_eval_final.json` |
| accuracy law: 1.23 at 800k, 1.06 at 80k, 0.8 percent per bin | `accuracy_law_gen.py --out output/accuracy_law_final.json` | `accuracy_law_final.json` |

## Section 5.4, the number of terms (Fig. 7)

| item | produced by | from |
|---|---|---|
| Fig. 7, Stages A to C | the GPU farm: `farm/launch_farm.sh` running the jobs of `farm/all_jobs.txt` (`farm/r3_rank.py`, four networks, SiLU) | `rank_scan.json` |
| Fig. 7, the eight-parameter Sherpa model | `perlmutter/rank_scan.sbatch` (`STAGE=E RANK_ENS=4`), merged by `merge_rank_scan.py E` | `rank_scan.json` |
| retraining at K = K₂: 0.96, 1.20, 1.05 | `r2_ladder.py` with `LADDER_K=3, 6, 10`; mean over the held-out runs without 6704 and 6902 | `rank_production_v2.json` |
| singular values, R² 0.990 to 0.9996, correlation 0.98 | `factor_fig.py` | `factor_fig_values.json` |
| disjoint halves, correlation 0.89 | `farm/disjoint_score.py` | `disjoint_score.json` |
| `figs_pro.py head` | | `rank_scan.json` |

## Section 5.5, the wide coupling range (Fig. 8)

| item | produced by | from |
|---|---|---|
| Fig. 8, widths 2.03, 1.09, 0.96 and 6.87, effective events 13 and 56.9k | `farm/r2b_widebox_ctrl.py`, `figs_pro.py widebox` | `widebox_ctrl.json`, `wb_weights_*_0.083.npy` |

## Section 5.6, the mixture of two generators (Table 2)

| item | produced by | from |
|---|---|---|
| mixture samples | `gen_showerbox.sh`, `gen_showerbox2.sh` (Sherpa 3.1), `herwig_grid.py grid 80000 --out data_herwigbox2` (Herwig 7.2.3), `make_stageDM_data.py` | **(data)** |
| networks (two trainings) | `stageDM.py` via `perlmutter/stageDMxSc.sbatch` and `stageDMx_s10Sc.sbatch` (K = 24, 72k steps, mixture head) | **(data)** |
| temperature 1.125 | `recalibrate_T.py` (`DROP=9932`), then `apply_T.py` | `recal_T_DMDMXSc*.json` |
| Table 2, widths 1.13 and 1.09, baryon 1.49 | `ladder_widths_dedup.py MIXSTAGE` for each training, averaged | `ladder_dedup_MIXSTAGE_*.json` |
| baryon count one to three percent low | `rate_check_mix.py` | `rate_check_mix_*.json` |

## Section 5.7, larger models (Tables 3 and 7)

| item | produced by | from |
|---|---|---|
| samples | `design_stageE.py`, `design_stageF_herwig.py`, `design_stageF_augment.py`; `gen_stageE.sh`, `gen_stageF.sh` (`perlmutter/stageE.sbatch`, `stageF.sbatch`, `gen_stageF_augment.sbatch`); mixtures by `make_mixture_data.py` (`perlmutter/build_mixture.sbatch`, `mix17_v2_*.sbatch`) | **(data)** |
| networks | `perlmutter/train_stage.sbatch` with `STAGE=E`; `STAGE=F STAGEF_CSV=stageF_design_aug.csv LADDER_STEPS=72000`; `STAGE=MIX LADDER_DRIVER=stage_mixture.py` (K = 48) | **(data)** |
| Table 3 | `ladder_widths_big.py` via `perlmutter/widths_big.sbatch` | `widths_v2_*.json` |
| multiplicity remarks | `make_stage_table.py` in `perlmutter/refresh_release_v2.sbatch` | `release/closure_*.csv` |

## Section 5.8, factorized and concatenation networks (Table 4)

| item | produced by | from |
|---|---|---|
| concatenation (early fusion), Stages A to C | `concat_baseline.py` via `perlmutter/concat_small.sbatch` | `concat_baseline_{A,B,C}_prod.json` |
| concatenation, Sherpa 8 and mixture 17 | `concat_baseline.py` via `perlmutter/concat_big.sbatch` | `concat_baseline_{E,MIX}_silu.json` |
| late fusion, Stage C and mixture 17 | `concat_baseline.py` with `CONCAT_MODE=late` via `perlmutter/concat_late.sbatch` | `concat_baseline_{C,MIX}_late.json` |
| cost of the three designs | `bench_cost.py` via `perlmutter/bench_cpu.sbatch` and `bench_gpu.sbatch` | `bench_cost_*.json` |
| factorized columns | as Tables 1 and 3 | |

## Section 6, anchoring and the fit (Table 5, Fig. 9)

| item | produced by | from |
|---|---|---|
| theory targets on the (α_s, α0) grid | `anchor_targets_grid.py` with `FO_NORM=total ARES_EXT=1 A0_MIN=0.15 A0_MAX=0.85` (uses `thrust_chain.py`, `np_shift.py`, `theory_cov.py`) | `thrust_targets_grid_ext.npz` |
| profile on the grid (starting points) | `profile_column.py`, `polish_parallel.py` via `perlmutter/profile_variant.sbatch` | `profile_MIX17ext_central.json` |
| Eq. (6.8): coupling, α0, correlation, χ² | `profile_rows.py` via `perlmutter/rows_main.sbatch` (stage 1) | `profile_MIX17ext_central_rows.json` |
| perturbative and nonperturbative errors | `perlmutter/rows_main.sbatch` (stage 2) | `profile_MIX17ext_central_rows_var.json` |
| box limited to 90 percent | `perlmutter/rows_shrink.sbatch` | `profile_MIX17ext_central_rows_shrink0.90*.json` |
| product mixture, window bins only | `perlmutter/rows_variants.sbatch` | `profile_MIX17{geo,win}_central_rows*.json` |
| three-parameter family | `run_rows_local.sh output/profile_C_1Mext_central.json 4` (`COLS=1 COL_STEP=0.005 ROW_STEP=0.001`) | `profile_C_1Mext_central_rows.json` |
| pseudo-data closure, 40 sets | `pseudo_rows.py` via `perlmutter/closure_rows.sbatch`; sets 2 and 22 with `ROW_HALF=0.024` (`closure_wide.sbatch`); `pseudo_rows.py summary` | `pseudo_rows_sets/`, `pseudo_rows_*_summary.json` |
| χ² per experiment, residuals, edges, fraction | `fit_summary_direct.py` | `profile_MIX17ext_central_summary.json` |
| Table 5 and the bands of Fig. 9 | `bands_exact.py point`, `part`, `merge` via `perlmutter/bands_exact.sbatch` | `*_exchange.json`, `*_exchange_dist.npz`, `*_bands_point.json` |
| Fig. 9 (c), the two-parameter regions | `surface_rows.py` via `perlmutter/surface.sbatch` | `profile_MIX17ext_central_surface.json` |
| Fig. 9 | `fig_anchored_fit.py output/profile_MIX17ext_central.json` | the files above |
| fraction below τ = 0.05 | `peak_fraction.py` | `profile_MIX17ext_central_peak.json` |
| calculation-only window fits (Sec. 6.4) | `calc_window_fits.py` | `calc_window_fits.json` |
| band at the ALEPH pair, 0.8 percent | `band_refit_compare.py` | `band_refit_compare.json` |
| parton-level check (Sec. 6.6) | `parton_direct.py` | `profile_MIX17ext_central_check.json` |
| **every number of Sec. 6** | `make_sec6_numbers.py` | all of the above |

## Appendix D, the thrust calculation (Fig. 10)

| item | produced by | from |
|---|---|---|
| fixed order | EERAD3 1.0 with `eerad3/eerad3_blk.patch`, jobs `eerad3/nnlo.sbatch`; cumulant by `eerad_cumulant_combine.py` | `eerad_*.npy` |
| resummation | ARES with the changes of `ares_recovered/ares_edits.txt`, driven by `ares_recovered/*.wls` | `ares_recovered/ares_prod_asgrid*.csv` |
| fit to ALEPH and its refits | `thrust_chain.py` | `chain_summary.json`, `moments_joint.json` |
| χ² at the fitted pair, 13.7, and the refitted χ² range | `calc_window_fits.py` (ALEPH, own normalization) | `calc_window_fits.json` |
| cross-check with y0 = 1e-7 | `eerad_y7_check.py` | `eerad_NNLO_cum.npy`, `eerad_NNLO_ranktrim_y7.npy` |
| clipping of the cumulant, third-order subtraction | `appD_checks.py`, `NP_SUB_ORDER=3 python appD_checks.py` | `appD_checks*.json` |
| Fig. 10 and the deviations from ALEPH | `fig_thrust_vs_aleph.py` | the files above |

## Appendix E

| item | produced by | from |
|---|---|---|
| condition numbers, eigenvalues, diagonal-term scan | `floor_scan_direct.py` | `profile_MIX17ext_central_floorscan.json` |
| noise test of the diagonal term | `floor_noise_direct.py` | `profile_MIX17ext_central_floornoise.json` |

## Appendices A and B

| item | produced by | from |
|---|---|---|
| temperatures | `recalibrate_T.py`; E, F, MIX17 by `perlmutter/recal_T.sbatch` | `recal_T_*.json` |
| single networks against the average, 1.22 to 1.55 and 1.10 to 1.62 | `ensemble_ablation.py` (`LADDER_ACT=silu FINE_T=1 DROP_DUP=1`) | `ensemble_ablation_dedup.json` |
| 82457 weights | `param_count.py` | `param_count.json` |
| comparison with the energyflow PFN | `run_crosscheck.sh` (`crosscheck_ef.py` in a TensorFlow environment, `crosscheck_torch.py`) | `crosscheck_*.json` |
| SiLU against ReLU at Stage B | `r2_ladder.py` Stage B at 24k steps with `LADDER_ACT=relu` and `silu` | `probe_B_*.log` |
| exact-axis widths 0.97, 1.23, 1.00 | `ladder_widths_v2.py` (`post_AB_v2.sh`), `ladder_widths_dedup.py` on `*_ref_v2.npz` (`rebuild_ref_obs.py`) | **(data)** |
| three-parameter network on all 54 Stage C runs | `stageC_1M_train.py`; samples `pm_campaign.sh` (1M campaign), `reextract_anchor1M.sh`; closure by `package_release.py C_1M`, `make_stage_table.py` | `release/closure_C_1M.csv` |
