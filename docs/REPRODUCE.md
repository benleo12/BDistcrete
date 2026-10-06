# Reproducing the paper

All commands run from `analysis/`. Result files named below are in `analysis/output/` unless marked
**(data)**, which means they need network exports or event samples from the archives of
`docs/DATA.md`. Each section names the archives it needs, by their names in `docs/DATA.md`, and the
group of `tools/get_data.sh` that downloads them, for example `../tools/get_data.sh sec6` from
`analysis/`. Section, figure and table numbers follow the paper.

## Before you start

- The networks are trained with the production settings, which `r2_ladder.py` does not use by
  default. Every training command below assumes

  ```bash
  export LADDER_ACT=silu LADDER_EMB_SCALE=auto LADDER_LR=1e-3
  ```

  The Perlmutter batch scripts set them themselves.
- The figure scripts draw their labels with LaTeX (`usetex`, with `amsmath`, `amssymb` and `bm`).
  `environment.yml` does not provide LaTeX, so install a TeX distribution first.
- `logs/` is not in git. Run `mkdir -p logs` once, because the batch scripts and
  `run_rows_local.sh` write their logs there.
- Training writes into `output/models/` under the published names and overwrites downloaded
  exports. Copy the exports elsewhere before retraining, and run `../tools/get_data.sh --unlink`
  first. Some reference names there are links to the slim reference files, and training would
  write through a link into the slim file (`docs/DATA.md`, "Reference files").
- Retrained networks are not bit for bit the published ones, because GPU training is not
  deterministic.
- Batch scripts: set the account placeholders, and check the `#SBATCH -o` and `-e` lines, because
  Slurm does not expand variables there.

## Section 5.1, the toy model

Needs nothing beyond the repository. The per-job files of the CPU jobs ship with it in
`output/_kruler/` and `output/_law/`, so the merges can be redone without rerunning the jobs.

| item | produced by | from |
|---|---|---|
| Fig. 4 (a), (b) | `python validate_pfn.py`, then `python figs_pro.py toy` | `toy_fvf_data.json` |
| Fig. 4 (c) | `python ab_rank.py`, then `python figs_pro.py toy` | `toy_ab_data.json` |
| closure widths 17.9, 2.89, 1.08, flat for K = 2 to 32 | `bash run_kscan_toy_ruler.sh` (12 CPU jobs, then `kscan_toy_ruler_merge.py`). With the per-job files in `output/_kruler/`, which ship with the repository, `python kscan_toy_ruler_merge.py` alone redoes the merge | `kscan_toy_ruler.json` |
| 1.05 ± 0.02 and 1.08 ± 0.09 | `bash run_toy_law_ruler.sh` (split and law jobs, then `toy_law_ruler_merge.py`) | `kscan_toy_law_ruler.json` |
| N* = 89k, R² = 0.99, intercept 0.86 | the law jobs above, plus three null jobs that no script runs: `for SD in 0 1 2; do python -u toy_law_ruler.py --mode null --seed $SD --out output/_law/null_s$SD.json; done`, then `python toy_law_ruler_merge.py` and `python law_released_intercept.py`. N* and R² are written only when the null files exist. The merge rewrites `kscan_toy_law_ruler.json` without the top-level key `released_intercept`, which `law_released_intercept.py` adds back: it fits χ²/ndf = c + N/N*, with the intercept c free, to the χ²/ndf values the merge stored and gives the same N* and the intercept 0.86. The merge also reads `split_scan.json` and `budget_law.json`. These and the per-job files in `output/_law/`, the null files included, ship with the repository, so the two scripts alone redo the numbers | `kscan_toy_law_ruler.json` |

## Section 5.2, the ladder (Table 1, Fig. 5)

Needs `B1_exports_ABC` and `B2_held_ABC` (group `table1`). Retraining Stage A, B or C needs
`C1_train_A`, `C2_train_B` or `C3_train_C` with `B2_held_ABC` (groups `train-A`, `train-B`,
`train-C`).

| item | produced by | from |
|---|---|---|
| Stage A network | `LADDER_STAGES=A python r2_ladder.py` (step 1a of `campaign_gpu.sh`, run alone because the rest of that script launches hours of other studies) | event samples **(data)** |
| Stage B network | `LADDER_STAGES=B LADDER_SEED_OFFSET=100 python r2_ladder.py` | event samples **(data)** |
| Stage C network | `LADDER_STAGES=C python r2_ladder.py`, then `python apply_T.py C 1.0`. The refined temperature grid of `r2_ladder.py` selects 0.95 for C, and the paper uses 1.0 | event samples **(data)** |
| Table 1, Stage A | `r2_ladder.py`. Check from the export without training: `python ladder_widths_dedup.py A output/models/A_ref.npz` (0.953, writes `ladder_dedup_A_A_ref_all.json`) | `ladder_final.json` |
| Table 1, Stages B and C, and the per-observable ranges | `python ladder_widths_dedup.py B output/models/B_ref.npz` and `python ladder_widths_dedup.py C output/models/C_ref.npz`. Held-out runs 6704 and 6902 are dropped by default, they repeat training runs | `ladder_dedup_B_B_ref.json`, `ladder_dedup_C_C_ref.json`, exports **(data)** |
| mean baryon and strange rates, \|z\| and percent | `python rate_check_dedup.py B` and `C` (prints only) | exports **(data)** |
| energy-fraction width 0.85, same-run 0.91 ± 0.13 | mean over runs 6900, 6901, 6903, 6904 of `valid_map.json` (`eval_valid.py`) and of the three split widths per run in `xp_samerun.json` (`audit_xp_samerun.py`). No script takes this four-run mean | included |
| Fig. 5 | `python figs_pro.py hero` | `C_cond.npz`, `C_ref.npz`, run 6901 **(data)**, `valid_map.json` |

## Section 5.3, validation across the box (Fig. 6)

Needs `B1_exports_ABC` and `B2_held_ABC`, with `B3_held_C_valid` for Fig. 6 and the tails and
`B4_held_C_floor` for the offsets and the accuracy law (the four together are group `sec5`).
`B2_held_ABC` is needed because `eval_valid.py` also evaluates held runs 6900 to 6904, which Figs. 5
and 6 read from `valid_map.json`, and because the offsets and the accuracy law compare with held
runs 6901 and 6902.

| item | produced by | from |
|---|---|---|
| Fig. 6 and the twelve-run widths and \|z\| | `python eval_valid.py` (step 2 of `campaign_cpu.sh`), `python figs_pro.py anymap` | `valid_map.json` |
| multiplicity tails, 11 to 14 percent | `python tail_stress.py` | `tail_stress.json` |
| offsets at 80k and 800k events | runs 6970 and 6971 from `SHERPA3_PREFIX=<Sherpa 3 install> bash gen_floor.sh` (or `B4_held_C_floor`), then `python floor_eval_gen.py --out output/floor_eval_final.json --force`. Without `--force` the script stops because the file is shipped. Give another `--out` to keep the shipped file for comparison | `floor_eval_final.json` |
| accuracy law: 1.23 at 800k, 1.06 at 80k, 0.8 percent per bin | `python accuracy_law_gen.py --out output/accuracy_law_final.json --force` (same remark on `--force`) | `accuracy_law_final.json` |

## Section 5.4, the number of terms (Fig. 7)

The scans and retrainings need the training archives with their held-out runs: groups `train-A`,
`train-B` and `train-C` for Stages A to C, and `train-E` for the Sherpa model.

| item | produced by | from |
|---|---|---|
| Fig. 7, Stages A to C | the GPU farm: `farm/launch_farm.sh` running the jobs of `farm/all_jobs.txt` (`farm/r3_rank.py`, four networks, SiLU). Without the farm, for each stage `<S>` and each `<K>` in 1, 2, 3, 4, 6, 8, 16: `RANK_STAGES=<S> RANK_KS=<K> RANK_ENS=4 RANK_OUT=output/rank_scan_<S>_K<K>.json python r3_rank.py`, then `python merge_rank_scan.py A B C` | `rank_scan.json` |
| Fig. 7, the eight-parameter Sherpa model | `export STAGE=E RANK_ENS=4 KLIST=1,2,4,6,8,9,10,12,16,24,32,45,48`, then `sbatch --array=0-12 perlmutter/rank_scan.sbatch` (the script defaults to two networks, so `RANK_ENS=4` must be set), then `RANK_KS=$KLIST python merge_rank_scan.py E`. The per-K files of the published scan ship with the repository in `output/ens4/`, so `RANK_DIR=output/ens4 RANK_KS=$KLIST python merge_rank_scan.py E` redoes the merge without GPUs | `rank_scan.json` |
| retraining at K = K₂: 0.96, 1.20, 1.05 | farm jobs `k3_A`, `k6_B`, `k10_C`. By hand: `LADDER_STAGES=A LADDER_K=3 LADDER_SUFFIX=_k3 LADDER_OUT=output/rank_v2_A.json python farm/r2_ladder.py`, the same for B (`LADDER_K=6 LADDER_SEED_OFFSET=100 LADDER_SUFFIX=_k6`) and C (`LADDER_K=10 LADDER_SUFFIX=_k10`). The numbers are the means over the held-out runs without 6704 and 6902 | `rank_production_v2.json` |
| singular values, R² 0.990 to 0.9996, correlation 0.98 | `python factor_fig.py` (needs `B1_exports_ABC`) | `factor_fig_values.json` |
| disjoint halves, correlation 0.89 | `python farm/disjoint_score.py` (needs `C1_train_A`, and `B1_exports_ABC` for the reference events it compares on) | `disjoint_score.json` |
| the figure | `python figs_pro.py head` | `rank_scan.json` |

## Section 5.5, the wide coupling range (Fig. 8)

| item | produced by | from |
|---|---|---|
| Fig. 8, widths 2.03, 1.09, 0.96 and 6.87, effective events 13 and 56.9k | `python farm/r2b_widebox_ctrl.py` (needs `C5_widebox`, group `widebox`), `python figs_pro.py widebox` | `widebox_ctrl.json`, `wb_weights_*_0.083.npy` |

## Section 5.6, the mixture of two generators (Table 2)

Needs `B5_mix7` (group `table2`). Retraining needs `C6_train_DM2` with `B5_mix7` (group
`train-DM2`). `B5_mix7` ships the reference once, because the two trainings share it. The scripts
below name their outputs after the reference file, so they need the second name
`output/models/DMDMXSc10_ref.npz`. `tools/get_data.sh` makes this link. Only after a manual
download do you make it yourself: `ln -sfn DMDMXSc0_ref.npz output/models/DMDMXSc10_ref.npz`.

| item | produced by | from |
|---|---|---|
| mixture samples | `gen_showerbox.sh`, `gen_showerbox2.sh` (Sherpa 3.1), `herwig_grid.py grid 80000 --out data_herwigbox2` (Herwig 7.2.3), then `DM_H_SRC=data_herwigbox2 DM_DST=data_stageDM2 python make_stageDM_data.py`. The merge of each pair of 40k-event Sherpa runs into one 80k run (9300 to 9331) and the merge of the Herwig slice `meta.json` files are not scripted | **(data)** |
| networks (two trainings) | `stageDM.py` via `sbatch perlmutter/stageDMxSc.sbatch` and `sbatch perlmutter/stageDMx_s10Sc.sbatch` (K = 24, 72k steps, mixture head) | **(data)** |
| temperature 1.125 | `DM_DATA=data_stageDM2 DROP=9932 python recalibrate_T.py DMDMXSc0`, then `python apply_T.py DMDMXSc0`, and the same for `DMDMXSc10` | `recal_T_DMDMXSc*.json` |
| Table 2, widths 1.13 and 1.09, baryon 1.49 | `EXPORT=output/models/DMDMXSc0_cond.npz python ladder_widths_dedup.py MIXSTAGE output/models/DMDMXSc0_ref.npz`, the same for `DMDMXSc10`, averaged. Run after `apply_T.py`, because the temperature is read from the export | `ladder_dedup_MIXSTAGE_*.json` |
| baryon count one to three percent low | `EXPORT=output/models/DMDMXSc0_cond.npz python rate_check_mix.py output/models/DMDMXSc0_ref.npz`, the same for `DMDMXSc10` | `rate_check_mix_*.json` |

## Section 5.7, larger models (Tables 3 and 7)

Needs `B6_sec6_MIX17`, `B8_E`, `B9_F` and `B10_held_8param` (group `table3`). Retraining needs
`C7a` to `C7d_train_E` (Sherpa), `C8a` to `C8e_train_F` (Herwig) or `C9a` and `C9b_train_MIX17`
(mixture), each with `B10_held_8param` (groups `train-E`, `train-F`, `train-MIX17`). The training
and closure scripts look for `stageE_design.csv` and `stageF_design_aug.csv` inside `data_stageE/`
and `data_stageF/` (the archives carry them there) or in `analysis/`. Copies are in
`release/generators/`.

| item | produced by | from |
|---|---|---|
| samples | `design_stageE.py`, `design_stageF_herwig.py`, `design_stageF_augment.py`, then `perlmutter/gen_stageE.sh`, `release/generators/gen_stageF.sh` (`perlmutter/stageE.sbatch`, `stageF.sbatch`, `gen_stageF_augment.sbatch`), and the mixtures by `make_mixture_data.py` (`perlmutter/build_mixture.sbatch`, `mix17_v2_*.sbatch`) | **(data)** |
| Table 7 | the per-axis minimum and maximum of the design CSVs, also in `release/STAGES.md` | `release/generators/*.csv` |
| Sherpa network (E) | `STAGE=E sbatch perlmutter/train_stage.sbatch` | **(data)** |
| Herwig network (F, trained as `Fauglong`) | `STAGE=F STAGEF_CSV=stageF_design_aug.csv LADDER_SUFFIX=auglong LADDER_STEPS=72000 LADDER_OUT=output/ladder_stageFauglong.json sbatch -t 02:30:00 perlmutter/train_stage.sbatch` | **(data)** |
| mixture network (MIX17, trained as `MIX17aug`) | `STAGE=MIX DM_DATA=data_stageDM17aug_v2 DM_TAG=MIX17aug LADDER_DRIVER=stage_mixture.py STAGED_OUT=output/stage_mixture_MIX17aug.json LADDER_OUT=output/ladder_MIX17aug.json sbatch perlmutter/train_stage.sbatch` (K = 48 from the batch script). The published training read `data_stageDM17aug`, which has the same particles and the earlier-axis shapes | **(data)** |
| Table 3 | `python ladder_widths_big.py E output/models/E_cond.npz output/models/E_ref_v2_slim.npz`, `STAGEF_CSV=stageF_design_aug.csv python ladder_widths_big.py F output/models/Fauglong_cond.npz output/models/Fauglong_ref_v2_slim.npz`, `DM_DATA=data_stageDM17aug_v2 python ladder_widths_big.py MIX output/models/MIX17aug_cond.npz output/models/MIX17aug_ref_v2_slim.npz` (as in `perlmutter/widths_big.sbatch`, which passes `E_ref.npz`, `Fauglong_ref.npz` and `MIX17aug_ref.npz`, the names `tools/get_data.sh` links to the same slim references) | `widths_v2_*.json` |
| multiplicity remarks | `make_stage_table.py`, as in `perlmutter/refresh_release_v2.sbatch`. To leave the shipped `release/` alone, copy it and run `RELEASE_DIR=<copy> STAGE_TABLE_SUBSET=1 STAGEF_CSV=stageF_design_aug.csv DM_DATA=data_stageDM17aug_v2 python make_stage_table.py E Fauglong=F MIX17aug=MIX17`. It reads `output/models/<tag>_ref.npz`, and `tools/get_data.sh` links `E_ref.npz`, `Fauglong_ref.npz` and `MIX17aug_ref.npz` to the slim references. After a manual download make the links yourself, for example `ln -sfn E_ref_v2_slim.npz output/models/E_ref.npz`, and the same for `Fauglong` and `MIX17aug` | `release/closure_*.csv` |

## Section 5.8, factorized and concatenation networks (Table 4)

| item | produced by | from |
|---|---|---|
| concatenation (early fusion), Stages A to C | `for S in A B C; do STAGE=$S sbatch perlmutter/concat_small.sbatch; done` (needs `C1_train_A` to `C3_train_C` and the held runs of `B2_held_ABC`) | `concat_baseline_{A,B,C}_prod.json` |
| concatenation, Sherpa 8 and mixture 17 | `STAGE=E sbatch perlmutter/concat_big.sbatch` and `STAGE=MIX sbatch perlmutter/concat_big.sbatch` (needs `C7a` to `C7d_train_E`, `C9a` and `C9b_train_MIX17`, and `B10_held_8param`) | `concat_baseline_{E,MIX}_silu.json` |
| factorized columns | as Tables 1 and 3 | |
| truncation test: the concatenation output keeps its closure within 0.02 at r = 3, 5, 5, 9, 16 singular components for d = 1, 2, 3, 8, 17 | `sbatch perlmutter/dctr_truncation.sbatch` (`dctr_truncation.py <tag> <weights>`, which reloads the trained concatenation networks and recomputes the closure from the truncated output with the published evaluation) | `dctr_truncation_{A,B,C,E,MIX}_concat_*.json` |
| late fusion (θ after the sum), Stage C and mixture 17: 0.98 and 1.26 | `STAGE=C sbatch perlmutter/concat_late.sbatch` and `STAGE=MIX sbatch perlmutter/concat_late.sbatch` (`concat_baseline.py` with `CONCAT_MODE=late`) | `concat_baseline_{C,MIX}_late.json` |
| Fig. 9, the cost of the three networks: 0.20 s, 17 s and 20 min per step on one CPU process with four threads, 0.50 s for the reweighting and χ², memory 0.9, 4.7 and 1.7 GB | `sbatch perlmutter/bench_cpu.sbatch` and `sbatch perlmutter/bench_gpu.sbatch` (`bench_cost.py`, needs `B6_sec6_MIX17` and `B11_bench`, group `cost`), then `python fig_cost.py` | `output/bench/bench_cost_{cpu_4,cpu_6,cpu_32,cpu_32_full,cpu_128,cuda_32}.json` |
| training-seed spreads and the mixture-head control. Done for the seventeen-parameter mixture: factorized 1.09 to 1.13 over five trainings, concatenation with the mixture form 1.11 to 1.14 over three; the other stages are still running | `TASKS="design:stage:seed ..." sbatch perlmutter/seedstudy.sbatch`, at most four tasks per job, one per GPU. Designs: `fact` (factorized, stages A, B, C, E, MIX), `concat`, `late`, and for MIX only `mixe` and `mixl` (early and late fusion with the exact mixture head, `CONCAT_HEAD=mixture`). Example: `TASKS="mixe:MIX:0 mixl:MIX:0"`. A seed n trains the four networks with seeds n to n + 3, so seeds that are multiples of 4 give members independent of the published ones (0 to 3, and 100 to 103 for the factorized Stage B, which `fact:B:0` uses). `SMOKE=1` runs tiny budgets for timing. No published file is touched | `output/seedstudy/<design>_<stage>_s<seed>.json`, factorized closures in `ladder_dedup_<S>_<S>_seed<n>_ref.json` and `widths_v2_{E,MIX17aug}_seed<n>.json` |

## Section 6, anchoring and the fit (Table 5, Fig. 10)

Needs `B6_sec6_MIX17` for the MIX17 rows (group `sec6`) and `B7_C1M` for the three-parameter
family (group `family`). The three-parameter fit reads `output/models/C_1M_ref_v2.npz`, which
`tools/get_data.sh` links to `C_1M_ref_v2_slim.npz`. After a manual download make the link with
`ln -sfn C_1M_ref_v2_slim.npz output/models/C_1M_ref_v2.npz`. The theory targets, the
calculation-only fits and `make_sec6_numbers.py` need no archives. The grid profiles and
`thrust_anchor_targets_cond.npz` ship with the repository in `output/`.

| item | produced by | from |
|---|---|---|
| theory targets on the (α_s, α0) grid | `FO_NORM=total ARES_EXT=1 A0_MIN=0.15 A0_MAX=0.85 TARGETS_OUT=output/thrust_targets_grid_ext.npz python anchor_targets_grid.py` (uses `thrust_chain.py`, `np_shift.py`, `theory_cov.py`, and reads `output/thrust_anchor_targets_cond.npz`, which ships with the repository). Without `TARGETS_OUT` the grid goes to `thrust_targets_grid.npz`, which nothing reads | `thrust_targets_grid_ext.npz` |
| profile on the grid (starting points) | `TAG=MIX17ext MODEL_TAG=MIX17aug A0_LIST='0.25 0.27 ... 0.79' sbatch perlmutter/profile_variant.sbatch` (`profile_column.py`, `polish_parallel.py`). The stored file was built in stages (first 21 columns on a 9-coupling grid, then extended to 0.136), so a fresh run covers more coupling rows and is an independent recomputation, not a bit-for-bit one | `profile_MIX17ext_central.json` |
| Eq. (6.8): coupling, α0, correlation, χ² | `profile_rows.py` via `sbatch perlmutter/rows_main.sbatch` (stage 1). On a laptop: `COLS=1 COL_STEP=0.005 ROW_STEP=0.001 ./run_rows_local.sh output/profile_MIX17ext_central.json 4` | `profile_MIX17ext_central_rows.json` |
| perturbative and nonperturbative errors | `perlmutter/rows_main.sbatch` (stage 2) | `profile_MIX17ext_central_rows_var.json` |
| box limited to 90 percent | `sbatch perlmutter/rows_shrink.sbatch`, then `cp output/profile_MIX17ext_central_rows_shrink0.90sep.json output/profile_MIX17ext_central_rows_shrink0.90.json`, the name `make_sec6_numbers.py` reads. Stage 3 of `rows_main.sbatch` does the same and is superseded | `profile_MIX17ext_central_rows_shrink0.90*.json` |
| product mixture, window bins only | grid profiles with `TAG=MIX17geo MODEL_TAG=MIX17aug MIX_FORM=geometric A0_LIST='0.25 0.27 ... 0.75' sbatch perlmutter/profile_variant.sbatch` and `TAG=MIX17win MODEL_TAG=MIX17aug LAST_BIN=0.3334 USE_NCH=0` (the stored grid profiles `profile_MIX17geo_central.json` and `profile_MIX17win_central.json` ship with the repository), then `sbatch perlmutter/rows_variants.sbatch` and `for V in geo win; do cp output/profile_MIX17${V}_central_rows_sep.json output/profile_MIX17${V}_central_rows.json; done` | `profile_MIX17{geo,win}_central_rows*.json` |
| three-parameter family | `COLS=1 COL_STEP=0.005 ROW_STEP=0.001 ./run_rows_local.sh output/profile_C_1Mext_central.json 4` | `profile_C_1Mext_central_rows.json` |
| pseudo-data closure, 40 sets | `FIT=output/profile_MIX17ext_central.json sbatch perlmutter/closure_rows.sbatch` (`pseudo_rows.py`). Sets 2 and 22 again with longer rows: `SETS='2 22' sbatch perlmutter/closure_wide.sbatch`. Both scripts write the sets to `output/pseudo_rows_sets/`, run `pseudo_rows.py summary` on them and copy the summary to `output/`. In the summary a wide set replaces the original one. Without the two wide sets `make_sec6_numbers.py` stops | `pseudo_rows_sets/`, `pseudo_rows_*_summary.json` |
| χ² per experiment, residuals, edges, fraction | `python fit_summary_direct.py output/profile_MIX17ext_central.json` | `profile_MIX17ext_central_summary.json` |
| pseudo-data at the fitted point, 40 sets: the experimental errors 0.0052 and 0.062 with correlation +0.92 | `FIT=output/profile_MIX17ext_central.json sbatch perlmutter/closure_atfit.sbatch` (`pseudo_rows_atfit.py`, every truth at the fitted point) | `output/pseudo_atfit_sets/` |
| Table 5 and the bands of Fig. 10 | `FIT=output/profile_MIX17ext_central.json sbatch perlmutter/bands_exact.sbatch` (`bands_exact.py point`, `part`, `merge`). Rerun the point mode whenever the rows change | `*_exchange.json`, `*_exchange_dist.npz`, `*_bands_point.json` |
| Fig. 10 (c), the contour where the χ² of the fit rises by one | `FIT=output/profile_MIX17ext_central.json sbatch perlmutter/surface.sbatch` (`surface_rows.py`) | `profile_MIX17ext_central_surface.json` |
| Fig. 10 | `python fig_anchored_fit.py output/profile_MIX17ext_central.json` | the files above and `output/pseudo_atfit_sets/` |
| fraction below τ = 0.05 | `python peak_fraction.py output/profile_MIX17ext_central.json` | `profile_MIX17ext_central_peak.json` |
| calculation-only window fits (Sec. 6.4) | `python calc_window_fits.py` | `calc_window_fits.json` |
| band at the ALEPH pair, 0.8 percent | `python band_refit_compare.py` | `band_refit_compare.json` |
| parton-level check (Sec. 6.6) | `python parton_direct.py output/profile_MIX17ext_central.json` | `profile_MIX17ext_central_check.json` |
| **every number of Sec. 6** | `python make_sec6_numbers.py` | all of the above |

## Appendix D, the thrust calculation (Fig. 11)

| item | produced by | from |
|---|---|---|
| fixed order | EERAD3 1.0 with `eerad3/eerad3_blk.patch`. The production jobs ship with the repository in `eerad3/`, and the raw logs are in `A0_theory_logs` (group `appD`). Cumulant: `python eerad_cumulant_combine.py --dirs logs/eerad_NNLO logs/eerad_NNLO_prod4 --trim 0.10 --out output/eerad_NNLO_cum.npy` | `eerad_*.npy` |
| resummation | ARES is not redistributed (GPL-3). Clone it from upstream, https://github.com/lcarpino/ARES, at commit e75c88c and apply the three edits that `ares_recovered/ares_edits.txt` describes. The drivers load ARES from `ares_recovered/ARES_sym`, so build it there: `cd ares_recovered && git clone https://github.com/lcarpino/ARES ARES_sym && cd ARES_sym && git checkout e75c88c`, then run the edit commands listed in `ares_edits.txt` inside `ARES_sym/`. Back in `analysis/`: `wolframscript -file ares_recovered/ares_prod_asgrid_ext.wls` and `ares_prod_asgrid.wls`. Needs Wolfram Engine | `ares_recovered/ares_prod_asgrid*.csv` |
| fit to ALEPH and its refits | `ARES_EXT=1 python thrust_chain.py` | `chain_summary.json`, `moments_joint.json` |
| χ² at the fitted pair, 13.7, and the refitted χ² range | `python calc_window_fits.py` (ALEPH, own normalization) | `calc_window_fits.json` |
| cross-check with y0 = 1e-7 | `python eerad_y7_check.py`. The cumulant it compares with is rebuilt by `python eerad_cumulant_combine.py --dirs logs/eerad_NNLO_y7 --trim 0.10 --out output/eerad_NNLO_ranktrim_y7.npy` | `eerad_NNLO_cum.npy`, `eerad_NNLO_ranktrim_y7.npy` |
| clipping of the cumulant, third-order subtraction | `python appD_checks.py`, `NP_SUB_ORDER=3 python appD_checks.py`, both with `ARES_EXT` unset | `appD_checks*.json` |
| Fig. 11 and the deviations from ALEPH | `python fig_thrust_vs_aleph.py` | the files above |

## Appendix E

Needs `B6_sec6_MIX17` (group `sec6`).

| item | produced by | from |
|---|---|---|
| condition numbers, eigenvalues, diagonal-term scan | `python floor_scan_direct.py output/profile_MIX17ext_central.json` | `profile_MIX17ext_central_floorscan.json` |
| noise test of the diagonal term | `python floor_noise_direct.py output/profile_MIX17ext_central.json` | `profile_MIX17ext_central_floornoise.json` |

## Appendices A and B

| item | produced by | from |
|---|---|---|
| temperatures, Stages A to C and the 54-run network | `python recalibrate_T.py A`, `B`, `C` (`B1_exports_ABC`, `B2_held_ABC`) and `python recalibrate_T.py C_1M` (`B7_C1M`, `B2_held_ABC`), then `python apply_T.py C_1M`. For C_1M the script reads `output/models/C_1M_ref.npz`, which `tools/get_data.sh` links to `C_1M_ref_v2.npz`, the exact-axis reference (itself a link to `C_1M_ref_v2_slim.npz`). The published C_1M temperature was selected on the earlier axis, with the held runs of `B2_held_ABC`. To repeat that fit, point the name at the earlier-axis reference of `B7_C1M` first, `ln -sfn C_1M_ref_v1_slim.npz output/models/C_1M_ref.npz`, then run `python recalibrate_T.py C_1M`, and restore the link afterwards with `ln -sfn C_1M_ref_v2.npz output/models/C_1M_ref.npz` | `recal_T_{A,B,C,C_1M}.json` |
| temperatures, E, F and MIX17 | `sbatch perlmutter/recal_T.sbatch` (`recalibrate_T.py E`, `Fauglong`, and `MIX17aug` with `DM_DATA`), then `python apply_T.py Fauglong` and `python apply_T.py MIX17aug` (`B6_sec6_MIX17`, `B8_E`, `B9_F`, `B10_held_8param`, group `table3`). The script reads `output/models/<tag>_ref.npz`, which `tools/get_data.sh` links to the slim references (by hand after a manual download, as in Sec. 5.7). With the release data use `DM_DATA=data_stageDM17aug_v2`. The published fit used the earlier-axis shapes, so a rerun can differ slightly | `recal_T_{E,Fauglong,MIX17aug}.json` |
| single networks against the average, 1.22 to 1.55 and 1.10 to 1.62 | `LADDER_ACT=silu FINE_T=1 DROP_DUP=1 ABLATION_OUT=output/ensemble_ablation_dedup.json python ensemble_ablation.py` (`B1_exports_ABC`, `B2_held_ABC`). Without `ABLATION_OUT` it writes `ensemble_ablation.json` | `ensemble_ablation_dedup.json` |
| 82457 weights | `python param_count.py` | `param_count.json` |
| comparison with the energyflow PFN | `EF_PYTHON=<python of the bdistcrete-tf environment> bash run_crosscheck.sh` (`crosscheck_ef.py`, `crosscheck_torch.py`) on runs 9000 and 9010 of `data_widebox` (`C5_widebox`). The paper's 0.0004 came from a Stage D sample that no longer exists. The rerun on `data_widebox` is in progress | `crosscheck_*.json` |
| SiLU against ReLU at Stage B | `LADDER_ACT=relu LADDER_LR=1e-3 LADDER_STEPS=24000 LADDER_SUFFIX=_probe_relu LADDER_OUT=output/probe_B_relu.json python -u r2_ladder.py B > output/probe_B_relu.log`, and the same with `silu` and `_probe_silu`, with `LADDER_EMB_SCALE` unset (`C2_train_B` with `B2_held_ABC`, group `train-B`) | `probe_B_*.log` |
| exact-axis widths 0.97, 1.23, 1.00 | A: `STAGEA_DATA=data_stageA_v2 python ladder_widths_v2.py A output/models/A_ref_v2.npz` (prints 0.974, also in `output/post_A_v2.log`, which ships with the repository). B: `STAGEB_DATA=data_stageB_v2 python ladder_widths_dedup.py B output/models/B_ref_v2.npz`. C: `STAGEC_DATA=data_stageC_v2 python ladder_widths_dedup.py C output/models/C_ref_v2.npz`. `tools/get_data.sh` links `{A,B,C}_ref_v2.npz` to the slim references of `B12_appA_v2` (by hand: `ln -sfn A_ref_v2_slim.npz output/models/A_ref_v2.npz`, and the same for B and C). The references are rebuilt by `rebuild_ref_obs.py` (`post_AB_v2.sh`) | `B1_exports_ABC`, `B12_appA_v2` **(data)** |
| three-parameter network on all 54 Stage C runs | samples by `perlmutter/pm_campaign.sh` and `perlmutter/anchor1M_submit.sh` (1M campaign) and `reextract_anchor1M.sh`. The staging of these runs into `data_stageC` as 7800 to 7826 is not scripted. Training: `python stageC_1M_train.py` (`C3_train_C`, `C4_train_C1M`, `B2_held_ABC`, group `train-C1M`), then `python recalibrate_T.py C_1M` and `python apply_T.py C_1M` | **(data)** |
| closure of the 54-run network | copy `release/` and run `RELEASE_DIR=<copy> STAGEC_DATA=data_stageC_v2 python make_stage_table.py` (`B7_C1M`, `B12_appA_v2`, in group `appA`). It reads `output/models/C_1M_ref.npz`, which `tools/get_data.sh` links to `C_1M_ref_v2.npz` and that to `C_1M_ref_v2_slim.npz`. After a manual download make both links: `ln -sfn C_1M_ref_v2_slim.npz output/models/C_1M_ref_v2.npz` and `ln -sfn C_1M_ref_v2.npz output/models/C_1M_ref.npz`. `STAGEC_DATA=data_stageC_v2` is required, because the shipped table was made on those runs. After a retrain, `RELEASE_DIR=<copy> python package_release.py C_1M` first rebuilds the released head. It reads the particles of the reference, which for the published network are in `B13_C1M_particles` | `release/closure_C_1M.csv` |
