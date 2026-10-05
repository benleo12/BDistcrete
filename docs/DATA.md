# Data: event samples and network exports

The git repository holds the scripts, the small result files and the released models. The event
samples and the full network exports are too large for git. They are published as assets of the
GitHub release `data-v1`, cut into archives of at most 1.9 GB. Each archive unpacks into
`analysis/` with the paths the scripts use (`output/models/...`, `data_stageX/...`), so nothing
has to be moved or configured after unpacking.

You only need the archives of the items you want to check. The core set (A0 and B1 to B10,
9.6 GB) reproduces from the stored networks the closure tests of Secs. 5.2 to 5.7, every fit of
Sec. 6, and Apps. B, D and E. Of the optional archives (B11 to B13, 2.4 GB), App. A needs B12
and the cost comparison of Sec. 5.8 needs B11. The training archives (C1 to C9, 25.1 GB) are needed only to
retrain a network, which the rank scans of Sec. 5.4 and the wide coupling range of Sec. 5.5 do.

## Download

```bash
tools/get_data.sh sec6                 # the archives of one paper item, see the groups below
tools/get_data.sh B1_exports_ABC B2_held_ABC
tools/get_data.sh --list               # every archive with its size
```

`get_data.sh` downloads `SHA256SUMS` from the release, then for each archive downloads it,
checks its SHA256, unpacks it into `analysis/`, checks every unpacked file against
`analysis/data_checksums/<archive>.sha256` and deletes the download (`--keep` keeps it). An
interrupted download resumes. `--base URL` or `BDISTCRETE_DATA_URL` points it at another copy of
the archives. Without the script:

```bash
cd analysis
B=https://github.com/benleo12/BDistcrete/releases/download/data-v1
curl -LO $B/SHA256SUMS && curl -LO $B/B6_sec6_MIX17.tar
grep B6_sec6_MIX17 SHA256SUMS | shasum -a 256 -c
tar -xf B6_sec6_MIX17.tar && shasum -a 256 -c data_checksums/B6_sec6_MIX17.sha256
ln -sfn MIX17aug_ref_v2_slim.npz output/models/MIX17aug_ref.npz   # see "Reference files" below
```

| group | archives | what it reproduces |
|---|---|---|
| `sec6` | B6 | every fit of Sec. 6 and App. E, the mixture column of Table 3 (with B8 to B10) |
| `family` | B7 | the three-parameter family of Sec. 6 |
| `table1` | B1, B2 | Table 1, the rates and Fig. 5 |
| `sec5` | B1 to B4 | Secs. 5.2 and 5.3 |
| `table2` | B5 | Sec. 5.6 and Table 2 |
| `table3` | B6, B8, B9, B10 | Sec. 5.7 and Table 3 |
| `cost` | B6, B11 | the cost comparison of Sec. 5.8 |
| `appA` | B1, B7, B12 | App. A |
| `appD` | A0 | the fixed-order cumulant of App. D, rebuilt from the EERAD3 logs |
| `core` | A0, B1 to B10 | all of the above except Sec. 5.8 and App. A |
| `train-A`, `train-B`, `train-C`, `train-C1M`, `widebox`, `train-DM2`, `train-E`, `train-F`, `train-MIX17` | see "Retraining" | the training samples of one network, with its held-out runs |

## Archives

Sizes are in GB (10⁹ bytes). Run numbers name the files `particles_full_<run>.npz` (the
particles the networks read) and `shapes_run_<run>.csv` (the event shapes the closure tests
compare). The section numbers in the last column are those of `docs/REPRODUCE.md`.

### Core

| archive | size | contents | unlocks |
|---|---|---|---|
| `A0_theory_logs.tar` | 0.04 | `logs/eerad_NNLO/`, `logs/eerad_NNLO_prod4/`, `logs/eerad_NNLO_y7/` (1204 EERAD3 job logs), `logs/post_A_v2.log` | App. D: `eerad_cumulant_combine.py` rebuilds `eerad_NNLO_cum.npy`, `eerad_y7_check.py` from the logs. App. A: the Stage A exact-axis width |
| `B1_exports_ABC.tar` | 0.77 | `output/models/{A,B,C}_cond.npz` (network exports) and `{A,B,C}_ref.npz` (reference samples with particles) | with B2: Sec. 5.2 Table 1, rates, Fig. 5. Sec. 5.4 `factor_fig.py`. App. B temperatures of A, B, C and the ensemble ablation |
| `B2_held_ABC.tar` | 1.03 | held-out runs `data_stageA_full` 6200-6205, `data_stageB_full` 6700-6704, `data_stageC` 6900-6904 | with B1: as B1. Also the held runs of every Stage A, B, C and C_1M retraining |
| `B3_held_C_valid.tar` | 0.74 | `data_stageC` 6950-6961 | with B1 and B2 (`eval_valid.py` also reads the held runs 6900-6904, whose rows Figs. 5 and 6 use): Sec. 5.3 Fig. 6 widths and \|z\| (`eval_valid.py`), multiplicity tails (`tail_stress.py`) |
| `B4_held_C_floor.tar` | 1.23 | `data_stageC` 6970-6971 (800k events each) | with B1 and B2: Sec. 5.3 offsets at 80k and 800k events (`floor_eval_gen.py`), accuracy law (`accuracy_law_gen.py`) |
| `B5_mix7.tar` | 0.81 | `output/models/DMDMXSc0_cond.npz`, `DMDMXSc10_cond.npz` (the two trainings), `DMDMXSc0_ref.npz`, `data_stageDM2/meta.json` and held runs 9930-9936 | Sec. 5.6: Table 2 widths and baryon count (`ladder_widths_dedup.py MIXSTAGE`, `rate_check_mix.py`), temperature 1.125 (`recalibrate_T.py`) |
| `B6_sec6_MIX17.tar` | 0.94 | `output/models/MIX17aug_cond.npz`, `MIX17aug_ref_v2_slim.npz` | Sec. 6: every row from the grid profile to Table 5, Fig. 9, the pseudo-data closure, the parton-level check. App. E. With B8 to B10: Table 3 mixture column |
| `B7_C1M.tar` | 0.51 | `output/models/C_1M_cond.npz`, `C_1M_ref_v2_slim.npz`, `C_1M_ref_v1_slim.npz` | Sec. 6 three-parameter family (`run_rows_local.sh output/profile_C_1Mext_central.json`). With B12: App. A three-parameter closure. With B2: App. B temperature of C_1M |
| `B8_E.tar` | 0.88 | `output/models/E_cond.npz`, `E_ref_v2_slim.npz` | with B10: Sec. 5.7 Table 3 Sherpa column (`ladder_widths_big.py`), App. B temperature of E |
| `B9_F.tar` | 1.10 | `output/models/Fauglong_cond.npz`, `Fauglong_ref_v2_slim.npz` | with B10: Table 3 Herwig column, App. B temperature of F |
| `B10_held_8param.tar` | 1.51 | `data_stageE` 7200-7207 with `stageE_design.csv`, `data_stageF` 8200-8207 with `stageF_design.csv` and `stageF_design_aug.csv`, `data_stageDM17aug_v2` 9930-9939 with `meta.json` | with B6, B8, B9: Table 3, the multiplicity remarks of Sec. 5.7 (`make_stage_table.py`), App. B temperatures of E, F, MIX17 |

### Optional

| archive | size | contents | unlocks |
|---|---|---|---|
| `B11_bench.tar` | 0.67 | `output/models/MIX17aug_ref.npz` (the mixture reference with particles), `concat_{A,B,C,E,MIX}.pt`, and `concat_{C,MIX}_late.pt` if those jobs had finished at packaging | with B6: Sec. 5.8 cost of the three designs (`bench_cost.py`). The particles also let you reweight new observables on the Sec. 6 reference |
| `B12_appA_v2.tar` | 1.12 | `output/models/{A,B,C}_ref_v2_slim.npz`, `data_stageA_v2` 6200-6205, `data_stageB_v2` 6700-6704, `data_stageC_v2` 6900-6904 | with B1: App. A exact-axis widths 0.97, 1.23, 1.00 (`ladder_widths_v2.py`, `ladder_widths_dedup.py`). With B7: App. A three-parameter closure (`STAGEC_DATA=data_stageC_v2 make_stage_table.py`) |
| `B13_C1M_particles.tar.gz` | 0.63 | `output/models/C_1M_ref_v2.npz` with its particles (2.75 GB unpacked) | `package_release.py C_1M` only, which rebuilds the release bundle that `release/models/` already holds |

### Training samples

| archive | size | contents | trains |
|---|---|---|---|
| `C1_train_A.tar` | 1.02 | `data_stageA_full` 6100-6114 | the Stage A network, its rank scan, the retraining at K = 3, the disjoint halves (also needs B1), concatenation A |
| `C2_train_B.tar` | 1.72 | `data_stageB_full` 6600-6627 | Stage B, its rank scan, K = 6, concatenation B, the SiLU probe of App. B |
| `C3_train_C.tar` | 1.66 | `data_stageC` 6800-6826 | Stage C, its rank scan, K = 10, concatenation and late fusion C |
| `C4_train_C1M.tar` | 0.83 | `data_stageC` 7800-7826 (40k events each) | with C3: the C_1M network (`stageC_1M_train.py`) |
| `C5_widebox.tar` | 1.67 | `data_widebox` 9000-9020, 9100-9105 | Sec. 5.5 (`farm/r2b_widebox_ctrl.py`), the energyflow cross-check of App. B (runs 9000 and 9010) |
| `C6_train_DM2.tar` | 1.83 | `data_stageDM2/meta.json`, runs 9700-9795 | the two seven-parameter mixture trainings of Sec. 5.6 |
| `C7a` to `C7d_train_E.tar` | 1.47 to 1.48 each | `data_stageE` 7000-7023, 7024-7047, 7048-7071, 7072-7095, each with `stageE_design.csv` | the Sherpa eight-parameter network, its rank scan, concatenation E |
| `C8a` to `C8e_train_F.tar` | 1.24 to 1.54 each | `data_stageF` 8000-8020, 8021-8041, 8042-8063, 8100-8123, 8124-8147, each with both design files | the Herwig eight-parameter network (`STAGEF_CSV=stageF_design_aug.csv`) |
| `C9a`, `C9b_train_MIX17.tar` | 1.81 each | `data_stageDM17aug_v2` 9700-9795 and 9796-9891, each with `meta.json` | the seventeen-parameter mixture, concatenation and late fusion MIX |

### Retraining

A network needs its training archives and the archives with its held-out runs:

| network | archives |
|---|---|
| Stage A, B or C | C1, C2 or C3, with B2 (`train-A`, `train-B`, `train-C`) |
| C_1M | C3, C4, B2 (`train-C1M`) |
| wide coupling range | C5 (`widebox`) |
| seven-parameter mixture | C6, B5 (`train-DM2`) |
| Sherpa, Herwig eight-parameter | C7a to C7d or C8a to C8e, with B10 (`train-E`, `train-F`) |
| seventeen-parameter mixture | C9a, C9b, B10 (`train-MIX17`) |

Run `tools/get_data.sh --unlink` before retraining (see below).

### Checksums

`SHA256SUMS` in the release lists the SHA256 of every archive, and `MANIFEST.txt` the size and
SHA256 of every file inside each one. Both are written when the archives are made.

| archive | bytes | SHA256 |
|---|---|---|
| all | pending | filled in when the archives are uploaded |

## Reference files

Each network comes with the reference sample it reweights. The reference files for Secs. 5.7
and 6 and for App. A ship without their particle arrays (`*_ref_v2_slim.npz`, about 50 MB in
place of 0.6 to 2.7 GB). They hold every observable the scripts read and the charged
multiplicity `nch`, which `directlib.py` otherwise counts from the particles. Each was checked
array by array against the full file when the archive was made. `get_data.sh` then links the
names the scripts open to them:

| name the scripts read | points to | read by |
|---|---|---|
| `C_1M_ref_v2.npz` | `C_1M_ref_v2_slim.npz` | `profile_C_1Mext_central.json`, `directlib.py` |
| `C_1M_ref.npz` | `C_1M_ref_v2.npz` | `recalibrate_T.py C_1M`, `make_stage_table.py` |
| `E_ref.npz`, `Fauglong_ref.npz`, `MIX17aug_ref.npz` | the `_ref_v2_slim.npz` of each | `ladder_widths_big.py` (`perlmutter/widths_big.sbatch`), `recalibrate_T.py` |
| `{A,B,C}_ref_v2.npz` | the `_ref_v2_slim.npz` of each | `ladder_widths_v2.py`, `ladder_widths_dedup.py` |
| `DMDMXSc10_ref.npz` | `DMDMXSc0_ref.npz` | the second seven-parameter training, whose reference is byte-identical to the first |

A link is made only where no real file of that name exists, so `B11` and `B13`, which carry
full references, take precedence. The only scripts that need the particles of these references
are `bench_cost.py` (B11) and `package_release.py` (B13 for C_1M). Training writes
`output/models/<tag>_ref.npz`, which through a link would overwrite the slim file, so remove the
links first with `tools/get_data.sh --unlink`.

## Notes on exact reproduction

- The closure tests of Secs. 5.2 to 5.6 and their samples use the earlier thrust-axis finder
  (`compute_efps_old_axis.py`). Sec. 5.7, Sec. 6 and App. A use the exact axis. The archives keep
  each sample with the shapes its numbers were computed on.
- The App. B temperatures of E, F and MIX17 (`recal_T_*.json`) were selected on 28 September on
  the shape tables of the earlier axis, which the exact-axis tables replaced later that day. The
  released runs carry the exact-axis tables, so `recalibrate_T.py` on them gives the temperature
  on the exact axis. The exports carry the published values.
- The C_1M temperature (App. B) was selected on the earlier axis. To repeat that fit, point
  `C_1M_ref.npz` at the earlier-axis reference and use the held runs of B2:
  `ln -sfn C_1M_ref_v1_slim.npz analysis/output/models/C_1M_ref.npz`, then run
  `recalibrate_T.py C_1M`. Restore the link with `ln -sfn C_1M_ref_v2.npz ...` afterwards.
- The seventeen-parameter mixture was trained on `data_stageDM17aug`, whose particles, masks
  and baryon counts are bitwise those of `data_stageDM17aug_v2` (`verify_mix_v2.py`). Only the
  shapes tables differ, so the v2 runs serve both for retraining and for Table 3.
- Held run 9932 of the seven-parameter mixture and held runs 6704 and 6902 repeat a training run
  event for event. They are included because some scripts read every held run, and the closure
  scripts drop them (`DROP`).
- `concat_C.pt` holds the weights of the production Stage C concatenation run of
  `perlmutter/concat_small.sbatch` (`concat_baseline_C_prod.json`, 6902 dropped). It was written
  28 seconds before that result file.

## Regenerating the samples

Every sample can be regenerated from the run cards in the generation scripts. The run number is
the generator seed, so a run is fixed by its number and its parameter point. The HepMC files
are not kept (1.6 GB per 80k-event run). Each run is extracted to the two files above by
`extract_particles_full_flavor.py` (`extract_particles_full.py` for Stage A) and
`compute_shapes_only.py`.

| sample | generation | generator |
|---|---|---|
| `data_stageA_full`, `data_stageB_full` | `gen_stageA_full.sh`, `gen_stageB_full.sh` | Sherpa 3.0.4 |
| `data_stageC` 6800-6826, 6900-6904 | `gen_stageC.sh` | Sherpa 3.0.4 |
| `data_stageC` 6950-6961, 6970-6971 | `gen_stageC_valid.sh`, `gen_floor.sh` | Sherpa 3.0.4 |
| `data_stageC` 7800-7826 | the 1M campaign, `perlmutter/pm_campaign.sh` (40k events per point), extracted by `reextract_anchor1M.sh` | Sherpa 3.0.4 card |
| `data_widebox` | `gen_widebox.sh` | Sherpa 3.0.4 |
| `data_stage{A,B,C}_v2` | `gen_stageA_v2.sh`, `gen_stageB_v2.sh`, `gen_stageC_v2.sh` (the same runs, extracted with the exact axis) | Sherpa 3.0.4 |
| `data_stageDM2` | `gen_showerbox.sh`, `gen_showerbox2.sh`, `herwig_grid.py grid 80000 --out data_herwigbox2`, merged by `make_stageDM_data.py` | Sherpa 3.1 (development build), Herwig 7.2.3 |
| `data_stageE` | `release/generators/gen_stageE.sh` over `stageE_design.csv` | Sherpa 3.0.4 (CVMFS) |
| `data_stageF` | `release/generators/gen_stageF.sh` over `stageF_design_aug.csv` | Herwig 7.3.0 (CVMFS) |
| `data_stageDM17aug_v2` | `make_mixture_data.py` from E and F (`perlmutter/build_mixture.sbatch`, `mix17_v2_a.sbatch`, `mix17_v2_b.sbatch`) | resampled, no new events |

`release/generators/README.md` gives the container recipe, the pinned CVMFS builds and a check
that the pinned Herwig build reproduces the released events byte for byte. Four Stage B runs lost
their seeds and cannot be regenerated bit for bit. The network exports and reference files are
written by the training scripts of `docs/REPRODUCE.md`.

## Not released

- Copies that duplicate a released file: `C_1M_ref.npz` (identical to `C_1M_ref_v2.npz`),
  `DMDMXSc10_ref.npz`, and the `*_ref_v2.npz` of E, F and MIX17 (identical to their
  `*_ref.npz`).
- The full references of E and F with particles (0.66 and 0.77 GB). Only `package_release.py`
  reads their particles, and `release/models/` already holds what it makes.
- Exports and samples of earlier versions that no number of the paper uses: earlier-axis
  references (`*.v1`, except the slim C_1M copy in B7), exports at superseded temperatures, the
  exploration stages, the Herwig box with the old decay convention, and the Stage C runs
  7900-7904.
