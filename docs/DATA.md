# Data: the reference sample, network exports and event samples

The git repository holds the scripts, small inputs, the released networks in `analysis/release/`,
the result files of the paper's run in `analysis/output/`, and the reference sample of the mixture
with the paper's thrust calculation in `deposit/`. The event samples, the full network exports and
the large npz files of the run are too large for git. This file lists those that the paper needs,
and what reads each one. Paths are relative to `analysis/`, except those that start with `gen/` or
`deposit/`, which are relative to the top of the repository.

## In the repository: the reference sample of the mixture and the thrust calculation

| file | bytes | what | read by |
|---|---|---|---|
| `reference_MIX17.npz` | 865,346,461 | the 1,149,840 reference events of the seventeen-parameter mixture, released as `MIX17`: the event vectors its head reads, the per-event observables without the photons radiated by the beams, and the run and the generator of every event | in `analysis/release`: `examples/reproduce_paper.py` with `--reference`, `generator_band.py` and `learning_band.py` with `--reference-file`, `tests/test_mixture.py`, `tests/compare_directlib.py`, and the code of `IMPOSE.md` and `REFERENCE.md` |
| `targets_thrust_nlo.npz` | 30,098 | the NNLL+NLO thrust calculation the paper imposes: its 14 moments in the window 0.05 ≤ τ < 1/3, their covariance and its variations | `gentune.maxent.load_targets` |

Both are in `deposit/`, with `build_reference.json`, the record of the build of the reference
sample, and `SHA256SUMS`, the sha256 of the three files. With the reference sample, a clone of the
repository downloads about 1 GB. GitHub refuses files over 100 MB, so the sample is stored in 19
parts. `analysis/release/REFERENCE.md` describes the files. With numpy alone, and without any of
the files below, they reproduce the paper's central predictions with their bands
(`analysis/release/README.md`). To join the parts, copy the targets file and check both, from the
top of a checkout:

```bash
cd analysis/release
cat ../../deposit/reference_MIX17.npz.part-* > reference_MIX17.npz
cp ../../deposit/targets_thrust_nlo.npz .
grep -v build_reference ../../deposit/SHA256SUMS | shasum -a 256 -c
```

`shasum` must print OK for both files (on Linux, write `sha256sum -c` in place of
`shasum -a 256 -c`). `SHA256SUMS` also lists `build_reference.json`, which stays in `deposit/`, so
`grep -v` leaves its line out. The two files are then in `analysis/release`, where the commands
look for them, and git ignores them. The joined file takes another 865 MB of disk. If a line says
FAILED, do not use that file: run the last three lines again, and if it fails again, clone the
repository again. Then, still in `analysis/release`:

```bash
python examples/reproduce_paper.py --reference reference_MIX17.npz --targets targets_thrust_nlo.npz
```

## On request: the network exports and event samples

The files below are not public yet. They will be archived at Zenodo with the paper, and until then
they are available on request from the authors. They are needed to redo the result files of the
paper's scripts (`docs/REPRODUCE.md`).

The run behind the paper's numbers read them on the Symmetry cluster at Perimeter, with two changes
it made itself. It fitted the calibration constant of every network again
(`output/temperature/<tag>.json`) and wrote the new value into the exports it read. And it read
every event shape, and the per-event observables of every reference sample, without the photons
radiated by the beams: from copies of the data folders, `<folder>_nb`, and from refreshed reference
samples, each listed with its sha256 in `output/bundle_rule.json`. The particle files, which the
networks read, are the same in both. `docs/REPRODUCE.md` says how the copies were made. Only
`trunc_frac.py` reads the original shape files, because it counts the particles the networks read,
the beam photons included.

### Network exports

Each model has these files in `output/models/`: `<tag>_cond.npz` (the parameter networks, the
calibration constant and the stored event functions a(Φ) of the reference events),
`<tag>_trunk.pt` (the saved members), `<tag>_fitsplit.npz` (the events of every run set aside from
training) and its reference sample `<tag>_ref.npz`, with the particles of every event. The mixture
also has `<tag>_ref_v2_slim.npz`, the per-event observables of its reference events, which
`rebuild_ref_lowmem.py` makes from `<tag>_ref.npz`.

| files | what | read by |
|---|---|---|
| `output/models/MIXGEOV3S9_*` | the seventeen-parameter mixture, first ensemble, released as `MIX17` | `three_bands.py`, `theory_variations.py`, `band_checks.py`, `envelope_test.py`, `check_linear_mix.py`, `wifi_fit.py`, `wifi_diag.py`, `wifi_coverage.py`, `fig_closure_mix.py`, `bench_cost_final.py`, `package_release.py`, `release/tools_build_reference.py` |
| `output/models/MIXGEOV3S9_ref.npz` | its reference sample with the particles of every event | `bench_cost_final.py`, `package_release.py`, `release/tools_build_reference.py` |
| `output/models/MIXGEOV3S9s1_*`, `output/models/MIXGEOV3S9s2_*` | the second and third ensembles of the mixture | `three_bands.py`, `wifi_*.py`, the strict test of Table 1 |
| `output/wifi_<tag>_basis_rebuilt.npz`, for `MIXGEOV3S9`, `MIXGEOV3S9s1`, `MIXGEOV3S9s2` and `MIXGEOV3M8S9` | the basis functions of the learning band on the events set aside for the fit and on the held-out runs, with the held-out observables without the beam photons, written by `wifi_embed.py` in the jobs `lmx_wifi*`, starting from the basis that the network's training wrote (`docs/REPRODUCE.md`, Trained models), which the run reads as `output/published/wifi_<tag>_basis.npz` | `wifi_fit.py`, `wifi_diag.py`, `wifi_gensplit.py` |
| `output/models/E_P2v3fullfac0_*`, `output/models/F_P2v3fullfac0_*` | Stage D (Sherpa) and Stage E (Herwig), first ensemble, released as `E` and `F` | `package_release.py`, `make_stage_table.py`, `precision_gap.py`, `closure_vs_distance.py` |
| `output/models/C_P2augfullfac0_*` | Stage C, first ensemble | `eval_valid.py` (Fig. 4), `single_members.py` |

The other models of Table 1 and of the learning curve have the same files under their own tags
(`docs/REPRODUCE.md`, Trained models).

### Event samples

Each run is two files, `particles_full_<run>.npz` (the particles the networks read) and
`shapes_run_<run>.csv` (the event shapes the closure tests compare), and every run has
8 × 10^4 events unless stated.

| folder | what | read by |
|---|---|---|
| `data_stagePURE17_v3/` | the 720 training runs of Stages D and E (352 Sherpa, 368 Herwig) and their 48 held-out points, with `meta.json` | the training of Stages D and E and of the mixture, the standard test, `fig_closure_mix.py`, `trunc_frac.py` |
| `data_stagePURE17_v3_h400/` | the same with every held-out run at 4 × 10^5 events, the run and its four extensions | the strict test, `wifi_coverage.py`, `make_stage_table.py`, `closure_vs_distance.py` |
| `gen/data_abc/` | Stages A to C: the training runs of the designs `gen/designs/stage{A,B,C}_design_aug.csv` and their 16 held-out runs | the training of Stages A to C, `eval_valid.py`, `xp_null.py`, `single_members.py`, `trunc_frac.py` |
| `gen/data_abc_blind/` | twelve further fresh runs of Stage C, 6950 to 6961 | `eval_valid.py` (Fig. 4) |
| `gen/data_abc_ext/k1/` to `k4/` | four extensions of each of the 16 held-out runs of Stages A to C | `make_abc_strict.py`, which builds `data_abc_h400/` for the strict test of Stages A to C |
| `gen/data_widebox/` | the runs of the wide coupling scan | the trainings of that scan; runs 9000 and 9010 are also the data of the PFN cross-check |

### The samples of the convention factors

`conventions/factors.json` carries a prediction from the hadron level of these samples to that of a
measurement. `conventions/make_factors.py` measured it on four samples of each generator at its
default tune, kept as per-event tables (`conventions/README.md`). They are available on request
like the files above.

### The npz files of the paper's run

The run wrote its results into `output/`. Its json files, `numbers/numbers_lin.tex`, the figures in
`figs/` and its npz files up to 12.5 MB (`three_bands_<tag>_lin.npz`, `fig_closure_mix.npz`,
`toy_final_*.npz`, `wifi_<tag>_fit.npz` and `envelope_test_MIXGEOV3S9_<box>_lin.npz`, the draws of
the comparison with 2000 variations for the boxes `tune`, `full`, `tune_sherpa`, `tune_herwig`,
`full_sherpa` and `full_herwig`; about 75 MB in all) are in the repository, in `analysis/output/`,
so `phase2tools/make_numbers.py` gives the paper's macros from them alone. In a clone, run it with
`NUMBERS_TEST=1`: its checks that tie the inputs to the run's folder on Symmetry do not apply there,
and it marks its output TEST ONLY, with the same 615 macros (Nov 8). Its four large npz files, 4.3 GB in all, are
not: `wifi_<tag>_basis_rebuilt.npz` for `MIXGEOV3S9`, `MIXGEOV3S9s1`, `MIXGEOV3S9s2` and
`MIXGEOV3M8S9`, 0.6 to 2.4 GB each, the basis functions of the learning band (Network exports,
above). Nor are its caches, the per-event observables in `output/pobs/` and the network outputs in
`output/logitcache/`, which the jobs rebuild.

They are available on request like the files above.

### Other trainings and samples

The second and third ensembles of Stages A to E, every DCTR network, the baseline of the paper
that appends the parameters to every particle (`<S>_P2<tag>cat<k>_concat.pt`,
`MIXP2v3fullcat<k>_concat.pt`), the trainings of the scan of the
number of terms, of the wide coupling scan and of the learning curve, and their data sets, are kept
as well and are available on request.

### Regenerating the samples

Every sample can be regenerated with the scripts and designs in `gen/` (`gen/README.md`). The run
number is the generator seed, so a run is fixed by its number and its parameter point. The one
exception is run 32077, generated with seed 5032077 (`gen/designs/seed_overrides.csv`). The four
extensions of a held-out run use the seeds run + k × 10^6, k = 1 to 4. Stages A to D and the wide
coupling scan use Sherpa 3.0.4, and Stage E uses Herwig 7.3.0, both pinned CVMFS builds
(`analysis/release/generators/README.md`). The HepMC files are not kept. Each run is extracted to
the two files above by `extract_particles_full_flavor.py` and `compute_shapes_only.py`, and
`make_dataset_v3.py` links the runs of Stages D and E into the data sets of the mixture. The
photon-free copies were made from those files by the scripts in `photon_free_data/`.

## Earlier drafts

The data of earlier drafts, which the paper does not read, are described in the git history:
`git show 1d66811:docs/DATA.md`. Their result files are in `analysis/output/` at commit 98c0421.
