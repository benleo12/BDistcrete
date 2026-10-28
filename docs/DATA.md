# Data: network exports and event samples

The git repository holds the scripts, the result files in `analysis/output/` and the released
networks in `analysis/release/`. The network exports and the event samples are too large for git.
This file lists those that draft r8 of the paper needs, and what reads each one. Paths are relative
to `analysis/`, except those that start with `gen/`, which are relative to the repository root.
The scripts open them at exactly these paths.

## Draft r8

### Public: the reference sample of the mixture and the thrust calculation

Two files are public, as the assets of the GitHub release `data-r8`,
https://github.com/benleo12/BDistcrete/releases/tag/data-r8, with `REFERENCE.md`, which describes
them (a copy is `analysis/release/REFERENCE.md`), and `SHA256SUMS`. They will also be archived at
Zenodo.

| file | bytes | what | read by |
|---|---|---|---|
| `reference_MIX17.npz` | 865,353,418 | the 1,149,840 reference events of the mixture, released as `MIX17`, as the event vectors a(Φ) of `MIXGEOV3S9_cond.npz` and the per-event observables of `MIXGEOV3S9_ref_v2_slim.npz`, with the run and the generator of every event | `analysis/release/examples/reproduce_paper.py`, `generator_band.py` and `learning_band.py` with `--reference-file`, `gentune.maxent` (`analysis/release/IMPOSE.md`) |
| `targets_thrust_nlo.npz` | 33,524 | the NNLL+NLO thrust calculation imposed in the paper, its 14 moments in the window, their covariance and its variations, copied from `output/thrust_targets_nlo_point.npz` | `gentune.maxent.load_targets` |

With them, numpy alone reproduces the central predictions of Sec. 4, the generator band and the
learning band, without the exports below. To download and check them into `analysis/release`,
where the examples look for them:

```bash
cd analysis/release
for f in reference_MIX17.npz targets_thrust_nlo.npz REFERENCE.md SHA256SUMS; do
  curl -LO https://github.com/benleo12/BDistcrete/releases/download/data-r8/$f
done
shasum -a 256 -c SHA256SUMS          # sha256sum -c SHA256SUMS on Linux; every line must say OK
```

### On request: the other exports and samples

Only the exports and samples below are not public yet. They will be archived at Zenodo with the
paper, and until then they are available on request from the authors. Every number and figure of
the paper can be redone without them, from the result files in `analysis/output/`
(`docs/REPRODUCE.md`). They are needed to redo those result files.

### Network exports

Each model has these files in `output/models/`: `<tag>_cond.npz` (the parameter networks, the
calibration constant and the stored event functions a(Φ) of the reference events),
`<tag>_trunk.pt` (the saved members), `<tag>_fitsplit.npz` (the events of every run set aside from
training) and its reference sample `<tag>_ref.npz`. The mixture also has `<tag>_ref_v2_slim.npz`,
which `rebuild_ref_lowmem.py` makes from it.

| files | what | read by |
|---|---|---|
| `output/models/MIXGEOV3S9_*` | the seventeen-parameter mixture of Sec. 4, first set of members, released as `MIX17` | `three_bands.py`, `theory_variations.py`, `band_checks.py`, `wifi_fit.py`, `wifi_diag.py`, `wifi_coverage.py`, `fig_closure_mix.py`, `bench_cost_final.py`, `package_release.py` |
| `output/models/MIXGEOV3S9_ref.npz` | its reference sample with the particles of every event | `bench_cost_final.py`, `package_release.py` |
| `output/models/MIXGEOV3S9s1_*`, `output/models/MIXGEOV3S9s2_*` | the second and third sets of members of the mixture | `three_bands.py` (the spread of Sec. 4.3), `wifi_*.py` (Fig. 7, Table 6), the strict test of Table 1 |
| `output/wifi_MIXGEOV3S9_basis.npz`, `..._MIXGEOV3S9s1_basis.npz`, `..._MIXGEOV3S9s2_basis.npz` | the basis functions of the learning band on the fit set and on the held-out runs, written by `wifi_embed.py` | `wifi_fit.py`, `wifi_diag.py`, `wifi_gensplit.py` |
| `output/models/E_P2v3fullfac0_*`, `output/models/F_P2v3fullfac0_*` | Stage D (Sherpa) and Stage E (Herwig), first set of members, released as `E` and `F` | `package_release.py`, `make_stage_table.py`, `precision_gap.py`, `closure_vs_distance.py` |
| `output/models/C_P2augfullfac0_*` | Stage C, first set of members | `eval_valid.py` (Fig. 5), `single_members.py` |

### Event samples

Each run is two files, `particles_full_<run>.npz` (the particles the networks read) and
`shapes_run_<run>.csv` (the event shapes the closure tests compare), and every run has
8 × 10^4 events unless stated.

| folder | what | read by |
|---|---|---|
| `data_stagePURE17_v3/` | the 720 training runs of Stages D and E (352 Sherpa, 368 Herwig) and their 48 held-out points, with `meta.json` | the training of Stages D and E and of the mixture, the standard test, `fig_closure_mix.py`, `trunc_frac.py` |
| `data_stagePURE17_v3_h400/` | the same with every held-out run at 4 × 10^5 events, the run and its four extensions | the strict test, `wifi_coverage.py`, `make_stage_table.py`, `closure_vs_distance.py` |
| `gen/data_abc/` | Stages A to C: the training runs of the densified designs `gen/designs/stage{A,B,C}_design_aug.csv` and their 16 held-out runs | the training of Stages A to C, `eval_valid.py`, `xp_null.py`, `single_members.py`, `trunc_frac.py` |
| `gen/data_abc_blind/` | the twelve blind runs 6950 to 6961 of Stage C (Sec. 3.5) | `eval_valid.py` (Fig. 5) |
| `gen/data_abc_ext/k1/` to `k4/` | four extensions of each of the 16 held-out runs of Stages A to C | `make_abc_strict.py`, which builds `data_abc_h400/` for the strict test of Stages A to C |

### Other trainings and samples

The second and third sets of members of Stages A to E, every concatenation network
(`<S>_P2<tag>cat<k>_concat.pt`, `MIXP2v3fullcat<k>_concat.pt`), the trainings of the term-count
scan, of the wide coupling scan (with its runs in `gen/data_widebox/`) and of the learning curve of
Sec. 3.1, and their data sets, are kept as well and are available on request. Their result files
are in `analysis/output/`.

### Regenerating the samples

Every sample can be regenerated with the scripts and designs in `gen/` (`gen/README.md`). The run
number is the generator seed, so a run is fixed by its number and its parameter point. The one
exception is run 32077, generated with seed 5032077 (`gen/designs/seed_overrides.csv`). The four
extensions of a held-out run use the seeds run + k × 10^6, k = 1 to 4. Stages A to D and the wide
coupling scan use Sherpa 3.0.4, and Stage E uses Herwig 7.3.0, both pinned CVMFS builds
(`analysis/release/generators/README.md`). The HepMC files are not kept. Each run is extracted to
the two files above by `extract_particles_full_flavor.py` and `compute_shapes_only.py`, and
`make_dataset_v3.py` links the runs of Stages D and E into the data sets of the mixture.

## Earlier draft with an α_s fit (not in the paper)

The samples and exports of the earlier draft, which fitted α_s(M_Z) and α_0 to the LEP data, are
the assets of the GitHub release `data-v1`. They hold its Stages A to C, its seven-parameter
mixture, its networks `E`, `Fauglong`, `MIX17aug` and `C_1M`, and the EERAD3 logs. Draft r8 reads
none of these archives. Some of its runs have the run number and parameter point of an archived
run, for example the Stage D and E training runs of the archives `C7` and `C8` and the blind runs
6950 to 6961 of `B3_held_C_valid`. `tools/get_data.sh --list` lists the archives, and `tools/get_data.sh <archive or
group>` downloads one, checks it against the `SHA256SUMS` of the release and unpacks it into
`analysis/`. That `SHA256SUMS` lists only `A0_theory_logs.tar`, and the archives `C7b` to `C9b`
are not among the assets of the release, so the script fetches only `A0_theory_logs` until the
release is completed. The table of the archives and their contents is in the git history:
`git show 1d66811:docs/DATA.md`.
