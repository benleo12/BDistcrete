# How the photon-free observables were made

The paper leaves the photons that the incoming e+ and e- radiate along the beam out of every
observable. The event samples (`docs/DATA.md`) store event shapes computed with them. The scripts in
this folder made photon-free copies once, on the Symmetry cluster at Perimeter on 6 November 2026,
before the run behind the paper's numbers (`docs/REPRODUCE.md`). They are the files that ran, byte
for byte, except where step 2 says otherwise. `MD5SUMS` gives the md5 of each file as it is on
Symmetry, and `md5sum -c MD5SUMS` in this folder checks them. No other script of the repository
imports them.

## What they made

- **The `_nb` folders.** A copy `<folder>_nb` of each data folder the paper reads, with the same file
  names: 27 folders, listed in `step3_rule_v3/NB_FOLDERS.txt`. The particle files are links to the
  originals, because the networks read every particle. In each `shapes_run_<run>.csv` the columns
  `1_minus_thrust`, `B_total`, `rho_heavy` and `mult_total` leave the beam photons out. Every other
  column, and every event without a beam photon, keeps the original text.
- **The run folder**, `/mnt/beegfs/bassi/reweighting/lmx_runs/`. The paper's scripts ran in its
  `maxent_code/`, which this file and `docs/REPRODUCE.md` call RUN. RUN's data folders are links to
  the `_nb` folders (step 2 added `lmx_runs/data_nb/data_stagePURE17_v4_nb` for a set whose runs
  are partly gone), and `RUN/output/MANIFEST.tsv` lists the files of `RUN/output/` with their size
  and sha256.
- **The refreshed reference bundles.** The 27 files under `RUN/output/` that hold per-event
  observables of a reference sample or of held-out runs, and the copy of the release reference
  `lmx_runs/release_ref/reference_MIX17.npz`. They hold the same photon-free values, and the beam
  photons leave the denominator of the kaon fraction. `RUN/output/bundle_rule.json` lists the 28
  files with their sha256 and the md5 of the `beam_photons.py` that made them.

The rule of `step3_rule_v3/beam_photons.py` finds the beam photons among the stored particles. It is
the same file as `analysis/beam_photons.py`, and `analysis/release/REFERENCE.md` states it: a base
rule and three additions, (i) to (iii). Step 1 used the base rule alone. Rule v2 added first forms
of (i) and (iii), which its docstring numbers (i) and (ii). Rule v3, the final one, added (ii) and
gave (i) and (iii) their final form.

Each event shape is rebuilt from the stored particle features with and without the flagged
photons, by `compute_hemisphere_shapes` of `gen/compute_efps.py`. The new value is the stored value
minus that difference, so what the rebuild misses cancels to first order. Where the rebuild misses
the stored value by more than 1e-2, or the new value would be negative, the stored value is kept.
The multiplicity loses the flagged photons.

## What rests on them

- Every number of the paper computed from 1-T, B_T, rho_H, the multiplicity of all kept particles
  or the kaon fraction: the closure widths, the envelope, the three bands and the comparisons with
  the LEP measurements. The x_p spectrum is not made here: `analysis/score_obs.py` computes it from
  the particles, with the same rule, when a network is scored.
- The counts of the rule and of its residual ambiguity, in the paper and in
  `analysis/release/REFERENCE.md`, for example 579,296,866 photons flagged in 322,685,107 of the
  340,720,000 events. They come from the summary of step 3 (`nb3_summary.py`).
- The observables of the release's `reference_MIX17.npz`. Its build copies them from the refreshed
  bundle `MIXGEOV3S9_ref_v2_slim.npz` (`analysis/release/REFERENCE.md`, How it was made).

## The three steps

Each step ran in a work folder of its own on Symmetry (`W` in its batch files), with its scripts in
`W/scripts`. It kept its records there: one record per unit (a particle file and its CSV, 4543
units in all), its logs, its checks and a summary. The records are not in the repository.

| folder | work folder | what it did |
|---|---|---|
| `step1_base_rule/` | `lmx_phaseA/observables` | made the `_nb` folders and the three shapes without the photons of the base rule, for all 340,720,000 events |
| `step2_rule_v2/` | `lmx_phaseB/observables` | rewrote the four columns of every CSV with rule v2, refreshed the reference bundles and built the run folder |
| `step3_rule_v3/` | `lmx_phaseB/observables_v3` | applied rule v3, the final one: rewrote the rows of the 956 events whose flags changed (in 751 CSVs), refreshed the bundles again from the originals and wrote `RUN/output/bundle_rule.json` |

### Step 1: the base rule

| file | what it does |
|---|---|
| `inventory.py W/inventory` | lists every data folder the paper reads and writes one unit per real particle file (`W/inventory/units.tsv`); `inv_summary.py` prints the inventory |
| `plan.py W 16` | shares the units out over 16 array tasks (`W/units.tsv`) |
| `nb_links.py W npz` | makes the `_nb` folders, copies `meta.json` and links every particle file to the original's |
| `nb_array.sbatch` (`--array=0-15`) | runs `nb_units.py`, which writes each unit's new CSV into its `_nb` folder (a temp file, then a rename) and its record |
| `nb_big.sbatch`, `nb_work.sbatch` | the same computation for the units of preempted tasks: `nb_bigunit.py` spreads one unit over many processes, `nb_worker.py` takes the next unit nobody has claimed |
| `nb_finish.sbatch` | `nb_links.py W csv` links the CSVs that are links in the original, `nb_links.py W verify` compares every `_nb` folder with its original, `nb_consist.py` checks that the merged and truncated sets equal their sources, and `nb_summary.py` writes the summary |
| `nb_core.py` | the method: the base rule, the rebuild and the new value; steps 2 and 3 use its rebuild with their own rule |
| `nb_spot.py`, `nb_xcheck.py`, `nb_outliers.py` | checks run apart from the jobs: spot checks of written CSVs, a comparison with an earlier independent code (its output on Symmetry is not in the release), and the events of one unit that the rebuild misses most |

### Step 2: rule v2

It starts from step 1's `_nb` CSVs and list of units.

| file | what it does |
|---|---|
| `beam_photons.py` | rule v2 |
| `nb2_test.sbatch` | `nb2_units.py` in test mode on the units in `UIDS`, writing only into `W/test` |
| `nb2_array.sbatch` (`--array=0-11`) | runs `nb2_units.py`: every unit again from the original CSV, with `mult_total` minus the flagged photons. By default (`NB_FAST=1`) it keeps step 1's value where the flags did not change, checking every hundredth such event; with `NB_FAST=0` it recomputes every event. The first 208 units ran with its first version, kept here as `nb2_units_full_v1.py` (then named `nb2_units.py`), which recomputes every event. The other 4335 ran with the default. `NB_FAST=0` was not used. All 56,014,942 values compared with step 1's were equal. |
| `scan_data_tokens.py REPO OUT` | every data path that the paper's scripts name, read from a checkout of this repository (`W/results/data_tokens.tsv`) |
| `finish.sbatch` | the checks (`nb_links.py verify`, `nb_consist.py`, `nb2_spot.py`, the independent code `indep_check.py` with `indep_shapes.py`, and the summary `nb2_summary.py`); `refresh_bundles.py all` writes the refreshed bundles into `lmx_runs`, and `verify_refresh.py` reads them back; `layout_plan.py` and `layout_make.py` build the run folder, and `check_paths.py` checks that every data path of the paper's scripts leads from there into an `_nb` folder; it exited 1, for a known reason: the paper's scripts give the published `maxent_code/` as the default `MAXENT_DIR` in 17 places, and the run sets `MAXENT_DIR` to RUN |
| `scan_obs_cache.py`, `npzkeys.py` | helpers: which files hold per-event observables, and the arrays of an npz file |
| `nb_core.py`, `nb_links.py`, `nb_consist.py` | step 1's files |

Not every file of this step is here exactly as it ran. `nb2_units_full_v1.py` was saved under that
name 14 minutes after its 208 units started, and no md5 of the file they loaded was recorded. A
first refresh job made 17 of the 28 bundles with earlier versions of `refresh_bundles.py` and
`verify_refresh.py`, which are not kept. The finish job made the other 11 with the files here and
checked all 28. Step 3 made every bundle again from the originals, so the final bundles do not
depend on those earlier versions.

### Step 3: rule v3

It starts from step 2's `_nb` CSVs, list of units and refreshed bundles, and from its
`results/layout_plan.tsv` and `results/data_tokens.tsv`. RUN must already hold this
`beam_photons.py` (`analysis/symmetry/rerun/sync_run.sh` copies it there).

| file | what it does |
|---|---|
| `beam_photons.py`, `beam_photons_v2.py` | rule v3, the same file as `analysis/beam_photons.py`, and step 2's rule v2 |
| `NB_FOLDERS.txt`, `nb3_folders.py` | the 27 `_nb` folders, the only ones the step may write in, and the check that they are those `nb_links.py` knows |
| `nb3_test.sbatch` | the folder check, then six units twice, as a delta from the current CSV and recomputed in full from the original (`NB_FULL=1`), into `W/test` only; `nb3_equal.py` compares the two, `nb3_spot.py` spot-checks them and `show_events.py` prints chosen events |
| `nb3_array.sbatch` (`--array=0-1`) | runs `nb3_units.py`, which recomputes from the original CSV only the rows whose flags differ between rules v2 and v3, and rewrites only those CSVs |
| `nb3_finish.sbatch` | the checks (as in step 2, with `nb3_spot.py`, `indep_check3.py` and the summary `nb3_summary.py`), which `gate.py` requires; then `delta_caches.py snap` saves the bundles' values under rule v2, `refresh3_bundles.py all` refreshes the bundles again from the originals, `delta_caches.py check` allows changes only in the events whose flags changed, `delta_caches.py manifest` updates `MANIFEST.tsv` and `delta_caches.py rule` writes `RUN/output/bundle_rule.json`; last, `verify_refresh3.py` reads the bundles back and `check_paths3.py` checks the run folder again; it exited 1, as expected: besides the 17 defaults of step 2, two rows of `MANIFEST.tsv` no longer matched, because `sync_run.sh` had replaced `output/thrust_targets_nlo_point.npz` and `output/calc_point.json` with copies whose numbers are the same and whose citation text differs |
| `refresh3_bundles.py`, `verify_refresh3.py`, `check_paths3.py` | step 2's `refresh_bundles.py`, `verify_refresh.py` and `check_paths.py`, with the changes their docstrings name |
| `nb_core.py`, `nb_links.py`, `nb_consist.py`, `indep_shapes.py` | the files of steps 1 and 2 |

`nb3_array.sbatch` and the `refresh` and `rule` parts of `nb3_finish.sbatch` refuse to run once the
run behind the paper's numbers has started (`RUN/output/.lmx_rerun_start` exists), because that run
reads what they write.

## Running them elsewhere

The scripts name the folders of Symmetry, all under `/mnt/beegfs/bassi/reweighting`: `gen/` (the
generator runs), `maxent_code/` (the training and test sets, and the published run's `output/`),
`release_ref/` (the release's earlier `reference_MIX17.npz`), `env/` (the run's Python),
`lmx_runs/` (the run folder), the work folders under `lmx_phaseA/` and `lmx_phaseB/`, and
`lmx_tests/` (the earlier test that `nb_xcheck.py` reads). They name no folder of a laptop. To run
them elsewhere, change these paths in a copy; the files here stay as they ran.

- The Python scripts set `R = '/mnt/beegfs/bassi/reweighting'` near the top. `nb_core.py`,
  `nb_xcheck.py` and `NB_FOLDERS.txt` write out full paths instead. The self-test of
  `beam_photons.py` and `beam_photons_v2.py` takes particle files as arguments and falls back on
  two files of Symmetry.
- The batch files set their work folder `W` and the other folders they use, the log path of
  `#SBATCH -o`, Symmetry's partition `preq`, and the run's Python
  (`/mnt/beegfs/bassi/reweighting/env/torch/bin/python`).
- `nb_core.py` imports `compute_efps.py` from the folder `NB_GEN`, by default the cluster's `gen/`.
  This repository's `gen/compute_efps.py` is the same file (md5 adb1869a4339bd396eb155fece15fe23),
  so set `NB_GEN` to that folder.
- They need numpy and pandas (the run's Python has the packages of `environment.yml`), and the
  tools `zip`, `cmp` and `md5sum`. The batch files also need Slurm, `find` and GNU `time`. They read
  the event samples and the network files of `docs/DATA.md`, which are on request, and write only
  into the `_nb` folders, the run folder and their work folder.

## Names in the scripts

The docstrings use the names of the review rounds that wrote them. Phase A is step 1. Phase B and
B0 are step 2, whose "step 3" is the refresh and "step 4" the run folder. B5 and B6 are the rounds
that wrote step 3. D4 is the decision to leave the beam photons out of every observable, D11 the
refitted calibration constants, D12 the run folder, and I1 `bundle_rule.json`. "The rerun" is the
run behind the paper's numbers. "The skeptic" and "the verifier" are reviewers of a round, and
`indep_shapes.py` is the verifier's `v_shapes.py`, the name its usage line still gives.
