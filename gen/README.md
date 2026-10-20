# How the samples of the paper were generated

These are the scripts that generated every event sample of the paper on the Perimeter Symmetry
cluster, as they ran there. Each run uses the published recipe of `analysis/release/generators/`
(`gen_stageE.sh` for Sherpa, `gen_stageF.sh` for Herwig, the pinned environment `cvmfs_env.sh`).
This folder adds the designs, the task lists and the batch wrapper that ran a few thousand of
them. The paths start at `/mnt/beegfs/bassi/reweighting`, the parent of `gen/` and of the
analysis folder, so set `R` in the scripts to your own copy. The layout is the same here, so
`../gen` from `analysis/` is this folder.

| file | what it is |
|---|---|
| `make_design_v3.py` | 256 further training points and 16 further held-out points per generator, scrambled Sobol in the same box. With the earlier points these are the designs of the released Stage D (Sherpa) and Stage E (Herwig) networks |
| `make_design_v4.py` | 1024 further training points per generator, for the scan of the number of training runs |
| `make_design_abc_aug.py` | the denser training designs of Stages A, B and C |
| `designs/` | every design, one row per run, `run_id` is also the generator seed. `PURE17_meta.json` holds the boxes and axis order of the mixture |
| `make_tasks.py` | turns design CSVs into a task file, one run per line |
| `gen_batch.sbatch` | one array element per node: mounts CVMFS without root (cvmfsexec), enters the EL9 container once and runs its tasks in parallel |
| `gen_one.sh` | one run inside the container. It skips a run that is already done, and `GEN_SEED_OFFSET` gives more events at the same point with an independent seed |
| `run_el9.sh` | one run of a published script in the container, by hand |
| `gen_stageE.sh`, `gen_stageF.sh`, `gen_stageABC.sh` | the per-run scripts. `GEN_SEED` defaults to the run id, which is the published recipe |
| `extract_particles_full_flavor.py`, `compute_shapes_only.py`, `compute_efps.py`, `cvmfs_env.sh` | identical to the copies in `analysis/release/generators/` |
| `array_checksums.py` | the md5 of every array of every run, to compare two copies of a data set |

`analysis/make_dataset_v3.py`, run by `analysis/symmetry/build_dataset.sbatch`, links the
generated runs into the training sets.

One run does not use its run id as its seed. Sherpa 3.0.4 loops without end after event 60,000
of run 32077, so that run was generated with seed 5032077 (`designs/seed_overrides.csv`).
