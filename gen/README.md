# How the samples of the paper were generated

These are the scripts that generated every event sample of the paper on the Perimeter Symmetry
cluster, as they ran there. The runs of Stages D and E use the published recipe of
`analysis/release/generators/` (`gen_stageE.sh` for Sherpa, `gen_stageF.sh` for Herwig, the pinned
environment `cvmfs_env.sh`). Stages A to C and the wide coupling scan use `gen_stageABC.sh` in this
folder. This folder adds the designs and the batch wrapper that ran a few thousand of
them. The paths start at `R`, the parent of `gen/` and of the analysis folder. `gen_batch.sbatch`
takes it as the parent of `MAXENT_DIR`, the analysis folder, which defaults to the Symmetry copy
`/mnt/beegfs/bassi/reweighting/maxent_code`. Set `MAXENT_DIR` to the `analysis/` folder of your
copy to run elsewhere. `gen_one.sh` and `run_el9.sh` find `R` from their own place in `gen/`.
`R` also holds the cvmfsexec install (`tools/cvmfsexec/`) and the EL9 image
(`containers/almalinux9.sif`), which are not in this repository. The layout is the same here, so
`../gen` from `analysis/` is this folder.

Submit from `R`, the top of a checkout. A fresh clone needs `gen/logs` and `gen/tasks` first, and
`make_tasks.py` writes the task files:

```bash
mkdir -p gen/logs gen/tasks
(cd gen && python make_tasks.py 80000 tasks/v3.txt designs/stageE_design_v3.csv designs/stageF_design_v3.csv)
sbatch --array=0-3 --export=ALL,MAXENT_DIR=$PWD/analysis,TASKS=gen/tasks/v3.txt,GEN_DATA=gen/data_v3 gen/gen_batch.sbatch
```

`make_tasks.py` knows only the designs of Stages D and E. The 720 training runs also need
`python make_tasks.py 80000 tasks/release.txt designs/stageE_design.csv designs/stageF_design_aug.csv`,
submitted with `GEN_DATA=gen/data_release`. For Stages A to C, write the lines by hand as
`gen_stageABC.sh <rid> <alphas> <baryon_fraction|-> <strange_fraction|-> <kt_0|-> <events>`, as
`analysis/widebox_submit.sh` does.

`make_design_v3.py` and `make_design_v4.py` read `../data_stagePURE17/meta.json`, where that folder
sits on Symmetry. In a checkout it is `analysis/data_stagePURE17`, so link it first, from the top of
the checkout, with `ln -s analysis/data_stagePURE17 data_stagePURE17`. They then rewrite their designs
unchanged.

| file | what it is |
|---|---|
| `make_design_v3.py` | 256 further training points and 16 further held-out points per generator, scrambled Sobol in the same box. With the earlier points these are the designs of the released Stage D (Sherpa) and Stage E (Herwig) networks |
| `make_design_v4.py` | 1024 further training points per generator, for the scan of the number of training runs |
| `make_design_abc_aug.py` | the denser training designs of Stages A, B and C |
| `designs/` | the designs, one row per run, `run_id` is also the generator seed. `PURE17_meta.json` holds the boxes and axis order of the mixture. The designs of the released Stages D and E, `stage{E,F}_design_v3all.csv`, are in `analysis/` and `analysis/release/generators/` |
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
