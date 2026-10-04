# Regenerating the training samples

These are the generator scripts, parameter designs and extraction code that produced the event
samples the released heads were trained on. Nothing here is needed to USE a head. It is here so
the samples can be rebuilt, and so the parameter that a run id corresponds to is on the record.

| file | what it is |
|---|---|
| `gen_stageE.sh` | one Sherpa run: write the card, generate, extract, drop the HepMC |
| `gen_stageF.sh` | the same for Herwig |
| `cvmfs_env.sh` | pinned generator environment, sourced by both |
| `stageE_design.csv` | the Sherpa design, 96 training points and 8 held, 8 parameters |
| `stageF_design.csv` | the original Herwig design, 64 training and 8 held |
| `stageF_augment_design.csv` | 48 further Herwig points, added to fix the design's conditioning |
| `stageF_design_aug.csv` | the two Herwig files combined, 112 training points |
| `extract_particles_full_flavor.py` | HepMC to the per-particle arrays the models read |
| `compute_shapes_only.py` | HepMC to the event-shape table used for closure |
| `compute_efps.py` | the HepMC parser and shape definitions the previous file imports |

Each design row is `run_id, role, <the parameters>`. The run id is also the generator seed, so a
row fully determines a run. `role` is one of:

- `train`, a point the head was fitted on. 96 for Sherpa, 112 for Herwig once the augmentation
  is included, 64 before it.
- `held`, a point kept out of training entirely, 8 per generator, which is what every closure
  number in `STAGES.md` is measured on.
- `scan_<parameter>`, a one-dimensional scan along a single axis with the other seven fixed at a
  single reference point, 5 points each. The reference point is not the centre of the box. These
  were used to look at how one knob moves an observable and are no part of training or of the
  closure numbers.

`stageF_augment_design.csv` holds only `train` rows, since it adds training points to an existing
design rather than being a design of its own.

## The container

Both generators come from CVMFS and are built for EL9. Run them in an EL9 container. The
original Perlmutter scripts instead pointed the loader at the host's `libreadline.so.7` under
the name `libreadline.so.8`, which worked because Herwig only uses readline for interactive
line editing, but it is not a recipe to hand anyone.

```bash
podman-hpc pull docker.io/library/almalinux:9
podman-hpc run --rm --network=none -v /cvmfs:/cvmfs -v "$PWD:/work" almalinux:9 bash -lc '
  cd /work && ./gen_stageF.sh 8000 0.1060 0.5 3.2789 1.5575 0.8673 0.4678 0.2784 0.8960 80000'
```

The generator scripts source `cvmfs_env.sh` themselves, so there is nothing to set up by hand.
Nothing here reaches the network once the image is pulled, which is why `--network=none` is in
the command. `shifter` and `apptainer` work the same way given the same two bind mounts,
`/cvmfs` and your working directory. Any EL9 base image will do. The only thing the container
supplies is a userspace the CVMFS builds were compiled against, and `almalinux:9` is simply the
smallest one that has it.

`STAGEF_BASE` and `STAGEE_BASE` say where runs and extracted data go, defaulting to the working
directory. Output is `data/particles_full_<rid>.npz` and `data/shapes_run_<rid>.csv`. The HepMC
file is deleted once both have been written, because at 80,000 events it is about 1.6 GB per run
and nothing downstream reads it again.

`cvmfs_env.sh` checks for unresolved libraries and refuses to continue if it finds any, so
running it on the wrong host fails immediately and says why rather than failing later inside an
integration step.

## The whole design at once

One run of 80,000 events takes a few minutes, and the runs are independent, so a design fits on
a single node. On Perlmutter the 77-run Herwig campaign took 23 minutes of wall time this way:

```bash
tail -n +2 stageF_design.csv | tr ',' ' ' | xargs -P 48 -L 1 bash -c '
  rid=$0; role=$1; shift 1
  ./gen_stageF.sh "$rid" "$@" 80000'
```

Run this inside the container, not outside it. The `shift 1` drops `role` and leaves `$0` as the
run id, which is why there is one shift and not two.

## Sample sizes and conventions

Training runs are 80,000 events and so are the held runs of the two single-generator designs
here: the `n_held` column of `closure_E.csv` and `closure_F.csv` reads 80,000 throughout. The
mixture is built by resampling those same events rather than generating new ones, so two of its
ten held points hold 40,000 and the rest 80,000, which `closure_MIX17.csv` records per point.
Both generators keep every hadron with `c*tau` greater than 10 mm undecayed. Herwig decays
those by default and Sherpa does not, so `gen_stageF.sh` sets `DecayHandler:MaxLifeTime` to
match. A sample generated without that line is not comparable with these.

## What produced the released sample

Verified on 2026-09-21, inside the container described above:

- Herwig 7.3.0, ThePEG 2.3.0, Sherpa 3.0.4, all resolving every library from CVMFS with none
  taken from the host.
- CVMFS carries 88 separate builds of Herwig 7.3.0p1. The original script chose one with
  `ls -d '7.3.0p1-*' | head -1`, which is not a pin, since a build whose hash sorted earlier
  would have replaced it silently. `cvmfs_env.sh` pins a hash instead.
- The released sample was generated with build `18d2e`, and `cvmfs_env.sh` pins `c7a43`, which
  is the build the LCG_109 view itself selects. They differ in how GSL resolves: `18d2e` wants
  GSL 2.7 and took it from the host's `/usr/lib64` while LCG's GSL 2.8 was also loaded, so two
  versions of GSL were in one process, whereas `c7a43` loads GSL 2.8 alone.
- **The two builds were checked against each other rather than assumed equivalent**: 2000 events
  on the Stage F card at the box centre with seed 8000 give byte-identical HepMC, the same md5
  and the same 39,670,649 bytes. So the pinned recipe reproduces the released sample, and the
  duplicated GSL had no effect on the output.

Both scripts were then run as published, in the container, with the commands above. Each writes
its particle file and its shape file and removes the HepMC, and the results are as expected at
the Z pole: 39.2 particles per event and a mean of one minus thrust of 0.0641 for Herwig, 41.8
and 0.0687 for Sherpa, on 1000 events each.

## The thrust axis, and why two versions of it ship

`compute_efps.py` computes the event shapes with an exact thrust axis: for each pair of
momenta the axis perpendicular to both fixes the hemisphere of every other particle, the four
sign choices for the pair complete the candidate, and the largest candidate is the thrust. The
single-seed sign iteration it replaced on 2026-09-28 converges to a local maximum in about 4
percent of Z-pole events, biasing the mean of one minus thrust by +1.2 percent, the heavy jet
mass by +1.5 percent, and putting those events into the far tail. The closure tables shipped
with this release were computed before the change, with the same finder on both sides of every
comparison, so they remain valid as tests of the reweighting, and a regeneration with this
bundle gives slightly different shape values than the shipped `shapes_run_*.csv` would.

`gentune/features.py` keeps the old finder on purpose. It only sets the frame the per-particle
features are expressed in, the heads were trained in that frame, and a user's events must be
embedded in the same one. The two are separated so that the observable is right and the
embedding is consistent, and they change together when the heads are retrained.

