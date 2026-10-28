# Example HepMC files

Two small event files from the generators the heads were trained on, at their default tunes, and
the outputs gentune must give for them. The second stage of `python -m gentune.selftest models`
reads them, so the self-test now covers the HepMC parsing and the thrust frame too, which the
first stage, working from stored features, does not.

| file | generator | format | events | particles per event | gzip bytes | md5 of the uncompressed file |
|---|---|---|---|---|---|---|
| `herwig_default.hepmc.gz` | Herwig 7.3.0, ThePEG 2.3.0 | HepMC2 (IO_GenEvent, written by HepMC 3.03.01) | 30 | 37.7 | 189,511 | `fa39f82a5c554054b2785f6660e3aeda` |
| `sherpa_default.hepmc.gz` | Sherpa 3.0.4 | HepMC3 (Asciiv3, HepMC 3.03.01) | 40 | 43.6 | 180,255 | `c2459666a960d360e2b1277ec16e6ec1` |

`expected.npz` holds the outputs, and `make_expected.py` wrote it.

Both are e+e- -> gamma*/Z -> hadrons at sqrt(s) = 91.2 GeV, in GeV and mm (`U GEV MM`), with
every hadron of c*tau above 10 mm left undecayed, exactly as in the training samples. Both are
unweighted. Herwig writes a weight of 1 on every event, and Sherpa the same weight on every
event, 6.6066e4 to fifteen digits, so `parse_hepmc` has nothing to warn about.

## How they were made

With the published generator scripts, `generators/gen_stageF.sh` and `generators/gen_stageE.sh`,
changed in one line each: the published scripts delete the HepMC once the training arrays are
extracted, and these copies keep it.

```
gen_stageF.sh:  rm -f events.hepmc r$rid.run   ->   rm -f r$rid.run
gen_stageE.sh:  rm -f events.hepmc             ->   (removed)
```

They ran on 2026-10-27 on the Symmetry cluster, inside the EL9 container of
`generators/README.md` (almalinux 9 under apptainer, CVMFS mounted with cvmfsexec), with the
builds `cvmfs_env.sh` pins: the LCG_109 view, Herwig `7.3.0p1-c7a43` and Sherpa `3.0.4-22c71`.
The run id is the seed.

```
STAGEF_BASE=$PWD/herwig ./gen_stageF.sh 9730 0.102337 0.654714 3.528693 1.849375 0.914156 0.374094 0.33107 0.78 100
STAGEE_BASE=$PWD/sherpa ./gen_stageE.sh 9304 0.118 0.68 3.9 0.48 0.46 0.17 0.97 0.18 100
```

The cards these write, with the shell variables filled in:

```
# Herwig, r9730.in
read snippets/EECollider.in
cd /Herwig/MatrixElements
insert SubProcess:MatrixElements 0 MEee2gZ2qq
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 91.2
set /Herwig/Shower/AlphaQCDFSR:AlphaIn 0.102337
set /Herwig/Shower/PTCutOff:pTmin 0.654714*GeV
set /Herwig/Hadronization/ClusterFissioner:ClMaxLight 3.528693
set /Herwig/Hadronization/ClusterFissioner:ClPowLight 1.849375
set /Herwig/Hadronization/ClusterFissioner:PSplitLight 0.914156
set /Herwig/Hadronization/HadronSelector:PwtSquark 0.374094
set /Herwig/Hadronization/HadronSelector:PwtDIquark 0.33107
set /Herwig/Hadronization/ClusterDecayer:ClSmrLight 0.78
set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm
set /Herwig/Decays/DecayHandler:LifeTimeOption 0
read snippets/HepMC.in
set /Herwig/Analysis/HepMC:PrintEvent 100000000
set /Herwig/Analysis/HepMC:Filename events.hepmc
saverun r9730 EventGenerator
```
```
Herwig read -i $HERWIGPATH --repo $HERWIG_REPO r9730.in
Herwig run r9730.run -N 100 --seed 9730 --repo $HERWIG_REPO
```

```yaml
# Sherpa, Sherpa.yaml, run as `Sherpa` in its directory
EVENTS: 100
EVENT_OUTPUT: HepMC3_GenEvent[events]
RANDOM_SEED: 9304
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
ALPHAS(MZ): 0.118
ORDER_ALPHAS: 1
SHOWER_GENERATOR: CSS
FRAGMENTATION: Ahadic
AHADIC:
  PT_MAX: 0.68
  ALPHA_L: 3.9
  GAMMA_L: 0.48
  STRANGE_FRACTION: 0.46
  BARYON_FRACTION: 0.17
  ALPHA_G: 0.97
  BETA_L: 0.18
PROCESSES:
- 11 -11 -> 93 93:
    Order: {QCD: 0, EW: 2}
```

**The parameter values are the generators' own defaults**, which was checked rather than
assumed. The eight Herwig values are the `newdef` lines of Herwig 7.3.0's
`defaults/Shower.in` and `defaults/Hadronization.in`, digit for digit, and a `get` of each
setting after `read snippets/EECollider.in` returns them to the six digits it prints. Sherpa's
settings report for the run lists all eight under "Settings kept at their default value". The
cards set them anyway, because the training cards set every parameter. Two settings of the cards
are not defaults, as in training: Sherpa's `ORDER_ALPHAS: 1` (default 2), and Herwig's
`MaxLifeTime 10*mm` (default -1, no cut, so Herwig would decay K0S, Lambda and the rest), which
gives both generators the same stable particles. `LifeTimeOption 0` is Herwig's default and is
set only to be explicit.

**The scripts refuse fewer than 100 events, and 100 events are 640 and 430 kB once gzipped**, so
each file is the head of its run, cut after the last whole event that keeps it under 200 kB, with
the closing line of the listing kept:

```
awk -v n=30 '/^E /{k++} k<=n || /END_EVENT_LISTING/ || !NF' herwig/runs/run_9730/events.hepmc | gzip -9 -n > herwig_default.hepmc.gz
awk -v n=40 '/^E /{k++} k<=n || /END_EVENT_LISTING/ || !NF' sherpa/runs/run_9304/events.hepmc | gzip -9 -n > sherpa_default.hepmc.gz
```

**The recipe reproduces the files byte for byte.** Both runs were made twice, in two jobs, and the
100-event files have the same md5 both times, `527901b8c4b78f444df10b1b4e1762ba` for Herwig and
`0b8fab219cbee5921c984d93656a1c90` for Sherpa.

**gentune reads them exactly as the training pipeline did.** The generator scripts also ran
`extract_particles_full_flavor.py` on the full 100-event files, on the cluster and with the
LCG view's numpy. The features `gentune.features` builds from the same files on a Mac with numpy
2.2.4 are bit-identical to that extraction for every one of the 100 events of each run, masks and
baryon counts included.

## expected.npz

What `gentune` makes of each file, written by `make_expected.py` and read by the self-test. For
`s` in `samples` (`herwig`, `sherpa`):

| key | what it is |
|---|---|
| `s_file`, `s_generator` | the file and a label |
| `s_particles`, `s_mask`, `s_nbaryon` | `features.features_from_events(features.parse_hepmc(file))` |
| `s_sha256` | one sha256 of those three arrays, float32, bool and int32 in that order |
| `s_theta_from`, `s_theta_to` | the MIX17 parameter pair, 17 numbers each in the order of `axes.names('MIX17')` |
| `s_logw` | `(head.logit(A, to) - head.logit(A, from))/head.T` per event, the unnormalized log weight |
| `s_weights` | `head.ratio_weights(A, to, from)`, normalized to sum to one |

and `stage` (`MIX17`) and `numpy_version` (2.2.4, the one that wrote it).

Each pair starts at the tune its file was generated at and moves one shower coupling:

| file | from | to | N_eff/N |
|---|---|---|---|
| Herwig | Sherpa default, Herwig default, fraction 1 | Herwig `alpha_fsr` 0.102337 -> 0.110 | 0.975 |
| Sherpa | Sherpa default, Herwig default, fraction 0 | Sherpa `alphas` 0.118 -> 0.122 | 0.983 |

In the geometric mixture of MIX17 the Sherpa block drops out of the logit exactly at fraction 1
and the Herwig block at fraction 0, so the other generator's block has no effect on either pair.
All four points pass `head.check`.

The same weights from the command line:

```
python examples/reweight_own_events.py --stage MIX17 --hepmc examples/data/herwig_default.hepmc.gz \
  --from 0.118 0.68 3.9 0.48 0.46 0.17 0.97 0.18  0.102337 0.654714 3.528693 1.849375 0.914156 0.374094 0.33107 0.78  1 \
  --to   0.118 0.68 3.9 0.48 0.46 0.17 0.97 0.18  0.110    0.654714 3.528693 1.849375 0.914156 0.374094 0.33107 0.78  1
```

## What the self-test allows

On the machine that wrote `expected.npz` the files reproduce it bit for bit. Elsewhere the
comparison has to allow for the thrust axis. It comes from the single-seed sign iteration the
heads were trained with, and in an event where two candidate axes are nearly tied a difference
in the last bit of a dot product can send the iteration to the other one. Only cos theta and phi
change when that happens, and the event's log weight with them: by 0.02 at the median and 0.09 at
most when the axis just changes sign, measured on these files, and by up to about 0.9 when it
lands on a different axis altogether. So the self-test requires the mask and the columns that do
not depend on the axis (z, mass, baryon number, charge) to agree in every event to 1e-5, lets at
most 5 percent of a file's events move their axis, and requires every other event to reproduce
its log weight to 1e-2, once the normalization, which a moved event shifts, is taken out. Noise
of a few float32 units in the features moves a log weight here by at most 4e-4, because MIX17
rounds its inputs to float16 and such noise can cross a rounding boundary.

Rerun `make_expected.py` only when the example files or the MIX17 files change. Rerunning it to
make a failing self-test pass defeats the self-test.
