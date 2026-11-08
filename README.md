# BDistcrete

The code of *Combined Uncertainties from Event Generators and Precision QCD* (B. Assi and
J. Thaler). The LaTeX source of the paper is kept in a separate repository, private until
publication.

## What the paper does

A prediction from an event generator depends on the values of its parameters and on the choice of
generator, and a precision calculation has uncertainties of its own. The paper obtains both from
one stored sample of e+e- → hadrons events at the Z pole. As in DCTR (Andreassen and Nachman), a
classifier that takes the generator parameters as an input reweights the sample to any point of a
box of seventeen parameters: eight of Sherpa, eight of Herwig, and the fraction xi of Herwig. The
paper's classifier is factorized: its log weight is an inner product of learned functions of the
event and of the parameters, so a new point does not need the particles again. The paper compares
it with the DCTR network, which appends the parameters to every particle. A prediction at xi is
(1 - xi) times that of Sherpa plus xi times that of Herwig. From the same events every prediction
gets three bands: the generator band, its spread when every parameter, xi included, is drawn flat
over its range; the learning band, the uncertainty of the trained weights themselves; and the
theory band, from the variations of an NNLL+NLO thrust calculation that a maximum-entropy weight
imposes on the events. The predictions are compared with the LEP data.

## Start here

The trained networks and their reader, the Python package `gentune`, need only numpy. Four steps,
from nothing to a checked install:

```
git clone https://github.com/benleo12/BDistcrete
cd BDistcrete && python3 -m venv .venv && source .venv/bin/activate
pip install numpy
cd analysis/release && python -m gentune.selftest models
```

The clone downloads about 1 GB, most of it the paper's reference sample in `deposit/` (Data,
below), and takes about 2000 MB on disk. The self-test takes 3 seconds
on a MacBook Pro with an Apple M3 Max (14 cores, 36 GB) and numpy 2.5.4. It must print PASS for
every stage. [`analysis/release/README.md`](analysis/release/README.md) goes on from there: how to
weight a sample of your own, the generator and learning bands, how to impose a calculation, and
`examples/reproduce_paper.py`, which checks the paper's central predictions and bands on the
reference sample.

## Where to look

| file | what it gives |
|---|---|
| [`analysis/release/README.md`](analysis/release/README.md) | the released networks: how to use them, what they were trained on, and how accurate they are |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | every figure, table and quoted number of the paper, with the command and the inputs that make it |
| [`docs/DATA.md`](docs/DATA.md) | the reference sample and calculation in `deposit/`, how to join and check them, and the event samples, network exports and large npz files of the paper's run, which are too large for git |

## Contents

| path | what it holds |
|---|---|
| `analysis/` | every script of the paper; they run from this folder |
| `analysis/release/` | the released networks, `gentune`, the examples, the generator scripts and designs, the closure tables |
| `deposit/` | the reference sample of the mixture, in 19 parts, and the paper's thrust calculation, with the record of the build and their sha256 (`docs/DATA.md`) |
| `analysis/symmetry/` | the batch scripts, as they ran on the Symmetry cluster at Perimeter |
| `analysis/phase2tools/` | `make_numbers.py`, which writes every quoted number of the paper as a LaTeX macro |
| `analysis/output/` | the result files of the paper's run that the paper reads: its json files, its small npz files, `numbers/numbers_lin.tex` and the figures in `figs/` (its large npz files are not in git, `docs/DATA.md`); the inputs of the calculation (its fixed point `calc_point.json`, the EERAD3 coefficients `eerad_*.npy`); and its result at that point, `thrust_targets_nlo_point.npz`, which the run imposes |
| `analysis/conventions/` | the factors that take a prediction to the hadron level of a measurement, and the script that measured them |
| `analysis/photon_free_data/` | the scripts that made the event shapes without the photons radiated by the beams |
| `analysis/lepcov/`, `analysis/rivet_ref/` | the LEP measurements (HEPData tables and the YODA files of Rivet) and their reader |
| `analysis/ares_recovered/`, `analysis/eerad3/` | our drivers and patches for ARES and EERAD3, and the ARES tables |
| `gen/` | the scripts and designs that generated the event samples |
| `analysis/perlmutter/`, `analysis/farm/`, `tools/` | batch scripts and data downloads of earlier drafts |
| `environment.yml`, `environment-tf.yml` | the analysis environment, and the TensorFlow one that only the cross-check with the energyflow PFN needs |

## Data

The reference sample of the mixture and the paper's thrust calculation are in `deposit/`, with the
record of the build and their sha256. GitHub refuses files over 100 MB, so the reference sample,
865 MB, is stored there in 19 parts. [`docs/DATA.md`](docs/DATA.md) says how to join and check
them, and lists the event samples, the network exports and the large npz files of the paper's run,
which are too large for git. Those will be archived at Zenodo with the paper, and until then they
are available on request.

## External codes

They are not included: Sherpa 3.0.4 and Herwig 7.3.0 (pinned CVMFS builds,
`analysis/release/generators/README.md`), ARES (GPL-3, upstream commit e75c88c with our edits in
`analysis/ares_recovered/ares_edits.txt`) and EERAD3 1.0 (with `analysis/eerad3/eerad3_blk.patch`).

## Earlier drafts

Scripts of earlier drafts, which the paper does not use, stay in `analysis/`. Their result files
are in the git history, at commit 98c0421. The last section of [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
says how they differ and where their instructions are.

## License

The code is released under the MIT License (`LICENSE`). ARES and EERAD3 are not included and keep
their own licenses.

## Citation

See `analysis/release/CITATION.bib`.
