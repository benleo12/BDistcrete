# Released stages

The closure tables come from `make_stage_table.py`, run on the paper's photon-free scorings
(`analysis/symmetry/rerun/table_lmx.sbatch`), point by point in `closure_<stage>.csv`. Every
closure number is measured on parameter points held out of training, with the reference sample
reweighted to each point and compared with the generator run there. The six observables leave out
the photons radiated by the beams, as every observable of the paper does (`REFERENCE.md`): 1 - T,
the total multiplicity, the broadening, the heavy jet mass and the strange fraction change, and
the baryon count does not. Each head carries the calibration constant refitted on these
observables.

A `cond` head is the head of one generator, the factorized network of the paper: its log ratio is the mean over the 4 ensemble members of the inner product of a 48-vector of the event and one of the parameters (rank 48), divided by the calibration temperature. The head of `MIX17` has one parameter network per generator, against the same event vectors. Its file names it `geometric`, after the network's own interpolation between the two generators at a fraction between 0 and 1, which the paper does not use (`README.md`, Two generators at once).

Every stage was trained on e+e- -> gamma*/Z -> q qbar at sqrt(s) = 91.2 GeV, with every hadron of c*tau above 10 mm left undecayed in both generators. The cards are in `README.md`, under "Conventions you have to match". The line naming the generator and the `setting` and `default` columns come from `gentune.axes`. A Herwig setting goes after `set` in an input file, with `*GeV` after the value of `pTmin`. A Sherpa setting `AHADIC:KEY` is `KEY` inside the `AHADIC:` block of the run card. The default is the value the generator ships with, and for the fraction of `MIX17` it is the one half of the paper's central point.

## E

- 8 parameters, `cond` head (above), rank 48, 4 ensemble members
- reference sample 4,224,000 events, calibration temperature 1.000
- own events supported: yes
- Sherpa 3.0.4. In the paper: Stage D

| parameter | name | setting | low | high | centre | default |
|---|---|---|---|---|---|---|
| 0 | `alphas` | `ALPHAS(MZ)` | 0.1100 | 0.1299 | 0.1199 | 0.118 |
| 1 | `pt_max` | `AHADIC:PT_MAX` | 0.4030 | 0.9939 | 0.6985 | 0.68 |
| 2 | `alpha_l` | `AHADIC:ALPHA_L` | 2.5239 | 4.9744 | 3.7492 | 3.9 |
| 3 | `gamma_l` | `AHADIC:GAMMA_L` | 0.2855 | 0.9474 | 0.6165 | 0.48 |
| 4 | `strange_fraction` | `AHADIC:STRANGE_FRACTION` | 0.3608 | 0.5789 | 0.4698 | 0.46 |
| 5 | `baryon_fraction` | `AHADIC:BARYON_FRACTION` | 0.1109 | 0.2798 | 0.1954 | 0.17 |
| 6 | `alpha_g` | `AHADIC:ALPHA_G` | 0.6000 | 1.9000 | 1.2500 | 0.97 |
| 7 | `beta_l` | `AHADIC:BETA_L` | 0.0200 | 0.4000 | 0.2100 | 0.18 |

Closure on parameter points held out of training, reference reweighted against the generator:

| observable | worst relative shift | worst pull | held points |
|---|---|---|---|
| 1 - T | -0.61% | -4.1 | 24 |
| total multiplicity | +0.27% | +5.2 | 24 |
| broadening | -0.38% | -4.2 | 24 |
| heavy jet mass | -0.57% | -3.8 | 24 |
| baryons | -1.36% | -9.7 | 24 |
| strange fraction | -0.55% | -5.2 | 24 |

Effective sample fraction across the held points: 0.74 to 0.97.

## F

- 8 parameters, `cond` head (above), rank 48, 4 ensemble members
- reference sample 4,416,000 events, calibration temperature 1.000
- own events supported: yes
- Herwig 7.3.0. In the paper: Stage E

| parameter | name | setting | low | high | centre | default |
|---|---|---|---|---|---|---|
| 0 | `alpha_fsr` | `/Herwig/Shower/AlphaQCDFSR:AlphaIn` | 0.1005 | 0.1300 | 0.1153 | 0.102337 |
| 1 | `ptmin` | `/Herwig/Shower/PTCutOff:pTmin` | 0.5000 | 1.2986 | 0.8993 | 0.654714 |
| 2 | `clmax` | `/Herwig/Hadronization/ClusterFissioner:ClMaxLight` | 2.6000 | 4.8000 | 3.7000 | 3.528693 |
| 3 | `clpow` | `/Herwig/Hadronization/ClusterFissioner:ClPowLight` | 0.7500 | 3.2000 | 1.9750 | 1.849375 |
| 4 | `psplit` | `/Herwig/Hadronization/ClusterFissioner:PSplitLight` | 0.6544 | 1.1448 | 0.8996 | 0.914156 |
| 5 | `pwtsquark` | `/Herwig/Hadronization/HadronSelector:PwtSquark` | 0.2204 | 0.5794 | 0.3999 | 0.374094 |
| 6 | `pwtdiquark` | `/Herwig/Hadronization/HadronSelector:PwtDIquark` | 0.2019 | 0.4972 | 0.3496 | 0.33107 |
| 7 | `clsmr` | `/Herwig/Hadronization/ClusterDecayer:ClSmrLight` | 0.4000 | 1.2000 | 0.8000 | 0.78 |

Closure on parameter points held out of training, reference reweighted against the generator:

| observable | worst relative shift | worst pull | held points |
|---|---|---|---|
| 1 - T | -1.11% | -7.6 | 24 |
| total multiplicity | +0.71% | +13.7 | 24 |
| broadening | -0.74% | -8.4 | 24 |
| heavy jet mass | -0.88% | -6.0 | 24 |
| baryons | +2.05% | +12.2 | 24 |
| strange fraction | +0.53% | +5.3 | 24 |

Effective sample fraction across the held points: 0.29 to 0.92.

## MIX17

- 17 parameters, mixture head (`geometric`, above), rank 48, 4 ensemble members. The paper evaluates it at the fraction 0, where it is Sherpa alone, and at 1, where it is Herwig alone, and mixes the two predictions, (1 - xi) <O>_S + xi <O>_H (`README.md`)
- reference sample 1,149,840 events, calibration temperature 1.000
- own events supported: yes
- Sherpa 3.0.4 and Herwig 7.3.0. In the paper: the seventeen-parameter mixture

| parameter | name | setting | low | high | centre | default |
|---|---|---|---|---|---|---|
| 0 | `S:alphas` | `ALPHAS(MZ)` | 0.1100 | 0.1299 | 0.1199 | 0.118 |
| 1 | `S:pt_max` | `AHADIC:PT_MAX` | 0.4030 | 0.9939 | 0.6985 | 0.68 |
| 2 | `S:alpha_l` | `AHADIC:ALPHA_L` | 2.5239 | 4.9744 | 3.7492 | 3.9 |
| 3 | `S:gamma_l` | `AHADIC:GAMMA_L` | 0.2855 | 0.9474 | 0.6165 | 0.48 |
| 4 | `S:strange_fraction` | `AHADIC:STRANGE_FRACTION` | 0.3608 | 0.5789 | 0.4698 | 0.46 |
| 5 | `S:baryon_fraction` | `AHADIC:BARYON_FRACTION` | 0.1109 | 0.2798 | 0.1954 | 0.17 |
| 6 | `S:alpha_g` | `AHADIC:ALPHA_G` | 0.6000 | 1.9000 | 1.2500 | 0.97 |
| 7 | `S:beta_l` | `AHADIC:BETA_L` | 0.0200 | 0.4000 | 0.2100 | 0.18 |
| 8 | `H:alpha_fsr` | `/Herwig/Shower/AlphaQCDFSR:AlphaIn` | 0.1005 | 0.1300 | 0.1153 | 0.102337 |
| 9 | `H:ptmin` | `/Herwig/Shower/PTCutOff:pTmin` | 0.5000 | 1.2986 | 0.8993 | 0.654714 |
| 10 | `H:clmax` | `/Herwig/Hadronization/ClusterFissioner:ClMaxLight` | 2.6000 | 4.8000 | 3.7000 | 3.528693 |
| 11 | `H:clpow` | `/Herwig/Hadronization/ClusterFissioner:ClPowLight` | 0.7500 | 3.2000 | 1.9750 | 1.849375 |
| 12 | `H:psplit` | `/Herwig/Hadronization/ClusterFissioner:PSplitLight` | 0.6544 | 1.1448 | 0.8996 | 0.914156 |
| 13 | `H:pwtsquark` | `/Herwig/Hadronization/HadronSelector:PwtSquark` | 0.2204 | 0.5794 | 0.3999 | 0.374094 |
| 14 | `H:pwtdiquark` | `/Herwig/Hadronization/HadronSelector:PwtDIquark` | 0.2019 | 0.4972 | 0.3496 | 0.33107 |
| 15 | `H:clsmr` | `/Herwig/Hadronization/ClusterDecayer:ClSmrLight` | 0.4000 | 1.2000 | 0.8000 | 0.78 |
| 16 | `fraction` | none | 0.0000 | 1.0000 | 0.5000 | 0.5, the central point |

Closure on parameter points held out of training, reference reweighted against the generator. The 48 held points are the 24 of each generator, at the fraction 0 or 1, so the closure does not depend on how the predictions are mixed:

| observable | worst relative shift | worst pull | held points |
|---|---|---|---|
| 1 - T | -1.20% | -5.8 | 48 |
| total multiplicity | -0.72% | -9.0 | 48 |
| broadening | -0.79% | -6.4 | 48 |
| heavy jet mass | +0.76% | -4.0 | 48 |
| baryons | +1.15% | +5.3 | 48 |
| strange fraction | +0.58% | -4.5 | 48 |

Effective sample fraction across the held points: 0.17 to 0.58.
