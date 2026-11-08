# Hadron-level convention factors

`factors.json` holds the factors that carry a Sherpa or Herwig prediction from our hadron level to the hadron
level of a measurement. They were measured with each generator at its default tune.

Our hadron level is convention **a**: initial-state radiation (ISR) on, hadrons with c*tau > 10 mm stable, no
neutrinos, and the beam photons removed. The other conventions are:

- **c**: as a, but hadrons with lifetimes below 1 ns decayed (K0S, Lambda, Sigma+-, Xi0, Xi- and Omega-).
  This is DELPHI's convention.
- **nu**: as a, with neutrinos included.
- **cnu**: c and nu together.
- **b**: as a, with ISR off.

The file has three blocks: `meta`, `sherpa` and `herwig`. Each generator block gives the ratios `c_over_a`,
`nu_over_a`, `cnu_over_a` and `b_over_a`, each with a `_err` entry for its statistical error:

- per bin of the DELPHI all-particle thrust table d11-x01-y02 (`delphi_y02`);
- per bin of the ALEPH thrust table d54 (`aleph_d54`); both give the bin edges `lo` and `hi` in tau = 1-T;
- for the means (`means`): <1-T>, <B_T>, <rho_H>, P_win, tau_win, <n_ch> (the charged particles among the 100
  most energetic, with the beam photons kept in that list, as in the pipeline), <n_b> and the 14 window moments of
  `output/thrust_targets_nlo_point.npz`.

Every entry is a plain ratio: multiply a prediction in convention a by it. Distributions are fractions of all
events. Means are over all events, except tau_win and the window moments, which are means over the events with
0.05 <= tau < 1/3, as in the calculation. The `meta` block defines every convention, observable and estimator
exactly, and gives the sample sizes, the source with checksums, the number of events in each bin and
consistency checks.

Each generator's prediction is multiplied by its own factors, after the calculation is imposed and before the
two generators are mixed: DELPHI `c_over_a`, OPAL <1-T> `cnu_over_a`, L3 <n_ch> no factor.

## How it was made

`make_factors.py` (this folder) reads the per-event tables of the convention samples, on Symmetry in
`/mnt/beegfs/bassi/reweighting/lmx_phaseA/conventions/data`. Per generator they hold 2.0M events with ISR on
(a) and with ISR off (b), and 1.0M events with the 1 ns decays, ISR on (c) and off (d). They were generated with
the cards of `gen/gen_stageE.sh` (Sherpa 3.0.4) and `gen/gen_stageF.sh` (Herwig 7.3.0p1) at the default tunes.

c/a and cnu/a compare the same events: the decays in each c event are undone, which rebuilds convention a
exactly. nu/a also compares the same events. b/a compares independent samples.

To make it again, from `analysis/`:

    python3 -B conventions/make_factors.py --data /mnt/beegfs/bassi/reweighting/lmx_phaseA/conventions/data \
        --rivet rivet_ref --targets output/thrust_targets_nlo_point.npz --out conventions

This takes about 1.5 minutes and 3 GB of memory. The file here was made from a local copy of the data, md5
for md5 the same, and with the repository's targets file (md5 c25f4a23...). Run on Symmetry with its default
targets, `maxent_code/output/thrust_targets_nlo_point.npz` (md5 9324d3c8...), it gives the same factors and a
different `meta.source.targets.md5`: the two files differ only in their `calc_point` text, and only `keys`,
`window`, `central` and `sigma_pert` are read.

The errors are statistical only. The factors were measured at the default tunes. With 20% more or fewer K0S
and hyperons per event at a fixed event shape (<1-T> and <n_ch> held), the DELPHI factor moves by up to 0.44
(Sherpa) and 0.50 (Herwig) percentage points in 0.05 <= tau < 0.35 and by up to 2.45 and 2.19 points in the first
bin, and the OPAL factor by about 0.15 and 0.22 points. That study and the numbers for the paper are in
`/mnt/beegfs/bassi/reweighting/lmx_phaseB/conventions` on Symmetry (the fixed-shape values come from its
independent verification; the plain reweighting, which also moves <1-T> by 3-5%, gives 0.37 and 0.41).
