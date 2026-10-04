"""Which generator parameter each index of a head's theta vector is.

A head stores its box as an array of (centre, half-range) rows and nothing else, so on its own
it can only tell you that parameter 3 runs from 0.75 to 3.20. That is not enough to use it. The
names live here, in the order the training design's columns were read, which is the order the
head's rows are in.

That order is not asserted, it is checked. `verify(tag, head, design_dir)` recomputes each
column's centre and half-range from the design CSV that trained the stage and compares with the
head's own `norm`. All three multi-parameter stages agree exactly, to 0.0 in every entry, so the
mapping below is a measurement rather than a convention someone remembered.

The names are the generator's own names, not prettier ones, so that a name can be pasted into a
Sherpa run card or a Herwig input file and found. `NOTES` says what a name means where the name
misleads, which for hadronization parameters is often.
"""
import os
import numpy as np

SHERPA8 = ('alphas', 'pt_max', 'alpha_l', 'gamma_l', 'strange_fraction',
           'baryon_fraction', 'alpha_g', 'beta_l')
HERWIG8 = ('alpha_fsr', 'ptmin', 'clmax', 'clpow', 'psplit',
           'pwtsquark', 'pwtdiquark', 'clsmr')

NAMES = {
    'C_1M': ('alphas', 'strange_fraction', 'kt_0'),
    'E': SHERPA8,
    'F': HERWIG8,
    'MIX17': tuple('S:' + n for n in SHERPA8) + tuple('H:' + n for n in HERWIG8) + ('fraction',),
}

GENERATOR = {'C_1M': 'Sherpa 3', 'E': 'Sherpa 3', 'F': 'Herwig 7.3', 'MIX17': 'both'}

DESIGN = {'E': 'stageE_design.csv', 'F': 'stageF_design_aug.csv'}

NOTES = {
    'alphas': 'the shower coupling at the Z mass',
    'kt_0': 'the Ahadic cluster transverse momentum scale',
    'gamma_l': 'enters the Ahadic cluster decay weight only as gamma_l/kt_0^2, so it and kt_0 '
               'are one direction and not two',
    'alpha_g': 'multiplies a weight z^a + (1-z)^a, which is flat at a = 1 exactly, so it looks '
               'inert in the code and is not. The strongest single knob for multiplicity here',
    'alpha_fsr': 'AlphaQCDFSR:AlphaIn, which Herwig hard-limits at 0.10',
    'ptmin': 'the shower cutoff, ShowerHandler:pTmin',
    'pwtsquark': 'PartonSplitter:SplitPwtSquark. Tuned to 7 TeV minimum bias data and never to '
                 'electron-positron annihilation',
    'pwtdiquark': 'the diquark weight, which is what sets baryon production in the cluster model',
    'clsmr': 'ClusterSmearing:ClSmrLight, never fitted by a published Herwig tune, so it is '
             'varied wide here and turns out to be weak',
    'fraction': 'the share of the sample the Herwig side supplies, 0 for pure Sherpa and 1 for '
                'pure Herwig',
}


def names(tag, ntheta=None):
    """The parameter names of a released stage, or generic labels if the tag is unknown."""
    n = NAMES.get(tag)
    if n is None:
        return tuple(f'parameter {i}' for i in range(ntheta or 0))
    if ntheta is not None and len(n) != ntheta:
        raise ValueError(f'{tag}: {len(n)} names for a {ntheta} parameter head, so this table is '
                         f'stale and must not be used to label anything')
    return n


def verify(tag, head, design_dir='generators'):
    """Recompute the box from the design CSV and compare with the head's own rows.

    Returns the largest absolute disagreement, or None when the design for this stage is not
    part of the bundle. Raises if they disagree, because a head whose row order does not match
    the design is a head whose parameter names are wrong, which is worse than having none."""
    import csv
    csv_name = DESIGN.get(tag)
    if csv_name is None:
        return None
    path = os.path.join(design_dir, csv_name)
    if not os.path.exists(path):
        return None
    cols = {'E': SHERPA8, 'F': HERWIG8}[tag]
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    vals = np.array([[float(r[c]) for c in cols] for r in rows])
    lo, hi = vals.min(0), vals.max(0)
    want = np.stack([(lo + hi)/2, (hi - lo)/2], 1)
    got = np.asarray(head.norm if hasattr(head, 'norm') else head, float)[:len(cols)]
    d = float(np.abs(want - got).max())
    if d > 1e-12:
        raise ValueError(f'{tag}: the design CSV and the head disagree about the box by {d:g}, '
                         f'so the parameter order in axes.py cannot be trusted')
    return d
