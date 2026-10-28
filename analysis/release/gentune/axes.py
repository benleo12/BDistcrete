"""Which generator parameter each index of a head's theta vector is.

A head stores its box as an array of (centre, half-range) rows and nothing else, so on its own
it can only tell you that parameter 3 runs from 0.75 to 3.20. That is not enough to use it. The
names live here, in the order the training design's columns were read, which is the order the
head's rows are in.

That order is not asserted, it is checked. `verify(tag, head, design_dir)` recomputes each
column's centre and half-range from the design CSV that trained the stage and compares with the
head's own `norm`. All three multi-parameter stages agree exactly, to 0.0 in every entry, so the
mapping below is a measurement rather than a convention someone remembered.

The names are short labels that follow the generator's own, with `S:` and `H:` marking the
Sherpa and the Herwig block of the mixture. A label is not a setting. `SETTING` gives the setting
each one is, exactly as the training cards in generators/ write it, `DEFAULT` the value the
generator itself ships with, which with a fraction of one half is the paper's central point, and
`card(tag, theta)` the lines that put a generator at theta. `NOTES` says what a name means where the name misleads, which for
hadronization parameters is often.
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

DESIGN = {'E': 'stageE_design_v3all.csv', 'F': 'stageF_design_v3all.csv'}

# The setting behind each name, copied from the cards that generated the training runs:
# generators/gen_stageE.sh (Sherpa 3.0.4), generators/gen_stageF.sh (Herwig 7.3.0p1) and, for
# C_1M, analysis/gen_stageC_v2.sh. A Herwig path goes after `set` in an input file. A Sherpa
# name goes in the YAML run card, and BLOCK:KEY means KEY inside the BLOCK: block, which is how
# the card writes every Ahadic parameter. card() writes both out in full.
SETTING = {
    'alphas': 'ALPHAS(MZ)',
    'pt_max': 'AHADIC:PT_MAX',
    'alpha_l': 'AHADIC:ALPHA_L',
    'gamma_l': 'AHADIC:GAMMA_L',
    'strange_fraction': 'AHADIC:STRANGE_FRACTION',
    'baryon_fraction': 'AHADIC:BARYON_FRACTION',
    'alpha_g': 'AHADIC:ALPHA_G',
    'beta_l': 'AHADIC:BETA_L',
    'kt_0': 'AHADIC:KT_0',
    'alpha_fsr': '/Herwig/Shower/AlphaQCDFSR:AlphaIn',
    'ptmin': '/Herwig/Shower/PTCutOff:pTmin',
    'clmax': '/Herwig/Hadronization/ClusterFissioner:ClMaxLight',
    'clpow': '/Herwig/Hadronization/ClusterFissioner:ClPowLight',
    'psplit': '/Herwig/Hadronization/ClusterFissioner:PSplitLight',
    'pwtsquark': '/Herwig/Hadronization/HadronSelector:PwtSquark',
    'pwtdiquark': '/Herwig/Hadronization/HadronSelector:PwtDIquark',
    'clsmr': '/Herwig/Hadronization/ClusterDecayer:ClSmrLight',
}

# Herwig wants a unit on a dimensionful setting. Sherpa's PT_MAX is in GeV without one.
UNIT = {'ptmin': '*GeV'}

# The value each setting takes when the card does not set it. The Sherpa ones are read from the
# 3.0.4 source (MODEL/Main/Model_Base.C and AHADIC++/Tools/Hadronisation_Parameters.C), the
# Herwig ones from defaults/Shower.in and defaults/Hadronization.in of the CVMFS 7.3.0p1 build
# the samples were generated with (analysis/design_stageF_herwig.py records the probe). Together
# with a fraction of one half they are the paper's central point, the CENTRAL of
# analysis/symmetry/three_bands.sbatch.
DEFAULT = {
    'alphas': 0.118, 'pt_max': 0.68, 'alpha_l': 3.9, 'gamma_l': 0.48,
    'strange_fraction': 0.46, 'baryon_fraction': 0.17, 'alpha_g': 0.97, 'beta_l': 0.18,
    'kt_0': 1.21,
    'alpha_fsr': 0.102337, 'ptmin': 0.654714, 'clmax': 3.528693, 'clpow': 1.849375,
    'psplit': 0.914156, 'pwtsquark': 0.374094, 'pwtdiquark': 0.33107, 'clsmr': 0.78,
    # not a generator default: the paper's central point gives the two generators equal weight
    'fraction': 0.5,
}

NOTES = {
    'alphas': 'the shower coupling at the Z mass',
    'kt_0': 'the Ahadic cluster transverse momentum scale',
    'gamma_l': 'enters the Ahadic cluster decay weight only as gamma_l/kt_0^2, so it and kt_0 '
               'are one direction and not two',
    'alpha_g': 'multiplies a weight z^a + (1-z)^a, which is flat at a = 1 exactly, so it looks '
               'inert in the code and is not. The strongest single knob for multiplicity here',
    'alpha_fsr': 'AlphaQCDFSR:AlphaIn, which Herwig hard-limits at 0.10',
    'ptmin': 'the shower cutoff in transverse momentum, PTCutOff:pTmin, in GeV',
    'pwtsquark': 'HadronSelector:PwtSquark, the weight of a strange pair when a cluster decays '
                 'to hadrons. Not PartonSplitter:SplitPwtSquark, the strange weight of the '
                 'non-perturbative gluon splitting, which was tuned to 7 TeV minimum bias data, '
                 'never to electron-positron annihilation, and is not varied here',
    'pwtdiquark': 'the diquark weight, which is what sets baryon production in the cluster model',
    'clsmr': 'ClusterDecayer:ClSmrLight, the smearing of the hadron directions in a light '
             'cluster decay. Never fitted by a published Herwig tune, so it is varied wide here '
             'and turns out to be weak',
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


def generator_of(name):
    """'sherpa' or 'herwig' for a parameter name, with or without its S: or H: prefix, and None
    for the mixing fraction, which belongs to neither."""
    k = name.split(':')[-1]
    if k not in SETTING:
        return None
    return 'herwig' if SETTING[k].startswith('/Herwig/') else 'sherpa'


def find(nm, key):
    """The index of a parameter by name. A name without its S: or H: prefix is accepted when only
    one parameter carries it, so ptmin finds H:ptmin in MIX17."""
    if key in nm:
        return nm.index(key)
    hits = [i for i, n in enumerate(nm) if n.split(':')[-1] == key]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f'no parameter called {key!r}. The parameters are {", ".join(nm)}')


def default(tag, generator=None, centre=None):
    """A point at the generators' own defaults, in the head's parameter order.

    With generator None this is the paper's central point: every parameter at its generator's
    default and, for MIX17, the fraction at one half. With generator 'sherpa' or 'herwig' it is
    where a sample made by that generator alone at its defaults sits. For MIX17 that puts the
    fraction at 0 or 1, where the other generator's block drops out of the log ratio exactly, so
    its values change no weight. They are put at `centre`, the box centre, because that is where
    every training run of the mixture has them and so where head.support() expects them. Without
    `centre` they are put at their own defaults, which gives the same weights."""
    nm = NAMES.get(tag)
    if nm is None:
        raise ValueError(f'{tag}: no defaults are recorded for this stage')
    if generator not in (None, 'sherpa', 'herwig'):
        raise ValueError(f'generator must be sherpa or herwig, not {generator!r}')
    gens = [generator_of(n) for n in nm]
    if generator is not None and generator not in gens:
        raise ValueError(f'{tag} is a {GENERATOR[tag]} stage, so it has no {generator} default')
    t = []
    for i, (n, g) in enumerate(zip(nm, gens)):
        if g is None:
            t.append(DEFAULT['fraction'] if generator is None else float(generator == 'herwig'))
        elif generator is None or g == generator or centre is None:
            t.append(DEFAULT[n.split(':')[-1]])
        else:
            t.append(float(centre[i]))
    return np.array(t)


# The presets point() takes, and the generator argument of default() each stands for.
PRESETS = {'default': None, 'sherpa-default': 'sherpa', 'herwig-default': 'herwig'}


def point(tag, tokens, ntheta, centre=None):
    """A parameter point written the way the examples take it on the command line.

    tokens is either one number per parameter, in the head's order, or a preset of PRESETS
    followed by any number of name=value changes, for example
    ['herwig-default', 'H:pwtdiquark=0.40'] or ['default', 'alphas=0.120']. A name may drop its
    S: or H: prefix when only one parameter carries it. The presets are default(tag, generator,
    centre). Raises ValueError with a message that names what is wrong; whether the point lies in
    the box is for the head to check."""
    tokens = str(tokens).split() if isinstance(tokens, str) else [str(x) for x in tokens]
    nm = names(tag, ntheta)
    if tokens and tokens[0] in PRESETS:
        t = default(tag, PRESETS[tokens[0]], centre)
        for tok in tokens[1:]:
            k, eq, v = tok.partition('=')
            if not eq:
                raise ValueError(f'after a preset each value is name=value, as in '
                                 f'{nm[0]}={t[0]:g}, and {tok!r} is not')
            i = find(nm, k)
            try:
                t[i] = float(v)
            except ValueError:
                raise ValueError(f'{tok}: {v!r} is not a number')
    else:
        try:
            t = np.array([float(x) for x in tokens])
        except ValueError:
            raise ValueError(f'a point is {ntheta} numbers, or one of {", ".join(PRESETS)} '
                             f'followed by name=value changes, not {" ".join(tokens)!r}')
    if len(t) != ntheta:
        raise ValueError(f'the point has {len(t)} values but {tag} takes {ntheta}, in this '
                         f'order: {", ".join(nm)}')
    return t


def card(tag, theta):
    """The card lines that put the generator of a stage at theta, as the training cards write them.

    Sherpa lines go in the YAML run card and Herwig lines in the input file, each replacing the
    default. For MIX17 both blocks come back, and the fraction, which no generator has a setting
    for, as a comment."""
    nm = names(tag, len(theta))
    top, block, herwig, rest = [], {}, [], []
    for n, v in zip(nm, np.asarray(theta, np.float64)):
        k = n.split(':')[-1]
        s = SETTING.get(k)
        if s is None:
            rest.append(f'# {n} {v:.10g} is not a generator setting: {NOTES.get(k, "")}')
        elif s.startswith('/'):
            herwig.append(f'set {s} {v:.10g}{UNIT.get(k, "")}')
        elif ':' in s:
            b, key = s.split(':')
            block.setdefault(b, []).append(f'  {key}: {v:.10g}')
        else:
            top.append(f'{s}: {v:.10g}')
    for b, lines in block.items():
        top += [f'{b}:'] + lines
    return top + herwig + rest


def listing(tag, head):
    """The lines the examples print for a stage: the box with the generator's default for each
    parameter, the defaults as card lines, and the notes. Defaults are left out for a stage
    whose names are not in this file."""
    nm = names(tag, head.nt)
    known = tag in NAMES
    title = 'training box, one row per parameter'
    if known:
        title += ', and the default the generator ships with'
        if 'fraction' in nm:
            title += (", or for the fraction the paper's one half, which together make the "
                      "paper's central point")
    out = [title + ':']
    for i, (lo, hi) in enumerate(head.box):
        dflt = DEFAULT.get(nm[i].split(':')[-1]) if known else None
        out.append(f'  [{i:>2}]  {nm[i]:<20s} {lo:>10.4f}  to  {hi:>10.4f}   '
                   f'centre {head.centre[i]:.4f}' + ('' if dflt is None else f'   default {dflt:.10g}'))
    if known:
        out.append('the defaults as card lines, written the way the training cards write them:')
        out += ['  ' + line for line in card(tag, default(tag))]
    for k in nm:
        note = NOTES.get(k.split(':')[-1])
        if note:
            out.append(f'       {k}: {note}')
    return out


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
