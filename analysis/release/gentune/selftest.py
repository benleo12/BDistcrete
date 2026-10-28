"""Prove that this install reproduces the logits the shipped files produce.

Every released stage ships a small bundle of reference events together with the logits to
reproduce at several parameter points. This runs the numpy trunk and the head over those events
and compares.

Its scope, stated precisely, because a PASS is easy to read more into than it carries. The
stored events hold the six per-particle columns ALREADY CONSTRUCTED, so this first stage
exercises the two networks and the final column mapping but NOT the HepMC parsing or the
thrust-frame construction that produced those columns. And the stored logits are the ones the
shipped files produce, with the features rounded to float16 for the stages trained that way, as
gentune.Trunk does for your own events. The training run's own logits are kept alongside as
logit_probe_from_export, and the largest difference between the two as precision_gap, so that
gap is on the record.

The second stage covers the rest. It reads the example HepMC files in examples/data, a Herwig
7.3.0 file in HepMC2 and a Sherpa 3.0.4 file in HepMC3, both at the default tunes and made with
the training cards, and compares what gentune makes of them with expected.npz there: the
features, and the MIX17 weights for one parameter pair per file. That comparison cannot be
exact, for a reason worth knowing. The thrust axis comes from the sign iteration the heads were
trained with, and in an event where two candidate axes are nearly tied, a difference in the
last bit of one dot product, from another CPU or another numpy, can send the iteration to the
other candidate. Only cos theta and phi move when that happens, and the event's log weight with
them, by a few hundredths when the axis only changes sign and by up to about 0.9 when it lands
on a different axis altogether, as events rebuilt in a rotated frame have shown. So:

  - the mask and the columns that do not depend on the axis, z, mass, baryon number and charge,
    must agree in every event, to TOL_FEATURE
  - an event whose cos theta or phi differ by more than TOL_FEATURE counts as moved, and at most
    MOVED of a file's events may move. The iteration ends in a local maximum in about 4 percent
    of Z-pole events, and only events that close to a tie can move at all
  - every other event must reproduce its log weight to TOL_LOGW, after the normalization, which
    a moved event shifts for all of them, is taken out

Usage: python -m gentune.selftest [models_dir [examples_dir]]
"""
import glob, os, sys
import numpy as np

from .head import load_head
from .trunk import Trunk
from .features import trunk_features, parse_hepmc, features_from_events
from . import axes

TOL = 1e-3            # far above the float32 rounding of the pooled sum, far below any real error

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'data')
# The features are of order one, so this is a few tens of float32 units.
TOL_FEATURE = 1e-5
# Noise of a few float32 units in the features moves a MIX17 log weight by up to 4e-4, measured
# on these files, because the stage rounds its inputs to float16 and the noise can cross a
# rounding boundary. A sign flip of the thrust axis moves one by 0.02 at the median.
TOL_LOGW = 1e-2
MOVED = 0.05
AXIS_FREE = [0, 3, 4, 5]      # z, log10 mass, baryon number, charge


def check(tag, models='models'):
    hp = f'{models}/{tag}_head.npz'
    c = np.load(hp)
    kind = str(c.get('head_kind', 'cond'))
    head = load_head(hp)
    sb = np.load(f'{models}/{tag}_selftest.npz')
    tp = f'{models}/{tag}_trunk.npz'
    if os.path.exists(tp):
        A = Trunk(tp).embed(trunk_features(sb['particles']), sb['mask'])
        if int(c['additive']):
            A = np.concatenate([A, np.ones(A.shape[:2] + (1,), A.dtype)], -1)
        dA = float(np.max(np.abs(np.asarray(A, np.float64) - np.asarray(sb['AE'], np.float64))))
    else:
        # A stage exported before the event-side weights were saved. Its head is exact on the
        # reference sample and nothing else, so the feature path is not exercised here.
        A, dA = sb['AE'], float('nan')
    f = np.stack([head.logit(A, t) for t in sb['theta_probe']])
    df = float(np.max(np.abs(f - sb['logit_probe'])))
    w = head.ratio_weights(A, sb['theta_probe'][-1], sb['theta_probe'][0])
    ok = df < TOL
    tr = 'own events' if os.path.exists(tp) else 'reference only'
    # dA compares against the reference vectors the TRAINING run exported. It is reported and
    # not tested, because a stage exported before feature_precision was recorded reads its
    # float16-trained features in float32 and differs there by construction. The logit
    # comparison is the test: it is against the path a user actually runs.
    neff = head.n_eff(w)/len(w)
    print(f'{tag:>12} {kind:>9} d={head.nt:<3} events={len(sb["mask"]):<6} {tr:>14}  '
          f'dA(vs training)={dA:.1e}  max|dlogit|={df:.2e}  '
          f'N_eff/N={neff:.3f}  {"PASS" if ok else "FAIL"}')
    if neff < 0.5:
        print(f'{"":>12} {_probe_note(tag, head, sb["theta_probe"])}')
    return ok


def _probe_note(tag, head, probe):
    """What the N_eff/N of the first stage measures, read off the stored probe pair.

    The weights it summarizes take the first probe point to the last. In C_1M, E and F that pair
    moves one weak parameter and N_eff/N stays near one. In MIX17 it moves only the mixing
    fraction, from 0.5 to 0.9, so from an even mixture of the two generators most of the way to
    pure Herwig, and N_eff/N = 0.073 next to PASS reads like a failure when it is not one."""
    try:
        nm = axes.names(tag, head.nt)
    except ValueError:
        nm = tuple(f'parameter {i}' for i in range(head.nt))
    d = np.flatnonzero(np.abs(probe[-1] - probe[0]) > 1e-12)
    moves = ' and '.join(f'{nm[i]} {probe[0][i]:.3g} -> {probe[-1][i]:.3g}' for i in d)
    why = (', from an even mixture of the generators to mostly Herwig, so a low N_eff/N is '
           'expected and is not a failure' if [nm[i] for i in d] == ['fraction'] else '')
    return f'the probe pair moves only {moves}{why}'


def check_examples(models='models', examples=EXAMPLES):
    """The second stage: the example HepMC files through the whole path, against expected.npz.

    Returns True or False, or None when the files it needs are not there, which is a skip."""
    ep = os.path.join(examples, 'expected.npz')
    ex = np.load(ep) if os.path.exists(ep) else None
    stage = str(ex['stage']) if ex is not None else 'MIX17'
    hp, tp = f'{models}/{stage}_head.npz', f'{models}/{stage}_trunk.npz'
    missing = [p for p in (ep, hp, tp) if not os.path.exists(p)]
    if ex is not None:
        missing += [p for p in (os.path.join(examples, str(ex[f'{s}_file'])) for s in ex['samples'])
                    if not os.path.exists(p)]
    if missing:
        print(f'\nsecond stage SKIPPED, because these files are missing: {", ".join(missing)}. '
              f'It reads the example HepMC files of examples/data, which a checkout of v2.0.1 or '
              f'later has. With gentune installed by pip, give that folder as the second '
              f'argument: python -m gentune.selftest MODELS EXAMPLES_DATA. The first stage does '
              f'not use these files, so its result above stands.')
        return None
    head = load_head(hp)
    trunk = Trunk(tp)
    head.check_trunk(trunk)
    where = os.path.normpath(os.path.abspath(examples))
    if where.startswith(os.getcwd() + os.sep):
        where = os.path.relpath(where)
    print(f'\nsecond stage, the example HepMC files in {where} through parsing, the thrust '
          f'frame and {stage}')
    good = True
    for s in ex['samples']:
        s = str(s)
        path = os.path.join(examples, str(ex[f'{s}_file']))
        F, M, NB = features_from_events(parse_hepmc(path))
        F0, M0, NB0 = ex[f'{s}_particles'], ex[f'{s}_mask'], ex[f'{s}_nbaryon']
        n = len(F0)
        if F.shape != F0.shape or not np.array_equal(M, M0) or not np.array_equal(NB, NB0):
            print(f'{s:>12}  {str(ex[f"{s}_generator"]):<22} {len(F)} events against {n} stored, '
                  f'or a different particle selection  FAIL')
            good = False
            continue
        d = np.abs(F.astype(np.float64) - F0)
        free = float(d[..., AXIS_FREE].max())
        moved = d[..., 1:3].max(axis=(1, 2)) > TOL_FEATURE
        allowed = int(np.ceil(MOVED*n))
        A = trunk.embed(trunk_features(F), M)
        w = head.ratio_weights(A, ex[f'{s}_theta_to'], ex[f'{s}_theta_from'])
        dl = np.log(w) - np.log(ex[f'{s}_weights'])
        dl -= np.median(dl[~moved]) if (~moved).any() else 0.0
        dlogw = float(np.abs(dl[~moved]).max()) if (~moved).any() else float('nan')
        why = [r for r, bad in (('z, mass or species differ', free >= TOL_FEATURE),
                                ('too many axes moved', moved.sum() > allowed),
                                ('log weights differ', not dlogw < TOL_LOGW)) if bad]
        same = 'bit-identical' if not d.any() else f'max|d|={float(d.max()):.1e}'
        extra = f' (their max|dlogw|={float(np.abs(dl[moved]).max()):.2f})' if moved.any() else ''
        print(f'{s:>12}  {str(ex[f"{s}_generator"]):<22} events={n:<4} features {same:<14} '
              f'axis moved in {int(moved.sum())} of at most {allowed}{extra}  '
              f'max|dlogw|={dlogw:.1e}  N_eff/N={head.n_eff(w)/n:.3f}  '
              f'{"FAIL, " + ", ".join(why) if why else "PASS"}')
        good &= not why
    return good


def main(argv):
    models = argv[1] if len(argv) > 1 else 'models'
    examples = argv[2] if len(argv) > 2 else EXAMPLES
    tags = sorted(os.path.basename(p)[:-len('_selftest.npz')]
                  for p in glob.glob(f'{models}/*_selftest.npz'))
    if not tags:
        print(f'no self-test bundles in {models}')
        return 2
    print(f'gentune self-test, {len(tags)} stage(s) in {models}')
    bad = [t for t in tags if not check(t, models)]
    if bad:
        print(f'\nFAILED: {bad}. Do not use these weights. The most likely cause is a numpy '
              f'or file version mismatch, so check the sha256 values in MANIFEST.json first.')
        return 1
    print('\nall stages reproduce the shipped logits to better than 1e-3')
    ok2 = check_examples(models, examples)
    if ok2 is False:
        print(f'\nFAILED: the example HepMC files do not give the stored features and weights. '
              f'The networks passed above, so the difference is in reading the files or in '
              f'building the thrust frame, which is the path your own events take.')
        return 1
    if ok2:
        print(f'\nthe example files reproduce their features and their weights to {TOL_LOGW:g} '
              f'in log weight')
    else:
        print('\nthe networks pass, and the reading of HepMC files was not tested')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
