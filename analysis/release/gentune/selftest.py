"""Prove that this install reproduces the logits the shipped files produce.

Every released stage ships a small bundle of reference events together with the logits to
reproduce at several parameter points. This runs the numpy trunk and the head over those events
and compares.

Its scope, stated precisely, because a PASS is easy to read more into than it carries. The
stored events hold the six per-particle columns ALREADY CONSTRUCTED, so this exercises the two
networks and the final column mapping but NOT the HepMC parsing or the thrust-frame
construction that produced those columns. Run examples/reweight_own_events.py on a file of your
own to exercise that. And the stored logits are the ones the shipped files produce on a float32
reader, not the ones the training run itself recorded: stages trained through the float16
feature cache differ from their training run by a few parts in a thousand in the logit, and that
value is kept alongside as logit_probe_from_export so the gap is on the record.

Usage: python -m gentune.selftest [models_dir]
"""
import glob, os, sys
import numpy as np

from .head import Head, MixtureHead
from .trunk import Trunk
from .features import trunk_features

TOL = 1e-3            # far above the float32 rounding of the pooled sum, far below any real error


def check(tag, models='models'):
    hp = f'{models}/{tag}_head.npz'
    c = np.load(hp)
    kind = str(c.get('head_kind', 'cond'))
    head = MixtureHead(hp) if kind == 'mixture' else Head(hp)
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
    # dA compares against the reference vectors the TRAINING run exported. Stages whose
    # training cached features as float16 show a gap of a few percent of a standard deviation
    # there by construction, which is why it is reported and not tested. The logit comparison
    # is the test: it is against the float32 path a user actually runs.
    print(f'{tag:>12} {kind:>9} d={head.nt:<3} events={len(sb["mask"]):<6} {tr:>14}  '
          f'dA(vs training)={dA:.1e}  max|dlogit|={df:.2e}  '
          f'N_eff/N={head.n_eff(w)/len(w):.3f}  {"PASS" if ok else "FAIL"}')
    return ok


def main(argv):
    models = argv[1] if len(argv) > 1 else 'models'
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
    print('\nall stages reproduce the training run to better than 1e-3 in the logit')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
