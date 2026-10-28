#!/usr/bin/env python3
"""Write expected.npz, the outputs the second stage of gentune.selftest compares against.

    cd analysis/release && python examples/data/make_expected.py

For each example HepMC file this stores what gentune makes of it: the per-particle features with
their mask and baryon count, a sha256 of those three arrays, and the per-event weights of the
MIX17 head for one from/to pair. The pairs move one shower coupling and nothing else:

    herwig   from the Herwig default tune at fraction 1 (pure Herwig), alpha_fsr 0.102337 -> 0.110
    sherpa   from the Sherpa default tune at fraction 0 (pure Sherpa), alphas    0.118    -> 0.122

At fraction 1 the Sherpa block of a geometric mixture drops out of the logit exactly, and at
fraction 0 the Herwig block does, so the other generator's block is set to its own default tune
and has no effect. Both points of both pairs pass head.check.

Run it only when the example files or the MIX17 files change. Rerunning it to make a failing
self-test pass defeats the self-test.
"""
import hashlib, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from gentune import load_head, Trunk, features

# The default tunes of Sherpa 3.0.4 and Herwig 7.3.0, in the order of axes.names('MIX17').
# These are the values each generator reports for itself, and the training cards set them
# explicitly (README.md here says how they were checked).
SHERPA = [0.118, 0.68, 3.9, 0.48, 0.46, 0.17, 0.97, 0.18]
HERWIG = [0.102337, 0.654714, 3.528693, 1.849375, 0.914156, 0.374094, 0.33107, 0.78]

SAMPLES = {
    'herwig': ('herwig_default.hepmc.gz', 'Herwig 7.3.0, HepMC2',
               SHERPA + HERWIG + [1.0], SHERPA + [0.110] + HERWIG[1:] + [1.0]),
    'sherpa': ('sherpa_default.hepmc.gz', 'Sherpa 3.0.4, HepMC3',
               SHERPA + HERWIG + [0.0], [0.122] + SHERPA[1:] + HERWIG + [0.0]),
}


def feature_sha256(F, M, NB):
    """One hash of the three arrays the trunk is fed, so a bit-identical rebuild can say so."""
    h = hashlib.sha256()
    for a in (np.asarray(F, np.float32), np.asarray(M, bool), np.asarray(NB, np.int32)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    models = os.path.join(HERE, '..', '..', 'models')
    head = load_head(os.path.join(models, 'MIX17_head.npz'))
    trunk = Trunk(os.path.join(models, 'MIX17_trunk.npz'))
    head.check_trunk(trunk)
    out = dict(stage='MIX17', samples=np.array(list(SAMPLES)), numpy_version=np.__version__)
    for s, (fname, gen, t_from, t_to) in SAMPLES.items():
        for t in (t_from, t_to):
            head.check(t)
        events = features.parse_hepmc(os.path.join(HERE, fname))
        F, M, NB = features.features_from_events(events)
        A = trunk.embed(features.trunk_features(F), M)
        logw = (head.logit(A, t_to) - head.logit(A, t_from))/head.T
        w = head.ratio_weights(A, t_to, t_from)
        out.update({f'{s}_file': fname, f'{s}_generator': gen,
                    f'{s}_particles': F, f'{s}_mask': M, f'{s}_nbaryon': NB.astype(np.int32),
                    f'{s}_sha256': feature_sha256(F, M, NB),
                    f'{s}_theta_from': np.array(t_from), f'{s}_theta_to': np.array(t_to),
                    f'{s}_logw': logw, f'{s}_weights': w})
        print(f'{s}: {len(events)} events, {M.sum()/len(events):.1f} particles per event, '
              f'N_eff/N = {head.n_eff(w)/len(w):.3f}, sha256 {out[f"{s}_sha256"][:16]}')
    path = os.path.join(HERE, 'expected.npz')
    np.savez_compressed(path, **out)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
