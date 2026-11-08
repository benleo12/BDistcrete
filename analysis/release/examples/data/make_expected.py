#!/usr/bin/env python3
"""Make expected.npz, the outputs the second stage of gentune.selftest compares against.

    cd analysis/release
    python examples/data/make_expected.py               # compares with expected.npz, writes nothing
    python examples/data/make_expected.py --write       # writes expected.npz
    python examples/data/make_expected.py --out x.npz   # writes x.npz instead

For each example HepMC file this stores what gentune makes of it: the per-particle features with
their mask and baryon count, a sha256 of those three arrays, and the per-event weights of the
MIX17 head for one from/to pair. The pairs move one shower coupling and nothing else:

    herwig   from the Herwig default tune at fraction 1 (pure Herwig), alpha_fsr 0.102337 -> 0.110
    sherpa   from the Sherpa default tune at fraction 0 (pure Sherpa), alphas    0.118    -> 0.122

At fraction 1 the MIX17 head is Herwig alone and the Sherpa block drops out of the logit exactly,
and at fraction 0 the Herwig block does, so the other generator's block is set to its own default
tune and has no effect. These are the only fractions at which the paper evaluates the head
(gentune.mixture). Both points of both pairs pass head.check.

Without --write or --out it computes everything, compares it with the expected.npz next to it,
array by array, and writes nothing. An array of numbers agrees when it differs by at most TOL,
1e-6 of its largest value; every other array must be identical. It exits 0 when all agree and 1
when one does not, or when expected.npz is missing. Another numpy or another machine can move the
log weights by up to about 1e-7, through the float32 sums of the trunk; that agrees here, and the
self-test allows much more. numpy_version only records which numpy wrote the file and is not
compared. The hash of the features must match only where the features are the same bit for bit.
On another machine the thrust axis of an event close to a tie can land elsewhere; the self-test
allows that, but it shows here as a difference. So this is a check for the authors' machine, and
the self-test is the check of an install.

Write expected.npz only when the example files or the MIX17 files (head or trunk) change.
Rewriting it to make a failing self-test pass defeats the self-test.
"""
import argparse, hashlib, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from gentune import load_head, Trunk, features

EXPECTED = os.path.join(HERE, 'expected.npz')
TOL = 1e-6            # an array of numbers agrees when it differs by at most TOL times its largest value
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


def make(models):
    """Everything expected.npz holds, as a dict of arrays."""
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
    return out


def compare(out, path):
    """Print how out compares with the file path, array by array. Returns the names that do not
    agree (see the docstring), all of them when the file is missing; an empty list means that the
    file holds what this install makes of the example files."""
    if not os.path.exists(path):
        print(f'{path} does not exist, so there is nothing to compare with')
        return list(out)
    old = np.load(path)
    name = os.path.basename(path)
    same, near, differ = set(), [], []
    for k, v in out.items():
        v = np.asarray(v)
        if k not in old.files:
            print(f'  {k}: not in {name}')
            differ.append(k)
            continue
        o = old[k]
        if v.shape == o.shape and v.dtype == o.dtype and np.array_equal(v, o):
            same.add(k)
        elif k == 'numpy_version':
            print(f'  {k}: {o} there, {v} here (a record, not compared)')
        elif k.endswith('_sha256'):
            # the hash of the features: it must match where they are the same bit for bit
            s = k[:-len('_sha256')]
            if {f'{s}_particles', f'{s}_mask', f'{s}_nbaryon'} <= same:
                print(f'  {k}: differs, although the features it hashes are the same')
                differ.append(k)
            else:
                print(f'  {k}: differs, as the features do')
        elif v.shape == o.shape and v.dtype == o.dtype and v.dtype.kind in 'fc':
            d = float(np.max(np.abs(v - o)))
            scale = float(np.max(np.abs(o)))
            good = d <= TOL*scale
            print(f'  {k}: largest difference {d:.1e}, largest value {scale:.3g}: '
                  f'{"agrees" if good else "differs"}')
            (near if good else differ).append(k)
        else:
            print(f'  {k}: differs ({o.dtype}{list(o.shape)} {o.ravel()[:1]} there, '
                  f'{v.dtype}{list(v.shape)} {v.ravel()[:1]} here)')
            differ.append(k)
    extra = [k for k in old.files if k not in out]
    for k in extra:
        print(f'  {k}: in {name} only')
    differ += extra
    if differ:
        print(f'{len(differ)} array(s) do not agree with those of {path}: {", ".join(differ)}')
    elif len(same) == len(out):
        print(f'every array equals that of {path}, bit for bit')
    else:
        print(f'every array agrees with that of {path} to {TOL:g} of its largest value')
    return differ


def main(argv=None):
    head_line, rest = __doc__.split('\n', 1)
    ap = argparse.ArgumentParser(description=head_line, epilog=rest,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    to = ap.add_mutually_exclusive_group()
    to.add_argument('--write', action='store_true',
                    help='write examples/data/expected.npz, the file the self-test reads')
    to.add_argument('--out', metavar='PATH', help='write PATH instead, and leave expected.npz alone')
    ap.add_argument('--models', default=os.path.join(HERE, '..', '..', 'models'),
                    help='the folder of the MIX17 files (default models/ of the release)')
    a = ap.parse_args(argv)
    out = make(a.models)
    path = a.out or (EXPECTED if a.write else None)
    if path is None:
        bad = compare(out, EXPECTED)
        print('nothing written: --write writes expected.npz, --out PATH another file')
        return 1 if bad else 0
    path = path if path.endswith('.npz') else path + '.npz'     # as numpy names it
    np.savez_compressed(path, **out)
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
