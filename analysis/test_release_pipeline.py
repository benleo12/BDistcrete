#!/usr/bin/env python3
"""Does the public release compute the same thing the training code computes?

The release reimplements the whole event side in numpy so users need no deep learning
framework. That is a reimplementation, and a reimplementation of a convention is exactly where
a silent difference lives: a reordered feature column or a different azimuth reference changes
every weight while breaking nothing. This compares the release against the code the models
were actually trained with, link by link.

  1. the HepMC parser, in both generators' column orders, gzip included
  2. the thrust-frame features, against extract_particles_full_flavor.py
  3. the seven columns the trunk reads, against r2_ladder.build_feats
  4. the numpy trunk, against the a(Phi) vectors the training run exported
  5. the numpy head, against the logits the training run recorded

Links 4 and 5 need a packaged stage, so they are skipped with a notice if none is present.
Usage: python test_release_pipeline.py [<stage tag>]
"""
import gzip, importlib.util, os, sys, tempfile
import numpy as np

sys.path.insert(0, 'release')
from gentune import features as F
from gentune.head import Head, MixtureHead
from gentune.trunk import Trunk
from r2_ladder import build_feats

spec = importlib.util.spec_from_file_location('orig', 'extract_particles_full_flavor.py')
ORIG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIG)

PIDS = [211, -211, 321, -321, 2212, -2212, 111, 22, 130, 310, 2112, -2112,
        3122, -3122, 11, -11, 13, -13, 3312, -3334, 411, 521]
MASS = {211: 0.1396, 321: 0.4937, 2212: 0.9383, 111: 0.1350, 22: 0.0, 130: 0.4976,
        310: 0.4976, 2112: 0.9396, 3122: 1.1157, 11: 0.000511, 13: 0.1057,
        3312: 1.3217, 3334: 1.6725, 411: 1.8696, 521: 5.279}
fails = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}{("   " + detail) if detail else ""}')
    if not ok:
        fails.append(name)


def make_event(rng, n):
    pid = rng.choice(PIDS, n)
    m = np.array([MASS[abs(int(p))] for p in pid])
    p3 = rng.normal(0, 3, (n, 3))
    p3[:, 2] *= 4.0                      # two-jet-like, so the thrust axis is well defined
    return np.column_stack([np.sqrt((p3**2).sum(1) + m**2), p3, pid, m])


def test_parser():
    rows = [(211, 1.0, 2.0, 3.0, 3.75, 0.1396, 1),
            (14, 0.5, 0.5, 9.0, 9.03, 0.0, 1),        # neutrino, must be dropped
            (2212, -1.0, -2.0, -8.0, 8.30, 0.9383, 1),
            (321, 0.2, 0.1, 0.5, 0.74, 0.4937, 2)]    # not final state, must be dropped
    # HepMC3 writes pid at field 3 and status at 9, HepMC2 at 2 and 8, so the layout has to be
    # known. The files carry their real headers, because that is what fmt='auto' reads: choosing
    # by counting fields instead is what made an auto-detected HepMC2 file parse to nothing.
    fmts = {
        'hepmc3': ('auto', ['HepMC::Version 3.02.05\n', 'HepMC::Asciiv3-START_EVENT_LISTING\n',
                            'E 0 0 0\n'] + [f'P 1 0 {p} {a} {b} {c} {e} {m} {s} 0 0 0\n'
                                            for p, a, b, c, e, m, s in rows]),
        'hepmc2 auto': ('auto', ['HepMC::Version 2.06.09\n',
                                 'HepMC::IO_GenEvent-START_EVENT_LISTING\n',
                                 'E 0 0 0\n'] + [f'P 1 {p} {a} {b} {c} {e} {m} {s} 0 0 0\n'
                                                 for p, a, b, c, e, m, s in rows]),
        'herwig alias': ('herwig', ['E 0 0 0\n'] + [f'P 1 {p} {a} {b} {c} {e} {m} {s} 0 0 0\n'
                                                    for p, a, b, c, e, m, s in rows]),
    }
    want = np.array([[3.75, 1.0, 2.0, 3.0, 211.0, 0.1396],
                     [8.30, -1.0, -2.0, -8.0, 2212.0, 0.9383]])
    with tempfile.TemporaryDirectory() as td:
        for name, (fmt, lines) in fmts.items():
            p = f'{td}/{name.replace(chr(32), chr(95))}.hepmc'
            open(p, 'w').write(''.join(lines))
            a = F.parse_hepmc(p, fmt=fmt)
            b = ORIG.parse_hepmc(p, fmt=fmt)
            with gzip.open(p + '.gz', 'wt') as fh:
                fh.write(''.join(lines))
            az = F.parse_hepmc(p + '.gz', fmt=fmt)
            check(f'parser, {name} column order',
                  len(a) == 1 and np.allclose(a[0], want) and np.array_equal(a[0], b[0]),
                  f'{len(a[0])} of 4 particles kept')
            check(f'parser, {name} gzip', np.array_equal(a[0], az[0]))


def test_features(ntrial=60):
    rng = np.random.default_rng(11)
    wf = wm = wb = 0.0
    for _ in range(ntrial):
        ev = make_event(rng, int(rng.integers(1, 140)))   # spans the 100 particle truncation
        fa, ma, ba = F.event_features(ev)
        fb, mb, bb = ORIG.full_event_particles(ev, F.P_MAX)
        wf = max(wf, float(np.max(np.abs(fa - fb))))
        wm = max(wm, float(np.max(np.abs(ma.astype(int) - mb.astype(int)))))
        wb = max(wb, abs(ba - bb))
    check('thrust-frame features against the extractor', wf == 0 and wm == 0 and wb == 0,
          f'{ntrial} events, max|df|={wf:.1e}')
    P = np.stack([F.event_features(make_event(rng, int(rng.integers(20, 120))))[0]
                  for _ in range(40)])
    M = np.stack([F.event_features(make_event(rng, int(rng.integers(20, 120))))[1]
                  for _ in range(40)])
    tf, tm = build_feats(P, M, True)
    d = float(np.max(np.abs(F.trunk_features(P)*M[..., None] - tf.numpy())))
    check('trunk columns against r2_ladder.build_feats', d == 0, f'max|df|={d:.1e}')
    check('mask against r2_ladder.build_feats',
          np.array_equal(tm.numpy(), M.astype(np.float32)))


def test_packaged(tag):
    md = 'release/models'
    hp = f'{md}/{tag}_head.npz'
    sb = f'{md}/{tag}_selftest.npz'
    if not (os.path.exists(hp) and os.path.exists(sb)):
        print(f'  SKIP  packaged checks: {tag} is not in {md}')
        return
    c = np.load(hp)
    head = (MixtureHead if str(c.get('head_kind', 'bilinear')) == 'mixture' else Head)(hp)
    s = np.load(sb)
    tp = f'{md}/{tag}_trunk.npz'
    if os.path.exists(tp):
        # Two precisions, for the reason documented in package_release.py: a stage trained
        # through the memmap cache stored its features as float16, so its exported reference
        # vectors carry that quantization while a user works in float32. What must hold
        # exactly is that this numpy trunk IS the torch network, which the float16 round trip
        # tests. The float32 number is the size of the quantization gap, reported not tested.
        trunk = Trunk(tp)
        Fr = F.trunk_features(s['particles'])
        ref = np.asarray(s['AE'], np.float64)
        sd = float(np.std(ref))

        # Judge on the RMS, not the maximum. A maximum over thousands of events is set by
        # the single worst-quantized particle: float16 spacing at a feature magnitude of 14
        # is 0.008, so events holding a very soft or very collinear particle quantize far
        # worse than typical, and the max wanders with which events are sampled. The RMS
        # answers the question actually being asked, whether the two implementations are the
        # same network, and is stable across event subsets.
        def embed(feats):
            A = trunk.embed(feats, s['mask'])
            if int(c['additive']):
                A = np.concatenate([A, np.ones(A.shape[:2] + (1,), A.dtype)], -1)
            return A

        def gap(A):
            D = np.asarray(A, np.float64) - ref
            return float(np.sqrt(np.mean(D**2)))/sd, float(np.max(np.abs(D)))/sd

        A = embed(Fr)                      # the float32 user path, reused by the logit check
        r32, m32 = gap(A)
        r16, m16 = gap(embed(Fr.astype(np.float16).astype(np.float32)))
        check(f'{tag}: numpy trunk is the same network as the torch one',
              min(r32, r16) < 1e-4,
              f'rms {r32:.1e} float32 / {r16:.1e} via float16, '
              f'max {max(m32, m16):.1e} sd (float16 storage gap)')
    else:
        A = s['AE']
        print(f'  SKIP  {tag}: no trunk in the release, so the event side is not exercised')
    f = np.stack([head.logit(A, t) for t in s['theta_probe']])
    d = float(np.max(np.abs(f - s['logit_probe'])))
    check(f'{tag}: numpy head against the recorded logits', d < 1e-3,
          f'max|df| = {d:.1e} over {len(s["theta_probe"])} points')


def test_mixture(tag):
    """The mixture claims the README makes: the block split, and exactness at the edges.

    Writing the logit as log[(1-f) exp(l_S) + f exp(l_H)] makes the derivative with respect to
    one generator's parameters vanish IDENTICALLY when its weight is zero, not approximately.
    That is what lets a prior put weight at f = 0 or f = 1 without fighting the
    parameterization, so it is worth testing rather than trusting.
    """
    hp = f'release/models/{tag}_head.npz'
    if not os.path.exists(hp):
        print(f'  SKIP  {tag}: not packaged')
        return
    if str(np.load(hp).get('head_kind', 'cond')) != 'mixture':
        print(f'  SKIP  {tag}: not a mixture head')
        return
    h = MixtureHead(hp)
    check(f'{tag}: block sizes sum to the parameter count',
          h.nS + h.nH + 1 == h.nt, f'{h.nS} + {h.nH} + 1 = {h.nt}')
    tS, tH, _ = h.split(h.centre)
    check(f'{tag}: split returns the two blocks',
          len(tS) == h.nS and len(tH) == h.nH)
    sb = np.load(f'release/models/{tag}_selftest.npz')
    tp = f'release/models/{tag}_trunk.npz'
    A = (Trunk(tp).embed(F.trunk_features(sb['particles']), sb['mask'])
         if os.path.exists(tp) else sb['AE'])
    for fv, silent, active in ((0.0, range(h.nS, h.nS + h.nH), range(h.nS)),
                               (1.0, range(h.nS), range(h.nS, h.nS + h.nH))):
        t0 = h.centre.copy()
        t0[-1] = fv
        base = h.logit(A, t0)
        eps = 1e-4
        def sweep(idx):
            w = 0.0
            for j in idx:
                tq = t0.copy()
                tq[j] += eps*(h.box[j, 1] - h.box[j, 0])
                w = max(w, float(np.max(np.abs(h.logit(A, tq) - base))))
            return w
        ds, da = sweep(silent), sweep(active)
        check(f'{tag}: at f={fv:.0f} the silent generator does not move the logit',
              ds < 1e-11 and da > 1e-6, f'silent {ds:.1e}, active {da:.1e}')


def main():
    print('release pipeline, against the code the models were trained with')
    test_parser()
    test_features()
    for tag in (sys.argv[1:] or sorted({os.path.basename(p).split('_head.npz')[0]
                                        for p in __import__('glob').glob('release/models/*_head.npz')})):
        test_packaged(tag)
        test_mixture(tag)
    print(f'\n{"ALL CHECKS PASS" if not fails else "FAILURES: " + ", ".join(fails)}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
