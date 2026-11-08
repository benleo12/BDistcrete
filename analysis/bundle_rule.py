#!/usr/bin/env python3
"""Which refreshed bundles the beam-photon rule in use made (D4, E14).

The observables task refreshes every cache of RUN/output, and lmx_runs/release_ref/reference_MIX17.npz, under
the beam-photon rule of beam_photons.py, and then lists each one in RUN/output/bundle_rule.json:

    {"rule_version": 3, "beam_photons_md5": "<md5 of RUN/beam_photons.py>", "written": "<time>",
     "bundles": {"<real path>": {"sha256": "...", "beam_photons_md5": "...", "rule_version": 3, "kind": "slim"}}}

check(path) is called by every reader of a refreshed bundle. It does nothing when BEAM_PHOTONS is unset, empty or
keep, and stops (RuntimeError) on any other value but remove, as score_obs.py does. Under remove it stops unless the
real path of the bundle is listed there with the md5 of the beam_photons.py this process imports, so no bundle made
under another rule, or never refreshed, is read. RUN is MAXENT_DIR, else the folder of this file.

    python -B bundle_rule.py --verify [--hash]      # every listed bundle exists and has the rule of RUN/beam_photons.py
    python -B bundle_rule.py --check <bundle> ...   # check() on each, as under BEAM_PHOTONS=remove

Both exit 1 on any problem. --hash also compares the sha256 of every listed bundle with the table (it reads them all).
"""
import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join('output', 'bundle_rule.json')


def run_dir():
    return os.environ.get('MAXENT_DIR') or HERE


def _digest(path, algo):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def _table(run):
    f = os.path.join(run, TABLE)
    if not os.path.exists(f):
        raise RuntimeError(f'{f} is missing: the observables task writes it after it refreshes the bundles (E14)')
    try:
        return f, json.load(open(f))
    except ValueError as e:
        raise RuntimeError(f'{f} is not a complete table ({e})') from e


_RULE = None


def rule_md5():
    """(file, md5) of the beam_photons.py this process imports, the rule its observables follow (read once)."""
    global _RULE
    if _RULE is None:
        try:
            import beam_photons
        except ImportError as e:
            raise RuntimeError('BEAM_PHOTONS=remove needs beam_photons.py next to the scripts') from e
        f = os.path.abspath(beam_photons.__file__)
        _RULE = (f, _digest(f, 'md5'))
    return _RULE


def _check(path):
    f, t = _table(run_dir())
    real = os.path.realpath(path)
    e = (t.get('bundles') or {}).get(real)
    if e is None:
        raise RuntimeError(f'BEAM_PHOTONS=remove, but {path} ({real}) is not listed in {f}: it is not a bundle '
                           f'refreshed under the beam-photon rule')
    bp, have = rule_md5()
    if e.get('beam_photons_md5') != have:
        raise RuntimeError(f'{path} was refreshed with the beam_photons.py of md5 {e.get("beam_photons_md5")}, but this '
                           f'process imports {bp}, md5 {have}: refresh the bundle, or use that rule')
    return path


def check(path):
    """path, if it may be read: always when BEAM_PHOTONS is unset, empty or keep; under remove only if it is listed
    with the rule this process imports; any other value of BEAM_PHOTONS is refused (else RuntimeError)."""
    mode = os.environ.get('BEAM_PHOTONS') or 'keep'
    if mode not in ('keep', 'remove'):
        raise RuntimeError(f'BEAM_PHOTONS={mode!r}: use keep (as published) or remove (D4)')
    if mode == 'keep':
        return path
    return _check(path)


def verify(run, hash=False):
    """The problems of RUN/output/bundle_rule.json, [] if none: the table lists at least one bundle and carries the
    md5 and rule version of RUN/beam_photons.py; every listed bundle exists under its real path with that md5 and
    version; with hash, its sha256 is the table's."""
    try:
        f, t = _table(run)
    except RuntimeError as e:
        return [str(e)]
    bp = os.path.join(run, 'beam_photons.py')
    if not os.path.exists(bp):
        return [f'{bp} is missing']
    want = _digest(bp, 'md5')
    m = re.search(r'^RULE_VERSION\s*=\s*(\d+)', open(bp).read(), re.M)
    version = int(m.group(1)) if m else None
    bad = []
    if t.get('beam_photons_md5') != want:
        bad.append(f'{f}: beam_photons_md5 {t.get("beam_photons_md5")}, but {bp} has md5 {want}')
    if t.get('rule_version') != version:
        bad.append(f'{f}: rule_version {t.get("rule_version")}, but {bp} has RULE_VERSION {version}')
    if not t.get('bundles'):
        bad.append(f'{f} lists no bundle')
    for p, e in sorted((t.get('bundles') or {}).items()):
        if not os.path.isfile(p):
            bad.append(f'{p}: missing'); continue
        if os.path.realpath(p) != p:
            bad.append(f'{p}: listed under a path that is not its real path ({os.path.realpath(p)})')
        if e.get('beam_photons_md5') != want:
            bad.append(f'{p}: made with beam_photons.py md5 {e.get("beam_photons_md5")}, not {want}')
        if e.get('rule_version') != version:
            bad.append(f'{p}: rule_version {e.get("rule_version")}, not {version}')
        if hash and _digest(p, 'sha256') != e.get('sha256'):
            bad.append(f'{p}: its sha256 is not the one in {f}')
    return bad


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verify', action='store_true', help='check every bundle the table lists')
    ap.add_argument('--hash', action='store_true', help='with --verify: compare the sha256 of every bundle too')
    ap.add_argument('--check', nargs='+', metavar='BUNDLE', help='check() each, as under BEAM_PHOTONS=remove')
    a = ap.parse_args()
    if not (a.verify or a.check):
        ap.error('give --verify or --check')
    run, bad = run_dir(), []
    if a.verify:
        bad += verify(run, a.hash)
        try:
            n = len(_table(run)[1].get('bundles') or {})
        except RuntimeError:
            n = 0
        print(f'{os.path.join(run, TABLE)}: {n} bundles listed{", sha256 compared" if a.hash else ""}')
    for p in a.check or []:
        try:
            _check(p)
        except (RuntimeError, OSError, ValueError) as e:
            bad.append(str(e))
    for b in bad:
        print('BAD', b)
    print(f'BUNDLE RULE: {len(bad)} problem(s)' if bad else 'BUNDLE RULE OK')
    sys.exit(1 if bad else 0)
