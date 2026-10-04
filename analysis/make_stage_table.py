#!/usr/bin/env python3
"""Generate release/STAGES.md: what each released head is, and how well it closes.

Every number in the table is measured here rather than transcribed. For each released stage
and each parameter point that was HELD OUT of training, the reference sample is reweighted to
that point and the mean of every closure observable is compared with the mean the generator
itself produced there. The comparison is quoted as a relative shift and as a pull, using the
held run's statistical error and the weighted sample's error added in quadrature.

The pull is the honest number and it is not kind: on a million reference events a model error
of a few parts in a thousand on a hadron rate is several standard deviations. That is the
point of quoting both, since the relative shift says how wrong the model is and the pull says
whether the sample can tell.

Usage: python make_stage_table.py [<stage> ...]     (default: everything in the manifest)
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'release')
from r2_ladder import STAGES, strange_frac
from mixture_cfg import register as _register_mixture
from gentune.head import Head, MixtureHead
from gentune import axes

REL = os.environ.get('RELEASE_DIR', 'release')
ALIAS = {'C_1M': 'C', 'Faug': 'F'}
PRETTY = {'1_minus_thrust': '1 - T', 'mult_total': 'total multiplicity', 'B_total': 'broadening',
          'rho_heavy': 'heavy jet mass', 'nbaryon': 'baryons', 'strange': 'strange fraction'}



def design_key(tag, stages, alias):
    """Which registered stage's design belongs to this export tag.

    Export names are a stage key followed by LADDER_SUFFIX, so 'Faug' and 'Fauglong' are both
    stage F on some design, and the reverse mapping is the longest registered key that prefixes
    the tag. Listing the tags one at a time instead is how 'Faug' got an entry and 'Fauglong' did
    not, which silently shipped a head with no design and made the measurement script report on
    one of the two heads it was given.
    """
    if tag in stages:
        return tag
    if tag in alias:
        return alias[tag]
    cands = [k for k in stages if tag.startswith(k) and len(tag) > len(k)]
    return max(cands, key=len) if cands else None

def obs_of(cfg, data, rid, kind):
    if kind in ('nbaryon', 'strange'):
        d = np.load(f'{data}/particles_full_{rid:04d}.npz')
        return (d['nbaryon'].astype(float) if kind == 'nbaryon'
                else strange_frac(d['particles'], d['mask']).astype(float))
    return pd.read_csv(f'{data}/shapes_run_{rid:04d}.csv')[kind].values.astype(float)


def closure(tag):
    dtag = design_key(tag, STAGES, ALIAS)
    if dtag not in STAGES and _register_mixture(STAGES, [tag]) is not None:
        dtag = tag
    if dtag not in STAGES:
        return None, f'no design for {tag}'
    cfg = STAGES[dtag]
    data = cfg['data']
    cp = f'output/models/{tag}_cond.npz'
    rp = f'output/models/{tag}_ref.npz'
    if not (os.path.exists(cp) and os.path.exists(rp)):
        return None, f'the reference bundle for {tag} is not on this machine'
    cond = np.load(cp)
    # An unrecognised tag falls through to the mixture branch above, which registers a
    # 17-parameter config under it. Feeding that to an 8-parameter head raised a broadcast error
    # deep inside in_box(), which says nothing about the cause. Check the dimensions here.
    if int(cond['ntheta']) != cfg['ntheta']:
        return None, (f'{tag} is an {int(cond["ntheta"])} parameter head but design {dtag!r} has '
                      f'{cfg["ntheta"]} parameters. Add {tag!r} to ALIAS, or point DM_DATA at the '
                      f'right mixture data set.')
    HK = MixtureHead if str(cond.get('head_kind', 'cond')) == 'mixture' else Head
    h = HK(cp)
    r = np.load(rp)
    OBS = cfg['obs'] + cfg['flav_obs']
    ref = {}
    for o in OBS:
        ref[o] = (r[o].astype(float) if o in r.files
                  else (r['mask'].sum(1).astype(float) if o == 'mult_total' else None))
    rows = []
    # Held-out runs that duplicate a training run event for event (the Stage C ladder runs set no
    # random seed, so held-out 6902 at the grid centre repeats training run 6813) test nothing.
    DUPLICATE_HELD = {6704, 6902}
    for rid, th in cfg['held'].items():
        if rid in DUPLICATE_HELD:
            continue
        if not os.path.exists(f'{data}/particles_full_{rid:04d}.npz'):
            continue
        w = h.weights(h.AE, th)
        neff = h.n_eff(w)
        for o in OBS:
            if ref[o] is None:
                continue
            t = obs_of(cfg, data, rid, o)
            mt, et = float(t.mean()), float(t.std()/np.sqrt(len(t)))
            mw = float(np.sum(w*ref[o]))
            vw = float(np.sum(w*(ref[o] - mw)**2))
            ew = float(np.sqrt(vw/neff))
            rows.append(dict(run=rid, obs=o, generator=mt, reweighted=mw,
                             rel=100.0*(mw - mt)/mt if mt else np.nan,
                             pull=(mw - mt)/np.hypot(et, ew), neff_frac=neff/len(w),
                             n_held=len(t)))
    return pd.DataFrame(rows), None


def main(tags):
    """tags are export names, optionally EXPORT=PUBLIC_NAME.

    closure() recomputes from output/models/<tag>_cond.npz, the EXPORT, while the section
    heading, the closure CSV and the summary key are the PUBLIC name. When a retrained export is
    published under an older stage's name those two differ, and asking for the public name alone
    recomputed the OLD head and wrote its numbers under the new head's name. It really did: stage
    F was documented at 3.43 percent, which belongs to the 64-run head, after the 112-run head
    had replaced it.
    """
    man = json.load(open(f'{REL}/models/MANIFEST.json'))
    rename = {}
    clean = []
    for t in (tags or []):
        if '=' in t:
            a, b = t.split('=', 1)
            rename[a] = b
            clean.append(a)
        else:
            clean.append(t)
    explicit = list(clean)
    tags = clean or sorted(man)
    pub = lambda t: rename.get(t, t)
    # STAGES.md is a whole-bundle document and this writes it in full, so naming a subset of the
    # stages on the command line replaces the documentation of the others. That is how the shipped
    # file came to cover three of six, and the assertion at the end cannot see it, because a
    # one-stage run is self-consistent. Refuse when the target is a real release directory.
    subset = [t for t in man if t not in {pub(x) for x in tags}]
    if explicit and subset and os.path.exists(f'{REL}/STAGES.md'):
        allow = os.environ.get('STAGE_TABLE_SUBSET', '') == '1'
        msg = (f'refusing to rewrite {REL}/STAGES.md for {sorted(tags)} alone, because the '
               f'manifest also ships {sorted(subset)} and this writes the whole file. Run with no '
               f'arguments to cover every stage, or point RELEASE_DIR at a scratch directory to '
               f'measure one stage without touching the bundle. Set STAGE_TABLE_SUBSET=1 if '
               f'truncating it is really what you want.')
        if not allow:
            print(msg)
            return 2
        print(f'NOTE {msg}')
    out = ['# Released stages', '',
           'Generated by `make_stage_table.py`. Every closure number is measured on parameter '
           'points held out of training. A section says so when its numbers were recomputed on '
           'this machine, and when they were read back from the shipped '
           '`closure_<stage>.csv` because the reference bundle for that stage lives elsewhere.',
           '']
    summary = {}
    tabled = []
    for tag in tags:
        m = man.get(pub(tag), {})
        out += [f'## {pub(tag)}', '',
                f'- {m.get("ntheta", "?")} parameters, {m.get("head_kind", "?")} head, '
                f'rank {m.get("K_bilinear", "?")}, {m.get("ens", "?")} ensemble members',
                f'- reference sample {m.get("n_reference_events", 0):,} events, '
                f'calibration temperature {m.get("temperature", float("nan")):.3f}',
                f'- own events supported: {"yes" if m.get("has_trunk") else "no, reference sample only"}',
                '', '| parameter | name | low | high | centre |', '|---|---|---|---|---|']
        # A bare index tells a reader nothing. gentune.axes holds the names in the order the head
        # stores its rows, and refuses to supply them for a head whose length disagrees.
        try:
            anames = axes.names(pub(tag), int(m.get('ntheta', 0)))
        except Exception as e:
            print(f'[{pub(tag)}] no parameter names: {e}')
            anames = tuple(f'{i}' for i in range(int(m.get('ntheta', 0))))
        for i, (b, c) in enumerate(zip(m.get('box', []), m.get('centre', []))):
            out.append(f'| {i} | `{anames[i]}` | {b[0]:.4f} | {b[1]:.4f} | {c:.4f} |')
        df, err = closure(tag)
        fresh = df is not None and not df.empty
        if not fresh:
            # The reference bundle for this stage is not on this machine, so closure cannot be
            # recomputed. The shipped per-point CSV is the record of the run that did compute it,
            # and it carries every column these tables need. Reading it back is what stops
            # STAGES.md from silently documenting only the stages whose bundles happen to be
            # local, which is how it came to cover three of six shipped heads.
            cf = f'{REL}/closure_{pub(tag)}.csv'
            if os.path.exists(cf):
                df = pd.read_csv(cf)
        out.append('')
        if df is None or df.empty:
            out += [f'Closure not available: {err or "no held runs present"}, and no '
                    f'`closure_{pub(tag)}.csv` ships either.', '']
            continue
        out += ['Closure on parameter points held out of training, reference reweighted against '
                'the generator'
                + ('' if fresh else f', read back from `closure_{pub(tag)}.csv` rather than '
                                    f'recomputed here') + ':', '',
                '| observable | worst relative shift | worst pull | held points |', '|---|---|---|---|']
        for o in df.obs.unique():
            s = df[df.obs == o]
            i = s.rel.abs().idxmax()
            j = s.pull.abs().idxmax()
            out.append(f'| {PRETTY.get(o, o)} | {s.loc[i, "rel"]:+.2f}% | '
                       f'{s.loc[j, "pull"]:+.1f} | {len(s)} |')
        out += ['',
                f'Effective sample fraction across the held points: '
                f'{df.neff_frac.min():.2f} to {df.neff_frac.max():.2f}.', '']
        tabled.append(pub(tag))
        summary[pub(tag)] = dict(worst_rel=float(df.rel.abs().max()),
                            worst_pull=float(df.pull.abs().max()),
                            n_points=int(df.run.nunique()))
        if fresh:
            df.to_csv(f'{REL}/closure_{pub(tag)}.csv', index=False)
        print(f'{pub(tag)}: worst |shift| {df.rel.abs().max():.2f}%, worst |pull| '
              f'{df.pull.abs().max():.1f}, {df.run.nunique()} held points'
              f'{"" if fresh else " (from the shipped CSV)"}')
    # Every stage whose section carries a closure TABLE must appear in the summary, or the two
    # shipped files describe different sets of stages, which is exactly what they did. A section
    # that says outright that no closure is available is a different case and is allowed: it
    # documents the box and says the closure is not there, which is honest and is what happens
    # when neither a reference bundle nor a shipped CSV is present. Asserting over every section
    # instead aborted the whole run and wrote nothing.
    missing = [t for t in tabled if t not in summary]
    assert not missing, (f'{missing} have a closure table but no closure_summary entry. The two '
                         f'shipped files must cover the same stages.')
    sections = [l[3:].strip() for l in out if l.startswith('## ')]
    silent = [t for t in sections if t not in summary]
    if silent:
        print(f'NOTE {silent} are documented with a box but no closure, because neither a '
              f'reference bundle nor a shipped closure CSV is present for them')
    open(f'{REL}/STAGES.md', 'w').write('\n'.join(out) + '\n')
    json.dump(summary, open(f'{REL}/closure_summary.json', 'w'), indent=1)
    print(f'wrote {REL}/STAGES.md and closure_summary.json covering {len(sections)} stage(s): '
          f'{", ".join(sections)}')
    print(f'wrote {REL}/STAGES.md')


if __name__ == '__main__':
    main(sys.argv[1:])
