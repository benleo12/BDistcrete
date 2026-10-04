#!/usr/bin/env python3
"""Build a mixture stage's configuration from the data set's own meta.json.

stage_mixture.py constructs this at training time and registers it under a fixed key, so
scripts that run LATER against an exported mixture head, the packager and the closure table,
cannot find the design by looking in r2_ladder.STAGES. Keeping the construction in one place
means the box and the axis order cannot drift between the run that trained the head and the
scripts that describe it.
"""
import json, os


def mixture_cfg(data=None):
    """(config, meta) for the mixture data set at `data`, or None if it is not present."""
    data = data or os.environ.get('DM_DATA', 'data_stageDM17')
    path = f'{data}/meta.json'
    if not os.path.exists(path):
        return None, None
    meta = json.load(open(path))
    if 's_box' not in meta:
        return None, meta          # an older data set, built before the box was recorded
    d = meta['ntheta']
    norm = [(float((lo + hi)/2), float((hi - lo)/2)) for lo, hi in meta['s_box'] + meta['h_box']]
    norm.append((0.5, 0.5))        # the fraction is physical on [0, 1]
    cfg = dict(data=data, flavor=True, ntheta=d,
               train={m['rid']: tuple(m['theta']) for m in meta['train']},
               held={m['rid']: tuple(m['theta']) for m in meta['held']},
               norm=norm,
               obs=['1_minus_thrust', 'mult_total', 'B_total', 'rho_heavy'],
               flav_obs=['nbaryon', 'strange'])
    assert len(norm) == d, f'{len(norm)} normalizations for {d} parameters'
    return cfg, meta


def register(stages, tags, data=None):
    """Put the mixture config into a STAGES dict under every tag given, so a script can look
    up an exported head by the tag its files carry."""
    cfg, meta = mixture_cfg(data)
    if cfg is None:
        return None
    for t in tags:
        stages[t] = cfg
    return cfg


if __name__ == '__main__':
    cfg, meta = mixture_cfg()
    if cfg is None:
        print('no mixture data set found' + (' with a recorded box' if meta else ''))
    else:
        print(f'd={cfg["ntheta"]}, {len(cfg["train"])} train, {len(cfg["held"])} held, '
              f'axes {meta["s_axes"]} + {meta["h_axes"]} + fraction')
