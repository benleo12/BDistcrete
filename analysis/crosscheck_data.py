"""Shared data builder for the PFN cross-check, so both implementations see byte-identical
inputs. Task: Sherpa events at shower coupling 0.08 against 0.14 from the wide scan (runs 9000 and
9010), a classification with real signal, since a comparison of two near-chance classifiers cannot
tell agreement from noise. Features are the same four per-particle quantities the ladder uses.
Padded slots are exactly zero, which is also energyflow's masking convention (mask_val=0).
CROSSCHECK_DATA and CROSSCHECK_PAIR select another sample and pair of runs."""
import numpy as np

import os
DATA = os.environ.get('CROSSCHECK_DATA', 'data_widebox')
NPER = 60000          # events per class
EPS = 1e-6


def build(seed=0, pair=tuple(int(x) for x in os.environ.get('CROSSCHECK_PAIR', '9000,9010').split(','))):
    """The two runs of the pair are the two classes.

    Features are standardized using REAL particles only, then the mask is re-applied so padded
    slots stay exactly zero. That keeps energyflow's mask_val=0 convention valid and stops raw
    log z from destabilizing training in either framework."""
    Xs, ys, Ms = [], [], []
    for lab, rid in [(0, pair[0]), (1, pair[1])]:
        d = np.load(f'{DATA}/particles_full_{rid}.npz')
        p = d['particles'][:NPER]; m = d['mask'][:NPER]
        cos = p[..., 1]; phi = p[..., 2]; z = p[..., 0]
        f = np.stack([cos, phi,
                      np.log(np.clip(z, EPS, None)),
                      np.log(np.clip(1 - cos**2 + EPS, EPS, None))], -1)
        Xs.append(f.astype(np.float32)); Ms.append(m.astype(np.float32))
        ys.append(np.full(len(f), lab, dtype=np.float32))
    X = np.concatenate(Xs); M = np.concatenate(Ms); y = np.concatenate(ys)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, M, y = X[idx], M[idx], y[idx]
    ntr = int(0.8*len(X))
    real = M[:ntr].astype(bool)
    mu = X[:ntr][real].mean(0); sd = X[:ntr][real].std(0) + 1e-9
    X = (X - mu) / sd
    X = X * M[..., None]                                # padded slots exactly zero again
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:]
