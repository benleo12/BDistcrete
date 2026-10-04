#!/usr/bin/env python3
"""The perturbative covariance of the calculated moments, Eq. (sigmac) of the paper.

Of the eleven variations, the scale variations of the log-R scheme and the rescaling of the
third-order term come in up-and-down pairs, and the half-sum of their squared displacements counts
each pair as one standard deviation. The difference between the modified-R and the log-R scheme at
central scales is a single variation and enters once at full weight. The four scale variations of
the modified-R scheme are displaced from the log-R central value as well, so including them would
count the scheme difference two and a half times more. They enter the envelope of the refitted
parameters, not this covariance.

    from theory_cov import sigma_pert
    S = sigma_pert(central, scale, labels)        # central (..., nk), scale (..., nv, nk)
"""
import numpy as np

PAIRS = [("(2.0, 1.0, 'logR', 1.0)", "(0.5, 1.0, 'logR', 1.0)"),
         ("(1.0, 2.0, 'logR', 1.0)", "(1.0, 0.5, 'logR', 1.0)"),
         ("(1.0, 1.0, 'logR', 1.05)", "(1.0, 1.0, 'logR', 0.95)")]
SCHEME = "(1.0, 1.0, 'modR', 1.0)"


def sigma_pert(central, scale, labels):
    labels = [str(l) for l in labels]; idx = {l: i for i, l in enumerate(labels)}
    central = np.asarray(central, float); scale = np.asarray(scale, float)
    nk = central.shape[-1]
    S = np.zeros(central.shape[:-1] + (nk, nk))
    for a, b in PAIRS:
        for v in (a, b):
            d = scale[..., idx[v], :] - central
            S += 0.5*d[..., :, None]*d[..., None, :]
    d = scale[..., idx[SCHEME], :] - central
    S += d[..., :, None]*d[..., None, :]
    return S


def band(displacements, labels):
    """Half-width of the band of any prediction from its displacements under the eleven variations
    (re-solved, in the order of labels), combined like sigma_pert: the pairs at half weight, the
    scheme difference once. displacements has the variations on its first axis."""
    labels = [str(l) for l in labels]; idx = {l: i for i, l in enumerate(labels)}
    d = np.asarray(displacements, float)
    v = sum(0.5*d[idx[x]]**2 for a, b in PAIRS for x in (a, b)) + d[idx[SCHEME]]**2
    return np.sqrt(v)
