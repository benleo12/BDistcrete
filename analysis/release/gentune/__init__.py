"""Continuous generator-parameter reweighting: released heads and the code to use them.

    from gentune import Head, Trunk, features

    head  = Head('models/E_head.npz')            # Sherpa, eight parameters
    trunk = Trunk('models/E_trunk.npz')
    events = features.parse_hepmc('my_sample.hepmc.gz')
    F, M, _ = features.features_from_events(events)
    A = trunk.embed(features.trunk_features(F), M)
    w = head.ratio_weights(A, theta_to, theta_from)     # per event weights

Run `python -m gentune.selftest models` first. It proves your install reproduces the logits
the shipped files produce, which takes a few seconds and rules out every silent mismatch.
`load_head(path)` returns the right class for any export, a single generator or a mixture.
"""
from .head import Head, MixtureHead, MIXTURE_KINDS, load_head
from .trunk import Trunk
from . import features
from . import axes

__all__ = ['Head', 'MixtureHead', 'MIXTURE_KINDS', 'load_head', 'Trunk', 'features', 'axes']
