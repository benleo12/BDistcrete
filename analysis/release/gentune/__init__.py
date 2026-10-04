"""Continuous generator-parameter reweighting: released heads and the code to use them.

    from gentune import Head, Trunk, features

    head  = Head('models/E_head.npz')            # Sherpa, eight parameters
    trunk = Trunk('models/E_trunk.npz')
    events = features.parse_hepmc('my_sample.hepmc.gz')
    F, M, _ = features.features_from_events(events)
    A = trunk.embed(features.trunk_features(F), M)
    w = head.ratio_weights(A, theta_to, theta_from)     # per event weights

Run `python -m gentune.selftest models` first. It proves your install reproduces the logits
the training run produced, which takes a few seconds and rules out every silent mismatch.
"""
from .head import Head, MixtureHead
from .trunk import Trunk
from . import features
from . import axes

__all__ = ['Head', 'MixtureHead', 'Trunk', 'features', 'axes']
