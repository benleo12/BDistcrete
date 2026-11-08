"""Continuous generator-parameter reweighting: released heads and the code to use them.

    from gentune import Head, Trunk, features

    head  = Head('models/E_head.npz')            # Sherpa, eight parameters
    trunk = Trunk('models/E_trunk.npz')
    events = features.parse_hepmc('examples/data/sherpa_default.hepmc.gz')
    F, M, _ = features.features_from_events(events)
    A = trunk.embed(features.trunk_features(F), M)
    w = head.ratio_weights(A, theta_to, theta_from)     # per event weights, in file order

Run `python -m gentune.selftest models` first. It proves your install reproduces the logits
the shipped files produce, which rules out every silent mismatch.
`load_head(path)` returns the right class for any export, a single generator or a mixture.

MIX17 holds Sherpa and Herwig together. The paper's prediction at one of its points is the mixture
(1 - xi) <O>_S + xi <O>_H of the two generators' predictions, each from its own weights, and
`mixture` computes it: mixture.weights(head, A, theta) and mixture.mean(w, O, theta).

The modules: `features` reads HepMC files and builds what the trunk reads, `axes` names the
parameters and gives their settings and defaults, `mixture` gives the paper's predictions of
MIX17, `bands` has the pieces of the paper's bands (its points, per-event observables, statistical
errors, the thrust bins, the theory band), `beam_photons` finds the photons radiated by the beams,
which the paper leaves out of every observable, and `maxent` imposes a calculation on the
reweighted reference sample (IMPOSE.md). All of them need numpy alone.
"""
from .head import Head, MixtureHead, MIXTURE_KINDS, GeometricFractionWarning, load_head
from .trunk import Trunk
from . import features
from . import axes
from . import maxent
from . import bands
from . import mixture

__all__ = ['Head', 'MixtureHead', 'MIXTURE_KINDS', 'GeometricFractionWarning', 'load_head', 'Trunk',
           'features', 'axes', 'bands', 'maxent', 'beam_photons', 'mixture']


def __getattr__(name):
    # beam_photons is imported on first use, so that python -m gentune.beam_photons runs it as a
    # script without the package having imported it first
    if name == 'beam_photons':
        import importlib
        return importlib.import_module(f'{__name__}.beam_photons')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
