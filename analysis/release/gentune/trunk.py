"""The event side of an exported head, in pure numpy.

The trained model has two halves. b(theta), the parameter side, lives in gentune.head. This
module is the other half: the per-particle network Phi, masked sum pooling, and the map A to
the K-vector that gets contracted with b. Running it turns YOUR events into the a(Phi) vectors
the head needs, so you are not restricted to the reference events shipped with the export.

It is a Particle Flow Network (Komiske, Metodiev and Thaler, arXiv:1810.05165) at the
published sizes, so the forward pass is four matrix products and an activation, and numpy does
it exactly. The weights are read from the <tag>_trunk.npz written by package_release.py, which
is a plain conversion of the torch checkpoint, so no deep learning framework is needed here.

The pooled embedding is divided by a constant fixed at initialization (emb_scale), one value
per ensemble member. It is part of the model, not a preprocessing choice, so it travels in the
npz and is applied here.

So does the precision of the inputs. Stages trained through the memmap cache saw their
per-particle features stored as float16, and their trunk file says feature_precision = float16.
normalize() then rounds your features the same way before anything else, so the network reads
exactly what it was trained on. Reading them in float32 instead moves a reweighted mean of MIX17
by up to 0.17 percent, the size of its accuracy floor (analysis/precision_gap.py). A file without
the field is a float32 stage and is read as before.
"""
import numpy as np


def _act(z, name):
    if name == 'silu':
        return z*(0.5*(1.0 + np.tanh(0.5*z)))
    if name == 'relu':
        return np.maximum(z, 0.0)
    raise ValueError(f'unknown activation {name}')


class Trunk:
    def __init__(self, path, feature_precision=None):
        """feature_precision overrides the file's own value, 'float32' or 'float16'. It exists to
        measure the difference the rounding makes, and a user has no reason to set it."""
        t = np.load(path)
        self.path = path
        self.feature_precision = feature_precision or (
            str(t['feature_precision']) if 'feature_precision' in t.files else 'float32')
        assert self.feature_precision in ('float32', 'float16'), self.feature_precision
        self.act = str(t['act'])
        self.ENS = int(t['ens'])
        self.K = int(t['K'])
        self.C = int(t['C'])
        self.feat_mu = t['feat_mu'].astype(np.float32)
        self.feat_sd = t['feat_sd'].astype(np.float32)
        self.emb_scale = t['emb_scale_used'].astype(np.float64)
        self.nphi = int(t['n_phi_layers'])
        self.nA = int(t['n_A_layers'])
        self.phi = [[(t[f'phi{m}_W{l}'].astype(np.float32), t[f'phi{m}_b{l}'].astype(np.float32))
                     for l in range(self.nphi)] for m in range(self.ENS)]
        self.A = [[(t[f'A{m}_W{l}'].astype(np.float32), t[f'A{m}_b{l}'].astype(np.float32))
                   for l in range(self.nA)] for m in range(self.ENS)]

    def normalize(self, feats, mask):
        """Standardize the per-particle features with the statistics the model was trained
        with, then re-zero the padded slots so pooling still ignores them. Using your own
        sample's statistics here would remove exactly the between-run differences the model
        reads, so the shipped mu and sigma are the only correct choice. A float16 stage first
        rounds the raw features to float16, as its training did."""
        f = np.asarray(feats, np.float32)
        if self.feature_precision == 'float16':
            f = f.astype(np.float16).astype(np.float32)
        f = (f - self.feat_mu)/self.feat_sd
        return f*np.asarray(mask, np.float32)[..., None]

    def _member(self, m, fn, mask):
        x = fn.reshape(-1, fn.shape[-1])
        for W, b in self.phi[m]:
            x = _act(x @ W.T + b, self.act)                  # activation after every Phi layer
        x = x.reshape(fn.shape[0], fn.shape[1], -1)
        E = (x*mask[..., None]).sum(1)/self.emb_scale[m]     # masked SUM pooling, then the scale
        for W, b in self.A[m][:-1]:
            E = _act(E @ W.T + b, self.act)
        W, b = self.A[m][-1]
        return E @ W.T + b                                   # linear output, the a(Phi) vector

    def embed(self, feats, mask, chunk=20000, normalized=False):
        """a(Phi) for every event and every ensemble member, shape (ENS, Nev, K).

        feats is (Nev, P, C) raw per-particle features from gentune.features and mask is
        (Nev, P). Pass normalized=True only if you have already applied normalize(). Events
        are processed in chunks because the intermediate per-particle activations are the
        memory cost: one chunk holds chunk*P*256 floats."""
        feats = np.asarray(feats, np.float32)
        mask = np.asarray(mask, np.float32)
        assert feats.shape[:2] == mask.shape, f'{feats.shape} against {mask.shape}'
        assert feats.shape[-1] == self.C, f'expected {self.C} features, got {feats.shape[-1]}'
        out = np.empty((self.ENS, len(feats), self.K), np.float32)
        for i in range(0, len(feats), chunk):
            fb = feats[i:i+chunk]
            mb = mask[i:i+chunk]
            fn = fb if normalized else self.normalize(fb, mb)
            for m in range(self.ENS):
                out[m, i:i+chunk] = self._member(m, fn, mb)
        return out
