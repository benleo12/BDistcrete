#!/usr/bin/env python3
"""How much the float16 feature cache of training matters to a user who works in float32.

The networks of E, F and MIX17 were trained through the memmap cache, which stores the
per-particle features as float16, so the reference vectors a(Phi) of the training run carry that
rounding. This embeds a uniform subset of the reference sample both ways with the shipped numpy
trunk, in float32 and rounded to float16, and at every held-out point of the stage compares the
reweighted means of the closure observables with the statistical error of those means on the
full reference sample. The answer is why the trunk file records feature_precision and the
reader rounds a user's features to it: for MIX17 the float32 path moves a mean by up to 0.17
percent, the size of its accuracy floor.

Usage: python precision_gap.py EXPORT=PUBLIC ...   (RELEASE_DIR holds the packaged models)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "release")
from r2_ladder import STAGES
from mixture_cfg import register as _register_mixture
from make_stage_table import design_key, ALIAS
from gentune.head import Head, MixtureHead
from gentune.trunk import Trunk
from gentune.features import trunk_features

REL = os.environ.get("RELEASE_DIR", "release_r8")
NSUB = int(os.environ.get("GAP_NSUB", "200000"))


def gap(tag, pub):
    dtag = design_key(tag, STAGES, ALIAS)
    if dtag not in STAGES and _register_mixture(STAGES, [tag]) is not None:
        dtag = tag
    cfg = STAGES[dtag]
    hp = f"{REL}/models/{pub}_head.npz"
    kind = str(np.load(hp).get("head_kind", "cond"))
    h = (MixtureHead if kind in ("mixture", "geometric") else Head)(hp)
    tr = Trunk(f"{REL}/models/{pub}_trunk.npz", feature_precision="float32")   # rounding by hand below
    r = np.load(f"output/models/{tag}_ref.npz")
    mask = r["mask"]
    N = len(mask)
    sel = np.sort(np.random.default_rng(7).choice(N, min(NSUB, N), replace=False))
    P, M = r["particles"][sel], mask[sel]
    F32 = trunk_features(P)
    A32 = tr.embed(F32, M)
    A16 = tr.embed(F32.astype(np.float16).astype(np.float32), M)
    AEt = np.asarray(np.load(f"output/models/{tag}_cond.npz")["AE"][:, sel, :], np.float64)
    sd = float(np.std(AEt))
    out = dict(kind=kind, n_sub=int(len(sel)), n_ref=int(N),
               a16_vs_training=float(np.max(np.abs(A16 - AEt))/sd),
               a32_vs_training=float(np.max(np.abs(A32 - AEt))/sd))
    ref = {}
    for o in cfg["obs"] + cfg["flav_obs"]:
        ref[o] = (r[o][sel].astype(float) if o in r.files
                  else (M.sum(1).astype(float) if o == "mult_total" else None))
    worst_pull = worst_rel = 0.0
    lg_rms, lg_max = [], []
    for rid, th in cfg["held"].items():
        l32, l16 = h.logit(A32, th)/h.T, h.logit(A16, th)/h.T
        d = l32 - l16
        d = d - d.mean()
        lg_rms.append(float(np.sqrt(np.mean(d**2)))); lg_max.append(float(np.max(np.abs(d))))
        w32, w16 = h.weights(A32, th), h.weights(A16, th)
        neff_frac = h.n_eff(w16)/len(w16)
        for o, x in ref.items():
            if x is None:
                continue
            m32, m16 = float(w32 @ x), float(w16 @ x)
            err = float(np.sqrt((w16 @ (x - m16)**2)/(neff_frac*N)))
            worst_pull = max(worst_pull, abs(m32 - m16)/err)
            worst_rel = max(worst_rel, abs(m32 - m16)/abs(m16) if m16 else 0.0)
    out.update(n_points=len(cfg["held"]), logit_gap_rms=float(np.mean(lg_rms)),
               logit_gap_max=float(np.max(lg_max)), worst_mean_shift_over_stat=worst_pull,
               worst_mean_shift_relative=worst_rel)
    print(pub, json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for a in sys.argv[1:]:
        tag, pub = a.split("=")
        res[pub] = gap(tag, pub)
    json.dump(res, open(f"output/precision_gap_r8.json", "w"), indent=1)
    print("GAP DONE")
