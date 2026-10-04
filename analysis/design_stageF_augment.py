#!/usr/bin/env python3
"""Stage F augmentation: the off-locus Herwig points the existing design is missing.

The existing 64 Herwig points lie along a six-dimensional locus, because the six published
retunes the ranges came from move the shower coupling and the shower cutoff together, and the
cluster fission mass and power together. That was a defensible choice about where TUNES land,
and it turned out to be the wrong choice about where TRAINING POINTS should go. The head is
asked about the whole eight-parameter box, so it needs information in all eight directions,
and along the two it barely saw it fits a slope 2.05 times steeper than along the rest where
the isotropic Sherpa design gives 0.59. Measured consequence: the charged multiplicity misses
by 3.4 percent at the held point furthest off the locus, and the mixture inherits the same
pathology in its Herwig block, 2.10 against 0.73 for its Sherpa block.

The separation this gets wrong is worth stating, because it is the general lesson. A design
must span what the head will be ASKED about. A prior says where you think the answer lies.
Folding the prior into the design leaves the head unable to answer the questions the prior
happens to disfavour, and there is no way to put a different prior on it afterwards. So these
points deliberately go where no published tune sits: high coupling with a low cutoff, and the
reverse. They are perfectly runnable Herwig configurations, and restricting to the locus
remains available later as a prior rather than being baked in.

Placement is greedy E-optimal: at each step take the candidate that most raises the SMALLEST
eigenvalue of the design covariance, which is precisely the direction the head is guessing in.
That reaches the Sherpa box's conditioning with 48 points where uniform box-filling does not
get there with 64.

Every point stays inside the existing box, so nothing can cross a Herwig hard limit: the
coupling floor of 0.10 that killed 35 runs of an earlier campaign is below the box edge of
0.1005 by construction.

Writes stageF_augment_design.csv with run ids from 8100, clear of the existing 8000 block.
"""
import numpy as np
import pandas as pd

AX = ['alpha_fsr', 'ptmin', 'clmax', 'clpow', 'psplit', 'pwtsquark', 'pwtdiquark', 'clsmr']
N_NEW = 48
N_CAND = 4000
SEED = 20260920
RID0 = 8100
# Herwig hard limits, probed directly on the 7.3.0p1 build we generate with. The design box
# must sit inside these or runs die at `read` rather than producing events.
LIMITS = {'alpha_fsr': (0.10, None)}


def stats(Z):
    ev = np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T))
    return float(np.sqrt(ev.max()/ev.min())), float(ev.sum()**2/(ev**2).sum()), np.sqrt(ev)


def main():
    d = pd.read_csv('stageF_design.csv')
    lo, hi = d[AX].min().values, d[AX].max().values
    mid, half = (lo + hi)/2, (hi - lo)/2
    Z0 = (d[d.role == 'train'][AX].values - mid)/half
    a0, e0, sd0 = stats(Z0)
    print(f'existing design: {len(Z0)} points, anisotropy {a0:.2f}, effective dims {e0:.2f}')
    print(f'  weakest two directions have design sd {sd0[0]:.3f} and {sd0[1]:.3f} '
          f'against {sd0[-1]:.3f} for the widest')

    rng = np.random.default_rng(SEED)
    CAND = rng.uniform(-1, 1, size=(N_CAND, len(AX)))
    Z = Z0.copy()
    chosen = []
    for _ in range(N_NEW):
        ev, V = np.linalg.eigh(np.cov((Z - Z.mean(0)).T))
        proj = np.abs((CAND - Z.mean(0)) @ V[:, :2]).sum(1)
        best, bi = -np.inf, None
        for i in np.argsort(-proj)[:40]:
            if i in chosen:
                continue
            zt = np.vstack([Z, CAND[i]])
            m = float(np.linalg.eigvalsh(np.cov((zt - zt.mean(0)).T)).min())
            if m > best:
                best, bi = m, i
        chosen.append(bi)
        Z = np.vstack([Z, CAND[bi]])

    a, e, sd = stats(Z)
    print(f'augmented: {len(Z)} points, anisotropy {a:.2f}, effective dims {e:.2f}')
    print(f'  weakest two now {sd[0]:.3f} and {sd[1]:.3f}')
    print(f'  for reference the Sherpa box is 1.11 and 7.97')
    assert a < 1.5, f'augmentation did not fix the conditioning (anisotropy {a:.2f})'

    phys = mid + Z[len(Z0):]*half
    out = pd.DataFrame(phys, columns=AX)
    out.insert(0, 'role', 'train')
    out.insert(0, 'run_id', np.arange(RID0, RID0 + len(out)))
    for i, a_ in enumerate(AX):
        assert out[a_].min() >= lo[i] - 1e-12 and out[a_].max() <= hi[i] + 1e-12, \
            f'{a_} left the existing box'
        if a_ in LIMITS:
            floor, ceil = LIMITS[a_]
            if floor is not None:
                assert out[a_].min() > floor, (
                    f'{a_} minimum {out[a_].min():.6f} is at or below the Herwig limit {floor}')
    assert not set(out.run_id) & set(d.run_id), 'run ids collide with the existing design'
    out.to_csv('stageF_augment_design.csv', index=False)
    print(f'\nwrote stageF_augment_design.csv: {len(out)} runs, ids '
          f'{out.run_id.min()} to {out.run_id.max()}')
    print(out[AX].describe().loc[['min', 'max']].round(4).to_string())
    # how far off the locus these actually sit, which is the whole point
    for k1, k2 in (('alpha_fsr', 'ptmin'), ('clmax', 'clpow')):
        i1, i2 = AX.index(k1), AX.index(k2)
        s1 = (out[k1].values - mid[i1])/half[i1]
        s2 = (out[k2].values - mid[i2])/half[i2]
        oldc = np.corrcoef(Z0[:, i1], Z0[:, i2])[0, 1]
        print(f'  {k1}/{k2}: correlation {oldc:+.3f} in the existing design, '
              f'{np.corrcoef(s1, s2)[0,1]:+.3f} in the new points')


if __name__ == '__main__':
    main()
