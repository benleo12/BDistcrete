#!/usr/bin/env python3
"""One alpha_0 column of the direct profile, for running columns in parallel.

At the column's first coupling node the starting points come from a cheap scan: chi^2 without
gradients on a Latin hypercube over the standardized box, then L-BFGS-B from the best few. The
other nodes of the column start from the neighbouring node's optimum, and from any optima other
columns have already found at the same coupling (STARTS_FROM). polish_direct.py then merges the
columns and restarts every node from all four neighbours.

    AE_DTYPE=float32 ONLY_A0=0.43 python profile_column.py MIX17aug
    STARTS_FROM=output/profile_MIX17aug_central_a00.43.json ONLY_A0=0.39 python profile_column.py MIX17aug

LAST_BIN=0.3334 USE_NCH=0 fits only the thrust bins inside the window, normalized over those bins.
MIX_FORM=geometric combines the two generators of a mixture head as q_S^(1-f) q_H^f.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, '.')
from directlib import Model

tag = sys.argv[1]
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
which = os.environ.get('TARGETS', 'central')
a0 = float(os.environ['ONLY_A0'])
M = Model(tag, export=os.environ.get('EXPORT_PATH'), ref=os.environ.get('REF_PATH'),
          first_bin=float(os.environ.get('FIRST_BIN', '0.05')), floor_rel=float(os.environ.get('FLOOR_REL', '0.002')),
          nch_data=tuple(float(x) for x in os.environ.get('NCH_DATA', '18.63,0.11').split(',')),
          a0_stride=int(os.environ.get('A0_STRIDE', '2')), ae_dtype=os.environ.get('AE_DTYPE', 'float64'), grid=os.environ.get('TARGETS_GRID', 'output/thrust_targets_grid.npz'),
          last_bin=float(os.environ['LAST_BIN']) if os.environ.get('LAST_BIN') else None, use_nch=os.environ.get('USE_NCH', '1') == '1',
          mix_form=os.environ.get('MIX_FORM', 'additive'))
j = [k for k, x in enumerate(M.A0G) if abs(x - a0) < 1e-9]
assert j, f'alpha_0 = {a0} is not on the grid'
jj = M.j_of[j[0]]
OUT = os.environ.get('PROFILE_OUT', f'output/profile_{tag}_{which.replace(":", "")}_a0{a0:.2f}.json')
nA = len(M.ASG); t0 = time.time()
others = []
for f in [x for x in os.environ.get('STARTS_FROM', '').split(',') if x]:
    d = json.load(open(f))
    others.append([(np.array(d['profile_theta'][ia][0]) - M.NORM[:, 0])/M.NORM[:, 1] for ia in range(nA)])
mid = int(os.environ.get('FIRST_IA', str(nA//2)))
order = [mid] + list(range(mid + 1, nA)) + list(range(mid - 1, -1, -1))
prof = np.full(nA, np.nan); U = np.zeros((nA, M.NT)); EXT = [None]*nA; nfev = 0
# EXTEND_FROM: a column computed on a narrower coupling grid with identical targets at the shared
# nodes. Its nodes are copied, and only the new couplings are computed, each chained from the
# nearest node already known on its side.
known = set()
if os.environ.get('EXTEND_FROM'):
    d = json.load(open(os.environ['EXTEND_FROM']))
    assert abs(d['alpha0'][0] - a0) < 1e-9, 'the column to extend has a different alpha_0'
    for i_old, a_old in enumerate(d['alphas']):
        i_new = int(np.argmin(np.abs(M.ASG - a_old))); assert abs(M.ASG[i_new] - a_old) < 1e-9
        prof[i_new] = d['profile'][i_old][0]
        U[i_new] = (np.array(d['profile_theta'][i_old][0]) - M.NORM[:, 0])/M.NORM[:, 1]
        EXT[i_new] = d['extras'][i_old][0]; known.add(i_new)
    lo_k, hi_k = min(known), max(known)
    order = list(range(hi_k + 1, nA)) + ([] if os.environ.get('EXTEND_UP_ONLY') == '1' else list(range(lo_k - 1, -1, -1)))
    print(f'extending a column known at alpha_s {M.ASG[lo_k]:.3f}..{M.ASG[hi_k]:.3f}: {len(order)} new nodes', flush=True)
rng = np.random.default_rng(int(1000*a0))
print(f'{tag} {which} alpha_0 {a0}: {M.kind} head, {M.NT} parameters, {M.N:,} events, AE {M.AE.dtype}  [{time.time()-t0:.0f} s]', flush=True)
for k, ia in enumerate(order):
    starts = []
    if known:
        prev = ia - 1 if ia > max(known) else ia + 1
        starts.append(U[prev])
    elif k == 0:
        if others:
            starts += [o[ia] for o in others]
        nscan = int(os.environ.get('NSCAN', '0' if others else '192'))
        if nscan:
            sc = M.scan(ia, jj, nscan, rng, which)
            starts += [u for _, u in sc[:int(os.environ.get('NBEST', '3'))]]
            print(f'  scan of {nscan} points at alpha_s {M.ASG[ia]:.3f}: best chi2 {sc[0][0]:.2f}, '
                  f'median {sc[len(sc)//2][0]:.1f}  [{time.time()-t0:.0f} s]', flush=True)
    if not known and k > 0:
        prev = ia - 1 if ia > mid else ia + 1
        starts.append(U[prev])
        starts += [o[ia] for o in others]
    uniq = []
    for s in starts:
        if not any(np.allclose(s, q) for q in uniq):
            uniq.append(np.asarray(s, float))
    f, u, lam, ne = M.profile_node(ia, jj, uniq, which=which)
    nfev += ne; prof[ia] = f; U[ia] = u
    EXT[ia] = M.extras(u, ia, jj, which, lam)
    print(f'  alpha_s {M.ASG[ia]:.3f}: chi2 {f:.3f} from {len(uniq)} starts, {ne} evaluations, fraction {M.theta(torch.tensor(u))[-1].item() if M.kind == "mixture" else float("nan"):.3f}  [{time.time()-t0:.0f} s]', flush=True)
TH = M.NORM[:, 0] + M.NORM[:, 1]*U
json.dump(dict(tag=tag, kind=M.kind, export=M.export, ref=M.refpath, targets=which, experiments=M.exps, n_data=M.ndat,
               nch_data=M.NCH, first_bin=M.first_bin, last_bin=M.last_bin, use_nch=M.use_nch, mix_form=M.mix_form, floor_rel=M.floor_rel, alphas=M.ASG.tolist(), alpha0=[a0],
               profile=[[float(x)] for x in prof], profile_theta=[[t.tolist()] for t in TH], extras=[[e] for e in EXT],
               nfev=nfev, seconds=time.time() - t0), open(OUT, 'w'), indent=1)
print(f'wrote {OUT} in {(time.time()-t0)/60:.1f} min, {nfev} evaluations', flush=True)
