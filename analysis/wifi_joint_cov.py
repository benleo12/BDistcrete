#!/usr/bin/env python3
"""The covariance of the wifi weights with each fit-set event's numerator and denominator contributions
joined (review of 2026-11-10, item 2).

wifi_fit.py estimates U, the covariance of the gradient of L_fit, as the covariance over the numerator rows
plus the covariance over the per-event sums of the denominator rows. The denominator rows are drawn from the
pool of the numerator events themselves (wifi_embed.py: pool_eid = 10^6 * position of the run in `rids` +
row within the run, and the numerator rows are that pool in the same order), so an event's numerator row and
its denominator rows are correlated, and that cross-covariance is left out. This script computes, at the
fitted weights w of the published fit (nothing is refitted):

  pub      wifi_fit.py's U, recomputed (checked against the stored U)
  joint    the covariance over all pooled events of s_e = (numerator row of e) + (sum of its denominator rows)
  strat    as joint, but the covariance taken within each training run and summed over the runs (a fixed
           number of fit events per run)

and C = V^-1 U V^-1 for each, with the stored Hessian V. It writes <out>_<variant>.npz, the published fit
file with C and U replaced (so wifi_coverage.py and three_bands.py take it as their wifi fit), and <out>.json:
the decomposition of U_joint, and the generalized eigenvalues of C_variant against C_pub on the w_ij block
(w_0 drops out of every normalized prediction), which bound the ratio of any learning-band variance.

    python wifi_joint_cov.py output/wifi_<TAG>_basis_rebuilt.npz output/wifi_<TAG>_fit.npz <out prefix>
"""
import sys, json, time
import numpy as np

bas, fitp, out = sys.argv[1:4]
t0 = time.time()
B = np.load(bas); F = np.load(fitp)
w, V, U_st, C_st = F['w'], F['V'], F['U'], F['C']
rids = [int(r) for r in B['rids']]; pos = {r: k for k, r in enumerate(rids)}
num_rid = np.asarray(B['num_rid']).astype(np.int64)
kk = np.array([pos[r] for r in num_rid])
# the numerator rows are the runs' fit events in the order of rids, each run contiguous
assert np.all(np.diff(kk) >= 0), 'numerator rows are not grouped by run in the order of rids'
start = np.searchsorted(kk, np.arange(len(rids)))
num_eid = kk*10**6 + (np.arange(len(kk)) - start[kk])
den_eid = np.asarray(B['den_eid']).astype(np.int64)
assert len(np.unique(num_eid)) == len(num_eid)
ue, inv = np.unique(den_eid, return_inverse=True)
assert np.isin(ue, num_eid).all(), 'a denominator event is not a numerator event'
# where a denominator row is evaluated at its own run's theta, it must equal that event's numerator row
den_rid = np.asarray(B['den_rid']).astype(np.int64)
own = np.array([rids[k] for k in den_eid//10**6]) == den_rid
num_row_of = np.searchsorted(num_eid, den_eid)                   # num_eid is sorted
assert np.array_equal(num_eid[num_row_of], den_eid)
nf, df = B['num_f'], B['den_f']
diff_own = float(np.abs(df[own] - nf[num_row_of[own]]).max()) if own.any() else float('nan')
print(f'{len(num_eid)} numerator rows, {len(den_eid)} denominator rows over {len(ue)} distinct events; '
      f'{int(own.sum())} denominator rows at their own run\'s theta, largest difference to the numerator row {diff_own:.2e} '
      f'[{time.time()-t0:.0f} s]', flush=True)

design = lambda f: np.column_stack([np.ones(len(f)), np.asarray(f, np.float64)])
Xn, Xd = design(nf), design(df); Nn, Nd = len(Xn), len(Xd)
zn, zd = Xn @ w, Xd @ w
an = Xn*(-(1 + np.exp(-zn)))[:, None]/Nn                       # the mlc loss of wifi_fit.py (its default)
bd = Xd*(1 + np.exp(zd))[:, None]/Nd
del Xn, Xd
cov_of_sums = lambda S: S.T @ S - np.outer(S.sum(0), S.sum(0))/len(S)
bde = np.zeros((len(ue), len(w))); np.add.at(bde, inv, bd)      # per distinct denominator event
U_pub = cov_of_sums(an) + cov_of_sums(bde)
rel_pub = float(np.abs(U_pub - U_st).max()/np.abs(U_st).max())
# the joint per-event contributions over all pooled events (an event never drawn for the denominator has none)
bfull = np.zeros_like(an); np.add.at(bfull, np.searchsorted(num_eid, ue), bde)
s = an + bfull
U_joint = cov_of_sums(s)
A, Bc = cov_of_sums(an), cov_of_sums(bfull)
X = (an - an.mean(0)).T @ (bfull - bfull.mean(0))                  # the cross-covariance, both centered over all events
U_strat = np.zeros_like(U_joint)
for k in range(len(rids)):
    a, b = start[k], (start[k + 1] if k + 1 < len(rids) else len(s))
    U_strat += cov_of_sums(s[a:b])
Vi = np.linalg.inv(V)
Cs = {name: Vi @ Uv @ Vi for name, Uv in (('pub', U_pub), ('joint', U_joint), ('strat', U_strat))}
rel_C = float(np.abs(Cs['pub'] - C_st).max()/np.abs(C_st).max())
print(f'U recomputed as in wifi_fit.py: largest relative difference to the stored U {rel_pub:.1e}, to the stored C {rel_C:.1e}', flush=True)


def gen_eig(Cn, Co, block=slice(1, None)):
    """generalized eigenvalues of Cn against Co on a block: the range of g^T Cn g / g^T Co g over g"""
    L = np.linalg.cholesky(Co[block, block]); Li = np.linalg.inv(L)
    return np.linalg.eigvalsh(Li @ Cn[block, block] @ Li.T)


res = dict(basis=bas, fit=fitp, n_num=int(Nn), n_den=int(Nd), n_distinct_den=int(len(ue)),
           den_own_theta_rows=int(own.sum()), den_own_theta_max_diff=diff_own,
           U_pub_vs_stored_rel=rel_pub, C_pub_vs_stored_rel=rel_C,
           trace_ratio={}, gen_eig={}, diag_ratio={})
# how much of U_joint each piece is, on the trace of the w_ij block
tr = lambda M: float(np.trace(M[1:, 1:]))
res['U_trace'] = dict(numerator=tr(A), denominator_all_events=tr(Bc), denominator_as_wifi_fit=tr(cov_of_sums(bde)),
                      cross_twice=tr(X + X.T), joint=tr(U_joint), strat=tr(U_strat), pub=tr(U_pub))
for name in ('joint', 'strat'):
    ev = gen_eig(Cs[name], Cs['pub'])
    res['gen_eig'][name] = dict(min=float(ev.min()), max=float(ev.max()), median=float(np.median(ev)), all=ev.tolist())
    res['trace_ratio'][name] = tr(Cs[name])/tr(Cs['pub'])
    d = np.diag(Cs[name])[1:]/np.diag(Cs['pub'])[1:]
    res['diag_ratio'][name] = dict(min=float(d.min()), max=float(d.max()))
    print(f'{name:6s}: C/C_pub generalized eigenvalues {ev.min():.3f} to {ev.max():.3f} (median {np.median(ev):.3f}), '
          f'trace ratio {res["trace_ratio"][name]:.3f}', flush=True)
ev_js = gen_eig(Cs['strat'], Cs['joint']); res['gen_eig']['strat_vs_joint'] = dict(min=float(ev_js.min()), max=float(ev_js.max()))
print('U trace pieces (w_ij block):', {k: f'{v:.3e}' for k, v in res['U_trace'].items()}, flush=True)
for name in ('joint', 'strat'):
    np.savez(f'{out}_{name}.npz', **{k: F[k] for k in F.files if k not in ('C', 'U')}, C=Cs[name],
             U=U_joint if name == 'joint' else U_strat, C_published=C_st, U_published=U_st, variant=np.array(name))
json.dump(res, open(out + '.json', 'w'), indent=1)
print(f'wrote {out}_joint.npz, {out}_strat.npz and {out}.json [{time.time()-t0:.0f} s]')
