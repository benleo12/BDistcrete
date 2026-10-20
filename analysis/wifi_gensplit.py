#!/usr/bin/env python3
"""Where does the fitted basis weights' loss gain come from?

Every training run of the mixture is a pure run of one generator. On the wifi fit set (wifi_embed.py)
the denominator at a run's point is drawn from the pooled fit set of all runs, so about half of it
comes from the other generator. This splits the loss of wifi_fit.py,

    L = mean_num(-z + exp(-z) - 1) + mean_den(z + exp(z) - 1),

into the numerator, the denominator events of the run's own generator and those of the other
generator, for the published weights (mean of members) and the fitted ones, and gives the mean log
ratio of each denominator group.

    python wifi_gensplit.py output/wifi_<TAG>_basis.npz [fit file, default output/wifi_<TAG>_fit.npz]
Writes output/wifi_gensplit_<fit file name>.json."""
import os, sys, json
import numpy as np

bas = sys.argv[1]
fitf = sys.argv[2] if len(sys.argv) > 2 else bas.replace('_basis.npz', '_fit.npz')
TAG = os.path.basename(fitf).replace('wifi_', '').replace('.npz', '')
B = np.load(bas); F = np.load(fitf)
rids, theta = B['rids'], B['theta']
xi = {int(r): float(t[-1]) for r, t in zip(rids, theta)}                 # 0 Sherpa, 1 Herwig
src = rids[B['den_eid'] // 10**6]                                          # the run each pooled event came from
gen_den = np.array([xi[int(r)] for r in src]); gen_run_den = np.array([xi[int(r)] for r in B['den_rid']])
same = gen_den == gen_run_den
X = lambda f: np.column_stack([np.ones(len(f)), f.astype(np.float64)])
Xn, Xd = X(B['num_f']), X(B['den_f'])
out = dict(tag=TAG, fraction_other=float(1 - same.mean()))
for name, w in (('published', F['w_published']), ('fitted', F['w'])):
    zn, zd = Xn @ w, Xd @ w
    ln = -zn + np.exp(-zn) - 1; ld = zd + np.exp(zd) - 1
    out[name] = dict(loss=float(ln.mean() + ld.mean()), num=float(ln.mean()),
                     den_same=float(ld[same].sum()/len(ld)), den_other=float(ld[~same].sum()/len(ld)),
                     mean_z_same=float(zd[same].mean()), mean_z_other=float(zd[~same].mean()),
                     sd_z_same=float(zd[same].std()), sd_z_other=float(zd[~same].std()),
                     weight_other=float(np.exp(zd[~same]).sum()/np.exp(zd).sum()))
    o = out[name]
    print(f'[{TAG}] {name:9s}: loss {o["loss"]:+.4f} = numerator {o["num"]:+.4f} + own generator {o["den_same"]:+.4f} '
          f'+ other generator {o["den_other"]:+.4f};  mean z on the denominator: own {o["mean_z_same"]:+.2f} (sd {o["sd_z_same"]:.2f}), '
          f'other {o["mean_z_other"]:+.2f} (sd {o["sd_z_other"]:.2f}); share of the weight on the other generator {o["weight_other"]:.2e}')
d = {k: out['fitted'][k] - out['published'][k] for k in ('loss', 'num', 'den_same', 'den_other')}
print(f'[{TAG}] gain of the fitted weights: {d["loss"]:+.4f} = numerator {d["num"]:+.4f} + own generator {d["den_same"]:+.4f} '
      f'+ other generator {d["den_other"]:+.4f}')
out['gain'] = d
json.dump(out, open(f'output/wifi_gensplit_{TAG}.json', 'w'), indent=1)
