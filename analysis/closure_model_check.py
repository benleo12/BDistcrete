"""For every published closure record (Table 1 models, standard and strict): the null expectation of the
quantity the calibration and the checkpoint choice optimize (master = mean over held points of the mean over
observables of the WIDTH; objective |master - 1|), from the observables' kept bin counts, and whether the
temperature or the selected checkpoint could change if the target were that null instead of 1.
E[width] for B kept bins: equal-probability asymptotics sqrt(B/(B-1)) * sqrt(2/k) Gamma((k+1)/2)/Gamma(k/2),
k = B-1 (checked against null_sim.json for the actual binnings)."""
import json, os, sys, glob
from math import lgamma, exp, sqrt
D = sys.argv[1]; OUT = sys.argv[2]

def Ew(B):
    k = B - 1; return sqrt(B/(B - 1))*sqrt(2.0/k)*exp(lgamma((k + 1)/2) - lgamma(k/2))

def load(p):
    d = json.load(open(p)); return d if 'master' in d else list(d.values())[0]

files = {}
for S, pre in (('A', 'p2augfull'), ('B', 'p2augfull'), ('C', 'p2augfull'), ('E', 'p2v3full'), ('F', 'p2v3full'), ('MIX', 'p2v3full')):
    for net in ('fac', 'cat'):
        for k in range(3):
            for strict in ('', 'strict'):
                cand = [f'{D}/{pre}{strict}_{S}_{net}_s{k}.json']
                if S in 'ABC': cand.append(f'{D}/p2aug{strict}_{S}_{net}_s{k}.json')
                if S == 'MIX' and net == 'fac' and not strict:
                    cand = [f'{D}/stage_mixture_{t}.json' for t in [['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2'][k]]]
                if S == 'MIX' and net == 'fac' and strict:
                    cand = [f'{D}/stage_mixture_{t}_strict.json' for t in [['MIXGEOV3S9', 'MIXGEOV3S9s1', 'MIXGEOV3S9s2'][k]]] + \
                           [f'{D}/p2v3fullstrict_MIX_fac_s{k}.json']
                p = next((c for c in cand if os.path.exists(c)), None)
                if p: files[(S, net, k, strict or 'standard')] = p
rows = []
for key, p in sorted(files.items()):
    d = load(p); nb = d.get('nbins_used') or {}
    obs = list(d['per_obs'])
    t = sum(Ew(nb[o]) for o in obs)/len(obs) if nb and all(o in nb for o in obs) else float('nan')
    traj = d.get('stop_trajectory') or []
    sw = d.get('stop_width')
    T = d.get('temperature')
    # checkpoint choice on |ms - 1| (as trained) versus |ms - t|
    if traj:
        c1 = min(traj, key=lambda r: abs(r[1] - 1.0)); ct = min(traj, key=lambda r: abs(r[1] - t))
    else:
        c1 = ct = None
    rows.append(dict(model=key, file=os.path.basename(p), obs=obs, nbins=nb, null_master=t, temperature=T,
                     stop_width_at_T=sw, T_could_change=(sw is not None and sw < t),
                     ckpt_traj_last=traj[-1] if traj else None, ckpt_on_1=c1, ckpt_on_null=ct,
                     ckpt_changes=(c1 is not None and c1[0] != ct[0]), best_step=d.get('best_step'),
                     master=d['master'], per_obs=d['per_obs'],
                     per_obs_over_null={o: d['per_obs'][o]/Ew(nb[o]) for o in obs if o in nb},
                     calibration_check=d.get('calibration_check')))
json.dump(rows, open(OUT, 'w'), indent=1)
print(f"{'model':32s} {'null':>6s} {'master':>7s} {'T':>6s} {'stopw':>7s} {'Tmay':>5s} {'ckpt':>5s}  nbaryon_bins  per_obs/null(nbaryon)")
for r in rows:
    nbb = r['nbins'].get('nbaryon'); rb = r['per_obs_over_null'].get('nbaryon')
    print(f"{str(r['model']):32s} {r['null_master']:6.4f} {r['master']:7.4f} {str(r['temperature']):>6s} "
          f"{(r['stop_width_at_T'] or float('nan')):7.4f} {str(r['T_could_change']):>5s} {str(r['ckpt_changes']):>5s}  {str(nbb):>4s}  "
          f"{'' if rb is None else f'{rb:.3f}'}")
