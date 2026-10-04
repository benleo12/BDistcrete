#!/usr/bin/env python3
"""Stage D-mixture data: the seven-parameter family
    q(Phi; theta_S, theta_H, f) = (1-f) q_Sherpa(Phi; theta_S) + f q_Herwig(Phi; theta_H)
with theta_S = (alpha_s, STRANGE_FRACTION, KT_0) on the Sherpa box (rids 9300-9326 train,
9327-9331 held) and theta_H = (AlphaIn, PwtSquark, ClMaxLight) on Herwig's own box
(data_herwigbox rids 9000-9026 train, 9027-9031 held). Every run is a resampled mixture of
one pure run from each side, drawn without replacement per pool, so no event repeats.

Design: 96 training runs of 25k events (d=7, K2=36; Stage C used 2.7 points per mode).
 - 12 at f=0 (pure Sherpa) and 12 at f=1 (pure Herwig). Their silent side's label is drawn
   UNIFORMLY in that side's box: the data do not depend on it, and scattering the labels is
   what lets the network learn the derivative is zero there (checked later, not assumed).
 - 72 at f in {1/8..7/8}, (theta_S, theta_H) pairs balanced over both grids and decorrelated.
10 held-out runs of 80k from the held pools, at interior f values.
Reference: 96 x 3400 = 326k, Stage C's size. Writes data_stageDM/ rids 9700-9795 train,
9930-9939 held, meta.json with every triple."""
import numpy as np, pandas as pd, os, json, itertools
rng = np.random.default_rng(20260902)
DST = os.environ.get('DM_DST', 'data_stageDM'); os.makedirs(DST, exist_ok=True)
S_SRC, H_SRC = os.environ.get('DM_S_SRC', 'data_showerbox'), os.environ.get('DM_H_SRC', 'data_herwigbox')
ASV=[0.112,0.120,0.128]; SFV=[0.30,0.46,0.65]; KTV=[0.80,1.21,1.80]
S_GRID = list(itertools.product(ASV,SFV,KTV)); S_HELD=[(0.116,0.38,1.00),(0.124,0.55,1.50),(0.120,0.46,1.21),(0.114,0.60,0.95),(0.126,0.35,1.60)]
HB = json.load(open(f'{H_SRC}/meta.json'))
H_GRID = [tuple(m['theta']) for m in HB if m['rid'] < 9027]; H_HELD = [tuple(m['theta']) for m in HB if m['rid'] >= 9027]
S_BOX = [(0.112,0.128),(0.30,0.65),(0.80,1.80)]; H_BOX = [(0.1106,0.1266),(0.15,0.50),(2.80,4.80)]
NTR, NHE = 25000, 80000

class Pool:
    """Per-run cursors, draw without replacement, never exceed the run."""
    def __init__(s, src, rid0, n):
        s.src=src; s.perm={}; s.off={}; s.rid0=rid0
    def take(s, idx, n, held=False):
        rid = s.rid0 + idx
        if rid not in s.perm:
            sh = pd.read_csv(f'{s.src}/shapes_run_{rid:04d}.csv'); s.perm[rid]=rng.permutation(len(sh)); s.off[rid]=0
        k = s.perm[rid][s.off[rid]:s.off[rid]+n]; assert len(k)==n, f'pool exhausted rid {rid} (need {n}, have {len(s.perm[rid])-s.off[rid]})'
        s.off[rid]+=n
        d = np.load(f'{s.src}/particles_full_{rid:04d}.npz'); sh = pd.read_csv(f'{s.src}/shapes_run_{rid:04d}.csv')
        return {kk: d[kk][k] for kk in d.files}, sh.iloc[k]

def rand_in(box): return tuple(float(rng.uniform(lo,hi)) for lo,hi in box)

def build_design():
    """96 triples (iS, thS, iH, thH, f) with balanced usage and budget feasibility."""
    runs=[]
    # pure Sherpa / pure Herwig: cycle the grids so every third point appears
    for r in range(12):
        runs.append(dict(iS=(r*7)%27, iH=None, f=0.0, thH=rand_in(H_BOX)))
        runs.append(dict(iS=None, iH=(r*7+3)%27, f=1.0, thS=rand_in(S_BOX)))
    # mixtures: 72 runs, f cycles 1/8..7/8, pairs from shuffled balanced lists
    fs=[ (1+ (r%7))/8 for r in range(72) ]
    iS=[i%27 for i in rng.permutation(72)]; iH=[i%27 for i in rng.permutation(72)]
    for r in range(72): runs.append(dict(iS=iS[r], iH=iH[r], f=float(fs[r])))
    rng.shuffle(runs)
    # budget check: per-point usage
    useS=np.zeros(27); useH=np.zeros(27)
    for m in runs:
        if m['iS'] is not None: useS[m['iS']] += (1-m['f'])*NTR
        if m['iH'] is not None: useH[m['iH']] += m['f']*NTR
    assert useS.max()<=80000 and useH.max()<=80000, (useS.max(), useH.max())
    print(f'design ok: max Sherpa-point usage {useS.max():.0f}/80000, Herwig {useH.max():.0f}/80000')
    return runs

def mix(m, n, sp, hp, held=False):
    nb=int(round(m['f']*n)); na=n-nb; parts=[]
    if na: parts.append(sp.take(m['iS'], na))
    if nb: parts.append(hp.take(m['iH'], nb))
    arrs={k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    sh=pd.concat([p[1] for p in parts]); o=rng.permutation(n)
    return {k:v[o] for k,v in arrs.items()}, sh.iloc[o], na, nb

meta={'train':[], 'held':[]}
sp, hp = Pool(S_SRC, 9300, 80000), Pool(H_SRC, 9000, 80000)
for r, m in enumerate(build_design()):
    rid=9700+r
    thS = list(S_GRID[m['iS']]) if m['iS'] is not None else list(m['thS'])
    thH = list(H_GRID[m['iH']]) if m['iH'] is not None else list(m['thH'])
    arrs, sh, na, nb = mix(m, NTR, sp, hp)
    np.savez_compressed(f'{DST}/particles_full_{rid:04d}.npz', **arrs)
    sh.to_csv(f'{DST}/shapes_run_{rid:04d}.csv', index=False)
    meta['train'].append(dict(rid=rid, theta=thS+thH+[m['f']], n_sherpa=na, n_herwig=nb))
    if r%16==0: print(f'  train rid {rid}: f={m["f"]:.3f} {na}+{nb}  <1-T>={sh["1_minus_thrust"].mean():.5f}', flush=True)
sph, hph = Pool(S_SRC, 9327, 80000), Pool(H_SRC, 9027, 80000)
# 7 held-out runs. Each held pool point holds 80k, so five 80k mixtures use each Sherpa point
# once ((1-f) 80k) and each Herwig point once (f 80k); the two pure endpoints (40k) come from
# the points with the most left over (Sherpa point 4 after f=0.875, Herwig point 2 after f=0.125).
HELD_RUNS = [dict(iS=j, iH=(j+2)%5, f=f, n=NHE) for j, f in enumerate([0.125, 0.375, 0.5, 0.625, 0.875])]
HELD_RUNS += [dict(iS=4, iH=None, f=0.0, n=40000), dict(iS=None, iH=2, f=1.0, n=40000)]
for j, m in enumerate(HELD_RUNS):
    rid=9930+j; f=m['f']
    arrs, sh, na, nb = mix(m, m['n'], sph, hph, held=True)
    thS = list(S_HELD[m['iS']]) if f<1 else list(rand_in(S_BOX)); thH = list(H_HELD[m['iH']]) if f>0 else list(rand_in(H_BOX))
    np.savez_compressed(f'{DST}/particles_full_{rid:04d}.npz', **arrs)
    sh.to_csv(f'{DST}/shapes_run_{rid:04d}.csv', index=False)
    meta['held'].append(dict(rid=rid, theta=thS+thH+[f], n_sherpa=na, n_herwig=nb))
    print(f'  held rid {rid}: f={f:.3f} {na}+{nb}  <1-T>={sh["1_minus_thrust"].mean():.5f}', flush=True)
json.dump(meta, open(f'{DST}/meta.json','w'), indent=1); print('STAGE DM DATA DONE')
