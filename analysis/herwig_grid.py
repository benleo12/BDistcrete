#!/usr/bin/env python3
"""Herwig side of the symmetric Stage D': generate and extract Herwig 7.3 runs on the SAME
physical knob grid as the Sherpa box, so the mixture q_f = (1-f) q_Sherpa(theta) + f q_Herwig(theta)
is between two generators evaluated at the same physical point and the knobs keep their grip
at f = 1.

Physical knob -> parameter, verified against the local Herwig repository defaults:
    strong coupling        ALPHAS(MZ)        <-> /Herwig/Shower/AlphaQCD:AlphaIn        (0.1186, 2 loop)
    strangeness            STRANGE_FRACTION  <-> HadronSelector:PwtSquark               (0.291717)
    hadronization scale    KT_0              <-> ClusterFissioner:ClMaxLight            (3.649 GeV)
The coupling maps directly. The other two are different objects in the two hadronization
models, so their RANGES are calibrated (herwig_calibrate) to give the same response in the
observables the Sherpa range moves, rather than assumed equal.

Usage:  python herwig_grid.py scan   <n>            one-parameter scans for the calibration
        python herwig_grid.py grid   <n> [--rid0 R] the 27-point grid + held-out points
Events are generated, streamed straight into showerbox format, and the HepMC deleted, so
disk stays at one run and memory at one chunk."""
import argparse, os, subprocess, sys, gzip, shutil
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from extract_particles_full_flavor import full_event_particles
from compute_efps import compute_hemisphere_shapes

HW = Path(os.environ['HERWIG723_PREFIX']); SHARE = HW/'share'/'Herwig'
WORK = Path('/tmp/herwig_grid'); WORK.mkdir(exist_ok=True)
DEF = dict(alphas=0.1186, pwt=0.291717, clmax=3.649, clpow=2.780, psplit=0.899)
NEUTRINO = {12, 14, 16}

CARD = """read snippets/EECollider.in
cd /Herwig/MatrixElements
insert SubProcess:MatrixElements 0 MEee2gZ2qq
cd /Herwig/Generators
set EventGenerator:EventHandler:LuminosityFunction:Energy 91.2
set /Herwig/Shower/AlphaQCD:AlphaIn {alphas:.6f}
set /Herwig/Hadronization/HadronSelector:PwtSquark {pwt:.6f}
set /Herwig/Hadronization/ClusterFissioner:ClMaxLight {clmax:.6f}
set /Herwig/Hadronization/ClusterFissioner:ClPowLight {clpow:.6f}
set /Herwig/Hadronization/ClusterFissioner:PSplitLight {psplit:.6f}
set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm
set /Herwig/Decays/DecayHandler:LifeTimeOption 0
read snippets/HepMC.in
set /Herwig/Analysis/HepMC:PrintEvent 100000000
set /Herwig/Analysis/HepMC:Filename {out}
saverun {tag} EventGenerator
"""

def env():
    e = dict(os.environ)
    e['DYLD_LIBRARY_PATH'] = f"{HW/'lib'/'Herwig'}:{HW/'lib'}:" + e.get('DYLD_LIBRARY_PATH', '')
    return e

def generate(tag, n, seed, **kw):
    """Run Herwig at one parameter point, return the path of the HepMC file."""
    p = dict(DEF); p.update(kw); out = WORK/f'{tag}.hepmc'
    (WORK/f'{tag}.in').write_text(CARD.format(out=out.name, tag=tag, **p))
    for cmd in (['read', '-i', str(SHARE), f'{tag}.in'], ['run', f'{tag}.run', '-N', str(n), '--seed', str(seed)]):
        r = subprocess.run([str(HW/'bin'/'Herwig')] + cmd, cwd=WORK, env=env(), capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f'Herwig {cmd[0]} failed for {tag}')
    return out

def stream(path):
    cur = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('E '):
                if cur: yield np.asarray(cur, np.float64); cur = []
                continue
            if not line.startswith('P '): continue
            q = line.split()
            try:
                pid = int(q[2]); st = int(q[8])
                if st == 1 and abs(pid) not in NEUTRINO:
                    cur.append([float(q[6]), float(q[3]), float(q[4]), float(q[5]), pid, float(q[7])])
            except (ValueError, IndexError): continue
    if cur: yield np.asarray(cur, np.float64)

def extract(hepmc, n, maxp=100):
    evs = []
    for ev in stream(hepmc):
        evs.append(ev)
        if len(evs) == n: break
    N = len(evs)
    parts = np.zeros((N, maxp, 6), np.float32); masks = np.zeros((N, maxp), bool)
    counts = np.zeros(N, np.int32); nbar = np.zeros(N, np.int32); rows = []
    for i, ev in enumerate(evs):
        p, m, nb = full_event_particles(ev, maxp)
        parts[i] = p; masks[i] = m; counts[i] = m.sum(); nbar[i] = nb
        rows.append(compute_hemisphere_shapes(ev[:, :4]))
    return dict(particles=parts, mask=masks, counts=counts, nbaryon=nbar), pd.DataFrame(rows)

def strange_frac(parts, mask):
    """The project's strange observable, identical to r2_ladder.strange_frac: the kaon mass
    window on the stored log10(mass) column, since the cached format keeps no PID."""
    lm = parts[..., 3]; m = mask.astype(bool); ink = ((lm > -0.36) & (lm < -0.20)) & m
    return ink.sum(1)/np.maximum(m.sum(1), 1)

def summarize(tag, arrs, sh):
    return dict(tag=tag, n=len(sh), mult=float(arrs['counts'].mean()), nbaryon=float(arrs['nbaryon'].mean()),
                strange=float(strange_frac(arrs['particles'], arrs['mask']).mean()),
                thrust=float(sh['1_minus_thrust'].mean()), B=float(sh['B_total'].mean()), rho=float(sh['rho_heavy'].mean()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('n', type=int)
    ap.add_argument('--rid0', type=int, default=9000); ap.add_argument('--out', default='data_herwigbox')
    ap.add_argument('--seed', type=int, default=90000)
    a = ap.parse_args()
    if a.mode == 'scan':
        pts = [('default', {})]
        for v in (0.106, 0.112, 0.125, 0.132): pts.append((f'alphas_{v}', dict(alphas=v)))
        for v in (0.15, 0.20, 0.40, 0.55): pts.append((f'pwt_{v}', dict(pwt=v)))
        for v in (2.4, 3.0, 4.3, 5.2): pts.append((f'clmax_{v}', dict(clmax=v)))
        for v in (0.70, 0.80, 1.00, 1.10): pts.append((f'psplit_{v}', dict(psplit=v)))
        for v in (2.2, 2.5, 3.1, 3.4): pts.append((f'clpow_{v}', dict(clpow=v)))
        rows = []
        for i, (tag, kw) in enumerate(pts):
            h = generate(tag, a.n, a.seed + i, **kw); arrs, sh = extract(h, a.n); h.unlink()
            r = summarize(tag, arrs, sh); r.update(kw); rows.append(r)
            print(f"{tag:14s} mult {r['mult']:6.2f}  strange {r['strange']:.4f}  <1-T> {r['thrust']:.5f}  "
                  f"<B> {r['B']:.5f}  nbaryon {r['nbaryon']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(os.environ.get('SCAN_OUT','output/herwig_scan.csv'), index=False); print('wrote', os.environ.get('SCAN_OUT','output/herwig_scan.csv'))
    else:
        # Herwig's OWN box, three knobs chosen by their own response in the scan:
        #   AlphaIn     shapes        (ClPowLight moves nothing measurable; PSplitLight duplicates ClMaxLight)
        #   PwtSquark   strangeness
        #   ClMaxLight  multiplicity and the baryon rate
        # Ranges are Herwig's own, centred on its defaults, NOT matched to Sherpa: the two
        # generators are independent models and each carries its own parameters.
        AL = [0.1106, 0.1186, 0.1266]; PW = [0.15, 0.2917, 0.50]; CM = [2.80, 3.649, 4.80]
        GRID = [(a_, p_, c_) for a_ in AL for p_ in PW for c_ in CM]          # 27 training points
        HELD = [(0.1146, 0.22, 3.20), (0.1226, 0.40, 4.30), (0.1186, 0.2917, 3.649),
                (0.1126, 0.35, 3.00), (0.1246, 0.18, 4.60)]                   # 5 interior held points
        pts = [(f'hw_{a.rid0+i:04d}', dict(alphas=t[0], pwt=t[1], clmax=t[2]), t) for i, t in enumerate(GRID)]
        pts += [(f'hw_{a.rid0+27+j:04d}', dict(alphas=t[0], pwt=t[1], clmax=t[2]), t) for j, t in enumerate(HELD)]
        out = Path(a.out); out.mkdir(exist_ok=True)
        meta = []
        todo = [(tag, kw, th, i) for i, (tag, kw, th) in enumerate(pts)
                if not (out/f'particles_full_{a.rid0+i:04d}.npz').exists()]
        HW_K, HW_NK = int(os.environ.get('HW_K', 0)), int(os.environ.get('HW_NK', 1)); todo = todo[HW_K::HW_NK]
        print(f'{len(pts)} Herwig points, {len(todo)} to do in this slice {HW_K}/{HW_NK}, {a.n} events each -> {a.out}', flush=True)
        for tag, kw, th, i in todo:
            rid = a.rid0 + i
            h = generate(tag, a.n, a.seed + 7*i, **kw)
            arrs, sh = extract(h, a.n); h.unlink(); (WORK/f'{tag}.run').unlink(missing_ok=True)
            np.savez_compressed(out/f'particles_full_{rid:04d}.npz', **arrs)
            sh.to_csv(out/f'shapes_run_{rid:04d}.csv', index=False)
            r = summarize(tag, arrs, sh); r.update(kw); r['rid'] = rid; r['theta'] = list(th); meta.append(r)
            print(f"  rid {rid} a={kw['alphas']:.4f} pwt={kw['pwt']:.4f} clmax={kw['clmax']:.3f}: "
                  f"mult {r['mult']:6.2f} strange {r['strange']:.4f} <1-T> {r['thrust']:.5f} nbaryon {r['nbaryon']:.3f}", flush=True)
        import json as _j
        mf = out/(f'meta_{HW_K}.json' if HW_NK > 1 else 'meta.json')
        old = _j.load(open(mf)) if mf.exists() else []
        _j.dump(old + meta, open(mf, 'w'), indent=1)
        print('HERWIG BOX DONE')

if __name__ == '__main__':
    main()
