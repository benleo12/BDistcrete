#!/usr/bin/env python3
"""Flavor-aware extraction for Stage B+ (hadronization parameters).
Same heavy-hemisphere, thrust-frame representation as extract_particles.py, but each
particle now carries flavor features so the network can reweight flavor-changing
hadronization knobs (e.g. BARYON_FRACTION) that leave the kinematics unchanged:
    feats : [z, cos_theta, phi, log10(mass), baryon_number, charge]  (C=6)
Also stores nbaryon = number of baryons in the heavy hemisphere (a baryon-sensitive
closure observable). Mass and PID come straight from the HepMC P-line."""
import argparse, gzip, numpy as np
from pathlib import Path
NEUTRINO={12,14,16}
CHG1={211,321,2212,3222,3112,3312,3334,11,13,15,411,431,521}  # |charge|=1 species
def charge(pid):
    a=abs(pid)
    return (1 if pid>0 else -1) if a in CHG1 else 0      # neutrals (gamma,pi0,K0,n,Lambda,...) -> 0
def baryonnum(pid):
    a=abs(pid)
    return (1 if pid>0 else -1) if 1000<=a<1000000 else 0
def thrust_axis(p3):
    if len(p3)==0: return np.array([0,0,1.0])
    mags=np.linalg.norm(p3,axis=1); n=p3[np.argmax(mags)]/(mags[np.argmax(mags)]+1e-12)
    for _ in range(20):
        nn=(np.sign(p3@n)[:,None]*p3).sum(0); m=np.linalg.norm(nn)
        if m<1e-12: break
        nn=nn/m
        if np.dot(nn,n)>1-1e-12: break
        n=nn
    return n
def parse(filepath,max_events=None):
    opener=gzip.open if str(filepath).endswith('.gz') else open
    events=[]; cur=[]
    with opener(filepath,'rt') as fh:
        for line in fh:
            if line.startswith('E '):
                if cur: events.append(np.asarray(cur,dtype=np.float64)); cur=[]
                if max_events is not None and len(events)>=max_events: break
                continue
            if not line.startswith('P '): continue
            p=line.split()
            try:
                if len(p)>=10: pid=int(p[3]);status=int(p[9]);px,py,pz,E,m=float(p[4]),float(p[5]),float(p[6]),float(p[7]),float(p[8])
                elif len(p)>=9: pid=int(p[2]);status=int(p[8]);px,py,pz,E,m=float(p[3]),float(p[4]),float(p[5]),float(p[6]),float(p[7])
                else: continue
                if status==1 and abs(pid) not in NEUTRINO: cur.append([E,px,py,pz,pid,m])
            except (ValueError,IndexError): continue
    if cur: events.append(np.asarray(cur,dtype=np.float64))
    return events
def hemi(ev,P=60):
    out=np.zeros((P,6),np.float32); mask=np.zeros(P,bool)
    if ev.shape[0]==0: return out,mask,0
    p3=ev[:,1:4]; n=thrust_axis(p3); signs=np.sign(p3@n)
    def m2(sel):
        q=ev[sel]
        if len(q)==0: return -1.0
        S=q[:,:4].sum(0); return S[0]**2-(S[1]**2+S[2]**2+S[3]**2)
    hs=1 if m2(signs>0)>=m2(signs<0) else -1; keep=ev[signs==hs]
    if keep.shape[0]==0: return out,mask,0
    E=keep[:,0]; z=E/max(E.sum(),1e-12)
    ph=keep[:,1:4]/(np.linalg.norm(keep[:,1:4],axis=1,keepdims=True)+1e-12)
    cth=ph@n
    zx=np.array([0.,0.,1.]); e1=np.cross(n,zx)
    if np.linalg.norm(e1)<1e-6: e1=np.array([1.,0.,0.])
    e1/=np.linalg.norm(e1); e2=np.cross(n,e1); phi=np.arctan2(ph@e2,ph@e1)
    pid=keep[:,4].astype(int); mass=keep[:,5]
    bn=np.array([baryonnum(x) for x in pid],np.float32); ch=np.array([charge(x) for x in pid],np.float32)
    feats=np.stack([z,cth,phi,np.log10(np.maximum(mass,1e-3)),bn,ch],1).astype(np.float32)
    Pk=min(feats.shape[0],P); order=np.argsort(-z)[:Pk]
    out[:Pk]=feats[order]; mask[:Pk]=True
    return out,mask,int((np.abs(bn)>0).sum())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--max-events',type=int,default=None); ap.add_argument('--max-particles',type=int,default=60)
    a=ap.parse_args()
    print(f'parsing {a.input}',flush=True); evs=parse(a.input,a.max_events); N=len(evs)
    parts=np.zeros((N,a.max_particles,6),np.float32); masks=np.zeros((N,a.max_particles),bool)
    counts=np.zeros(N,np.int32); nbar=np.zeros(N,np.int32)
    for i,ev in enumerate(evs):
        p,m,nb=hemi(ev,a.max_particles); parts[i]=p; masks[i]=m; counts[i]=m.sum(); nbar[i]=nb
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,particles=parts,mask=masks,counts=counts,nbaryon=nbar)
    print(f'  saved {a.output}: N={N} mean mult={counts.mean():.1f} mean hemi-baryons={nbar.mean():.3f}')
if __name__=='__main__': main()
