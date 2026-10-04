#!/usr/bin/env python3
"""Does the PFN do real work? A toy where the sufficient statistic is NOT the plain
sum, so the per-particle map must learn a nontrivial feature.

Particles x_i ~ N(0, e^{2s}); parameter theta = s (a width), reference s=0. Then
  f = -N s + (1/2)(1 - e^{-2s}) * sum_i x_i^2,
so the event enters only through Q = sum_i x_i^2, and a sum-pool can produce Q only
if the per-particle map learns phi(x) ~ x^2. We check three things against the exact
answer:
  (1) f_psi recovers the exact weight (slope ~1, correlation ~1);
  (2) the PFN reduces the event to Q = sum x_i^2 (high R^2 on Q) and NOT to the plain
      sum S = sum x_i (low R^2 on S) -- the nontrivial feature was learned;
  (3) the single-particle response f_psi(x_0) is a parabola (phi learned x^2).
"""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':13,'axes.labelsize':14,
    'legend.fontsize':10.5,'legend.frameon':False,'figure.dpi':120,'savefig.dpi':220,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25})
NP=10
def sample(s,n,rng): return rng.normal(0.0,np.exp(s),(n,NP)).astype(np.float32)
def analytic(X,s): return -NP*s + 0.5*(1-np.exp(-2*s))*(X**2).sum(1)
class Cond(nn.Module):
    def __init__(s,L=24,h=64,K=12):
        super().__init__()
        s.phi=nn.Sequential(nn.Linear(1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,L))
        s.A=nn.Sequential(nn.Linear(L,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,K))
        s.B=nn.Sequential(nn.Linear(1,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,K))
    def emb(s,x): b=x.shape[0]; return s.phi(x.reshape(b*NP,1)).reshape(b,NP,-1).sum(1)
    def at(s,E,th): return (s.A(E)*s.B(th.reshape(-1,1))).sum(-1)
    def forward(s,x,th): return s.at(s.emb(x),th)
def train(epochs=2500,seed=0):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    grid=list(np.linspace(-0.25,0.25,12)); xref=sample(0.0,200000,rng)
    xr=torch.tensor(xref); tg=[torch.tensor(sample(s,2000,rng)) for s in grid]
    m=Cond(); opt=torch.optim.Adam(m.parameters(),1e-3); g=torch.Generator().manual_seed(seed)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=2e-5); bce=nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        j=int(torch.randint(12,(1,),generator=g)); ir=torch.randint(200000,(2048,),generator=g); it=torch.randint(2000,(2048,),generator=g)
        loss=bce(m(torch.cat([xr[ir],tg[j][it]]),torch.full((4096,),float(grid[j]))),torch.cat([torch.zeros(2048),torch.ones(2048)]))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); return m
def R2(y,x):
    c,*_=np.linalg.lstsq(np.vstack([x,np.ones_like(x)]).T,y,rcond=None); return 1-((y-(c[0]*x+c[1]))**2).mean()/y.var(), c[0]
def main():
    rng=np.random.default_rng(7); xref=sample(0.0,50000,rng); xr=torch.tensor(xref)
    Q=(xref**2).sum(1); S=xref.sum(1)
    m=train(2500)
    print('=== (1) recovers exact weight (width toy) ===')
    for s in [0.12,-0.12,0.2]:
        with torch.no_grad(): fp=m.at(m.emb(xr),torch.full((len(xref),),float(s))).numpy()
        fa=analytic(xref,s); A=np.vstack([fa,np.ones_like(fa)]).T; cc,*_=np.linalg.lstsq(A,fp,rcond=None)
        print(f'  s={s:+.2f}: slope alpha={cc[0]:.3f}  corr rho={np.corrcoef(fp,fa)[0,1]:.4f}')
    import json
    dump={}
    for s in [0.12,0.2]:
        with torch.no_grad(): fp=m.at(m.emb(xr),torch.full((len(xref),),float(s))).numpy()
        fa=analytic(xref,s); A=np.vstack([fa,np.ones_like(fa)]).T; cc,*_=np.linalg.lstsq(A,fp,rcond=None)
        idx=np.random.default_rng(1).choice(len(fa),3000,replace=False)
        dump[str(s)]=dict(f_exact=fa[idx].tolist(),f_learned=(fp[idx]-cc[1]).tolist(),
                          slope=float(cc[0]),corr=float(np.corrcoef(fp,fa)[0,1]))
    json.dump(dump,open('output/toy_fvf_data.json','w'))
    print('wrote output/toy_fvf_data.json')
    print('=== (2) which statistic did the PFN learn? (at s=0.2) ===')
    with torch.no_grad(): fp=m.at(m.emb(xr),torch.full((len(xref),),0.2)).numpy()
    r2Q,aQ=R2(fp,Q); r2S,aS=R2(fp,S)
    print(f'  R^2 of f_psi on Q=sum x_i^2  : {r2Q:.4f}  (slope {aQ:.4f})   <- nontrivial sufficient statistic')
    print(f'  R^2 of f_psi on S=sum x_i    : {r2S:.4f}                      <- plain sum, should be low')
    print('=== (3) single-particle response: vary one particle, hold the rest ===')
    base=sample(0.0,2000,rng).copy(); xs=np.linspace(-3,3,25); resp=[]
    for xv in xs:
        b=base.copy(); b[:,0]=xv
        with torch.no_grad(): resp.append(m.at(m.emb(torch.tensor(b)),torch.full((len(b),),0.2)).numpy().mean())
    resp=np.array(resp); cf=np.polyfit(xs,resp,2)
    print(f'  f_psi(x_0) fit a x^2 + b x + c:  a={cf[0]:.3f} (curvature), b={cf[1]:.3f} (~0 by symmetry)')
    # figure
    fig,ax=plt.subplots(1,3,figsize=(15.5,4.5))
    with torch.no_grad(): fp=m.at(m.emb(xr),torch.full((len(xref),),0.2)).numpy()
    fa=analytic(xref,0.2); idx=rng.choice(len(fa),4000,replace=False)
    A=np.vstack([fa,np.ones_like(fa)]).T; cc,*_=np.linalg.lstsq(A,fp,rcond=None)
    ax[0].scatter(fa[idx],fp[idx],s=4,alpha=0.3,color='#1b6ca8'); lo,hi=fa[idx].min(),fa[idx].max()
    ax[0].plot([lo,hi],[lo,hi],'k--',lw=1.5,label='exact ($y=x$)')
    ax[0].set_xlabel(r'exact $\log w=-Ns+\frac{1}{2}(1-e^{-2s})\sum_i x_i^2$'); ax[0].set_ylabel(r'learned $\langle a(\Phi),b(s)\rangle$')
    ax[0].set_title(rf'Recovers exact weight: slope ${cc[0]:.2f}$, corr ${np.corrcoef(fp,fa)[0,1]:.3f}$'); ax[0].legend()
    ax[1].scatter(Q[idx],fp[idx],s=4,alpha=0.3,color='#2e8b57',label=rf'vs $Q=\sum x_i^2$: $R^2={r2Q:.3f}$')
    ax[1].scatter(S[idx],fp[idx],s=4,alpha=0.3,color='#d1495b',label=rf'vs $S=\sum x_i$: $R^2={r2S:.3f}$')
    ax[1].set_xlabel(r'candidate sufficient statistic'); ax[1].set_ylabel(r'learned $\langle a(\Phi),b(s)\rangle$')
    ax[1].set_title('PFN reduces the event to $\\sum x_i^2$, not $\\sum x_i$'); ax[1].legend()
    ax[2].plot(xs,resp,'o',color='#5e548e',ms=6,label='network response')
    ax[2].plot(xs,np.polyval(cf,xs),'k-',lw=1.5,label=rf'parabola, curvature ${cf[0]:.2f}$')
    ax[2].set_xlabel(r'one particle $x_0$ (others fixed)'); ax[2].set_ylabel(r'$\langle a(\Phi),b(s)\rangle$')
    ax[2].set_title(r'Per-particle map learned $x\mapsto x^2$'); ax[2].legend()
    fig.tight_layout(); fig.savefig('output/fig_toy_pfn.pdf',bbox_inches='tight'); print('wrote output/fig_toy_pfn.pdf')
if __name__=='__main__': main()
