#!/usr/bin/env python3
"""Make the inner product believable: plot the learned b(s) against the exact natural
parameters, and show that the weight matrix f(Phi,s) is intrinsically rank 2, which is
why a low-dimensional inner product <a,b> suffices.

Exact answer (width toy):  f = -N s + (1/2)(1-e^{-2s}) Q,  Q = sum_i x_i^2.
As a function of (event, s) this is a sum of TWO separable terms,
   f(Phi,s) = 1 * (-N s)  +  Q(Phi) * (1/2)(1-e^{-2s}),
so the matrix M[i,j]=f(Phi_i,s_j) has rank 2. We recover b(s) by regressing the learned
f on (Q,1) at each s, and we read the effective dimension from the singular values of M.
Allocated latent size is K=12 (a,b in R^12); the problem needs only 2.
Writes output/fig_toy_ab.pdf and prints the dimensions and singular values."""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':13,'axes.labelsize':14,
    'legend.fontsize':10.5,'legend.frameon':False,'figure.dpi':120,'savefig.dpi':220,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25})
BLUE='#1b6ca8'; RED='#d1495b'; GREEN='#2e8b57'; NP=10; K=12
DEV='mps' if torch.backends.mps.is_available() else 'cpu'
def full(n,v): return torch.full((n,),float(v),device=DEV)
def sample(s,n,rng): return rng.normal(0.0,np.exp(s),(n,NP)).astype(np.float32)
class Cond(nn.Module):
    def __init__(s,L=24,h=64,Kd=K):
        super().__init__()
        s.phi=nn.Sequential(nn.Linear(1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,L))
        s.A=nn.Sequential(nn.Linear(L,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,Kd))
        s.B=nn.Sequential(nn.Linear(1,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,Kd))
    def emb(s,x): b=x.shape[0]; return s.phi(x.reshape(b*NP,1)).reshape(b,NP,-1).sum(1)
    def avec(s,x): return s.A(s.emb(x))          # a(Phi) in R^K
    def bvec(s,th): return s.B(th.reshape(-1,1))  # b(theta) in R^K
    def at(s,E,th): return (s.A(E)*s.B(th.reshape(-1,1))).sum(-1)
    def forward(s,x,th): return s.at(s.emb(x),th)
def train(epochs=8000,seed=0,NG=5000):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    grid=list(np.linspace(-0.25,0.25,12)); xref=sample(0.0,200000,rng); xr=torch.tensor(xref).to(DEV)
    tg=[torch.tensor(sample(s,NG,rng)).to(DEV) for s in grid]
    m=Cond().to(DEV); opt=torch.optim.Adam(m.parameters(),1e-3); g=torch.Generator().manual_seed(seed)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=2e-5); bce=nn.BCEWithLogitsLoss()
    lab=torch.cat([torch.zeros(2048),torch.ones(2048)]).to(DEV)
    for ep in range(epochs):
        j=int(torch.randint(12,(1,),generator=g)); ir=torch.randint(200000,(2048,),generator=g); it=torch.randint(NG,(2048,),generator=g)
        loss=bce(m(torch.cat([xr[ir],tg[j][it]]),full(4096,grid[j])),lab)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); return m
def main():
    print(f'device={DEV}  a,b dimension K={K}')
    m=train(8000); rng=np.random.default_rng(3)
    X=sample(0.0,4000,rng); Xt=torch.tensor(X).to(DEV); Q=(X**2).sum(1)
    ss=np.linspace(-0.25,0.25,60)
    # (1) recover b(s): regress learned f on (Q,1) at each s -> coefficients c1(s), c0(s)
    c1=[]; c0=[]
    for s in ss:
        with torch.no_grad(): f=m.at(m.emb(Xt),full(len(X),s)).cpu().numpy()
        A=np.vstack([Q,np.ones_like(Q)]).T; coef,*_=np.linalg.lstsq(A,f,rcond=None); c1.append(coef[0]); c0.append(coef[1])
    c1=np.array(c1); c0=np.array(c0)
    an_c1=0.5*(1-np.exp(-2*ss)); an_c0=-NP*ss
    # (2) SVD of the weight matrix M[i,j]=f(Phi_i, s_j)
    sj=np.linspace(-0.25,0.25,50); M=np.zeros((len(X),len(sj)))
    for j,s in enumerate(sj):
        with torch.no_grad(): M[:,j]=m.at(m.emb(Xt),full(len(X),s)).cpu().numpy()
    sv=np.linalg.svd(M,compute_uv=False); sv=sv/sv[0]
    print('singular values (normalised):', np.array2string(sv[:6],precision=4))
    print(f'effective rank (sv>1e-2): {(sv>1e-2).sum()}  -> exact answer is rank 2')
    import json
    json.dump(dict(ss=ss.tolist(),c0=c0.tolist(),c1=c1.tolist(),
                   an_c0=an_c0.tolist(),an_c1=an_c1.tolist(),sv=sv[:10].tolist()),
              open('output/toy_ab_data.json','w'),indent=1)
    fig,ax=plt.subplots(1,2,figsize=(12,4.7))
    ax[0].plot(ss,an_c1,'-',color=GREEN,lw=2.5,label=r'exact $\frac{1}{2}(1-e^{-2s})$ (coeff. of $Q$)')
    ax[0].plot(ss,c1,'o',color=GREEN,ms=4,mfc='none')
    ax[0].plot(ss,an_c0,'-',color=RED,lw=2.5,label=r'exact $-Ns$ (offset)')
    ax[0].plot(ss,c0,'s',color=RED,ms=4,mfc='none')
    ax[0].plot([],[],'ko',mfc='none',label='learned $b(s)$ (recovered)')
    ax[0].set_xlabel('parameter $s$'); ax[0].set_ylabel(r'coefficient in $f=\langle a,b(s)\rangle$')
    ax[0].set_title(r'The parameter network $b(s)$ recovers the known answer'); ax[0].legend(loc='upper left',fontsize=9.5)
    ax[1].semilogy(np.arange(1,len(sv)+1),sv,'o-',color=BLUE,ms=6)
    ax[1].axhline(1e-2,ls=':',color='#888'); ax[1].text(20,1.3e-2,'noise floor',fontsize=9,color='#666')
    ax[1].axvspan(0.5,2.5,color=GREEN,alpha=0.12); ax[1].text(2.7,0.3,'2 modes carry\nthe weight',fontsize=10,color=GREEN)
    ax[1].set_xlabel('singular-value index'); ax[1].set_ylabel('normalised singular value of $f(\\Phi,s)$')
    ax[1].set_title(r'The weight table needs only two directions'); ax[1].set_ylim(1e-4,2)
    fig.tight_layout(); fig.savefig('output/fig_toy_ab.pdf',bbox_inches='tight'); print('wrote output/fig_toy_ab.pdf')
if __name__=='__main__': main()
