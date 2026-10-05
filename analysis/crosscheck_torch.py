"""Our PyTorch implementation on the identical task, architecture and data. The conditional
head is collapsed to the unconditional case (K=1 with a constant theta), which is exactly a
standard PFN, so the two implementations are comparable."""
import json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from crosscheck_data import build
from r2_ladder import CondPFN, _mlp, PHI_SIZES, F_SIZES

import os
DEV = os.environ.get('CROSSCHECK_DEVICE', 'cpu')   # cpu by default: the ladder owns the GPU
# CROSSCHECK_SEED varies the initialization and shuffling (default 0, as published). CROSSCHECK_PLAIN=1
# trains the energyflow architecture exactly, a scalar output with no parameter factor, to separate
# the implementation from the K=1 head's extra multiplicative constant. CROSSCHECK_OUT names the file.
SEED = int(os.environ.get('CROSSCHECK_SEED', '0'))
PLAIN = os.environ.get('CROSSCHECK_PLAIN', '0') == '1'
OUT = os.environ.get('CROSSCHECK_OUT', 'output/crosscheck_torch.json')


class PlainPFN(nn.Module):
    def __init__(s, C):
        super().__init__()
        s.phi, L = _mlp(C, PHI_SIZES)
        s.F, _ = _mlp(L, F_SIZES, 1)
        for mod in s.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu'); nn.init.zeros_(mod.bias)
    def forward(s, f, m):
        B, P, _ = f.shape
        E = (s.phi(f.reshape(B*P, -1)).reshape(B, P, -1) * m.unsqueeze(-1)).sum(1)
        return s.F(E).squeeze(-1)


Xtr, ytr, Xte, yte = build(seed=0)
print('train', Xtr.shape, 'test', Xte.shape, 'device', DEV, flush=True)
xt = torch.tensor(Xtr); mt = torch.tensor((np.abs(Xtr).sum(-1) > 0).astype(np.float32))
yt = torch.tensor(ytr)
xe = torch.tensor(Xte); me = torch.tensor((np.abs(Xte).sum(-1) > 0).astype(np.float32))

torch.manual_seed(SEED)
m = (PlainPFN(Xtr.shape[-1]) if PLAIN else CondPFN(Xtr.shape[-1], 1, K=1)).to(DEV)   # K=1 + constant theta == plain PFN
def logit(fb, mb):
    return m(fb, mb) if PLAIN else m.f_from(m.emb(fb, mb), torch.zeros(len(fb), 1, device=DEV))
opt = torch.optim.Adam(m.parameters(), 1e-3)
bce = nn.BCEWithLogitsLoss()
BS, EPOCHS = 500, 20
n = len(xt)
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, BS):
        j = perm[i:i+BS]
        fb = xt[j].to(DEV); mb = mt[j].to(DEV); yb = yt[j].to(DEV)
        out = logit(fb, mb)
        loss = bce(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss)*len(j)
    print(f'  epoch {ep+1}: loss={tot/n:.4f}', flush=True)
m.eval(); preds = []
with torch.no_grad():
    for i in range(0, len(xe), 2000):
        fb = xe[i:i+2000].to(DEV); mb = me[i:i+2000].to(DEV)
        preds.append(logit(fb, mb).cpu().numpy())
auc = float(roc_auc_score(yte, np.concatenate(preds)))
print(f'OUR TORCH PFN test AUC = {auc:.4f}')
res = dict(impl='ours_pytorch_plainPFN' if PLAIN else 'ours_pytorch_CondPFN_K1', auc=auc, n_train=int(len(Xtr)),
           n_test=int(len(Xte)), Phi_sizes=[100,100,256], F_sizes=[100,100,100],
           epochs=20, batch_size=500)
if SEED or PLAIN:
    res.update(seed=SEED, device=DEV)
json.dump(res, open(OUT, 'w'), indent=1)
