"""Our PyTorch implementation on the identical task, architecture and data. The conditional
head is collapsed to the unconditional case (K=1 with a constant theta), which is exactly a
standard PFN, so the two implementations are comparable."""
import json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from crosscheck_data import build
from r2_ladder import CondPFN

import os
DEV = os.environ.get('CROSSCHECK_DEVICE', 'cpu')   # cpu by default: the ladder owns the GPU
Xtr, ytr, Xte, yte = build(seed=0)
print('train', Xtr.shape, 'test', Xte.shape, 'device', DEV, flush=True)
xt = torch.tensor(Xtr); mt = torch.tensor((np.abs(Xtr).sum(-1) > 0).astype(np.float32))
yt = torch.tensor(ytr)
xe = torch.tensor(Xte); me = torch.tensor((np.abs(Xte).sum(-1) > 0).astype(np.float32))

torch.manual_seed(0)
m = CondPFN(Xtr.shape[-1], 1, K=1).to(DEV)          # K=1 + constant theta == plain PFN
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
        out = m.f_from(m.emb(fb, mb), torch.zeros(len(j), 1, device=DEV))
        loss = bce(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss)*len(j)
    print(f'  epoch {ep+1}: loss={tot/n:.4f}', flush=True)
m.eval(); preds = []
with torch.no_grad():
    for i in range(0, len(xe), 2000):
        fb = xe[i:i+2000].to(DEV); mb = me[i:i+2000].to(DEV)
        preds.append(m.f_from(m.emb(fb, mb), torch.zeros(len(fb), 1, device=DEV)).cpu().numpy())
auc = float(roc_auc_score(yte, np.concatenate(preds)))
print(f'OUR TORCH PFN test AUC = {auc:.4f}')
json.dump(dict(impl='ours_pytorch_CondPFN_K1', auc=auc, n_train=int(len(Xtr)),
               n_test=int(len(Xte)), Phi_sizes=[100,100,256], F_sizes=[100,100,100],
               epochs=20, batch_size=500),
          open('output/crosscheck_torch.json','w'), indent=1)
