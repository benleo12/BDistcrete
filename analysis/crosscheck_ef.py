"""Reference implementation: the energyflow PFN (Komiske, Metodiev, Thaler), run in an
isolated venv with a TensorFlow backend. Published architecture, no tuning."""
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
from crosscheck_data import build
from energyflow.archs import PFN

# CROSSCHECK_SEED fixes the initialization and shuffling (unset: unseeded, as published), so the
# run-to-run spread can be measured. CROSSCHECK_OUT names the result file.
SEED = int(os.environ.get('CROSSCHECK_SEED', '-1'))
OUT = os.environ.get('CROSSCHECK_OUT', 'output/crosscheck_ef.json')
if SEED >= 0:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
Xtr, ytr, Xte, yte = build(seed=0)
print('train', Xtr.shape, 'test', Xte.shape, flush=True)
pfn = PFN(input_dim=Xtr.shape[-1], Phi_sizes=(100, 100, 256), F_sizes=(100, 100, 100),
          output_dim=1, output_act='sigmoid', loss='binary_crossentropy', summary=False)
pfn.fit(Xtr, ytr, epochs=20, batch_size=500, verbose=2)
pred = pfn.predict(Xte, batch_size=1000).ravel()
auc = float(roc_auc_score(yte, pred))
print(f'ENERGYFLOW PFN test AUC = {auc:.4f}')
res = dict(impl='energyflow_PFN_tensorflow', auc=auc, n_train=int(len(Xtr)),
           n_test=int(len(Xte)), Phi_sizes=[100,100,256], F_sizes=[100,100,100],
           epochs=20, batch_size=500)
if SEED >= 0:
    res['seed'] = SEED
json.dump(res, open(OUT, 'w'), indent=1)
