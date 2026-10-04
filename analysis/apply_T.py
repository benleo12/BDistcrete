#!/usr/bin/env python3
"""Write a recalibrated temperature into an export, with the original kept and the reason recorded.

    python apply_T.py C_1M            # takes best_fine.T from output/recal_T_C_1M.json
    python apply_T.py C_1M 1.125      # or an explicit value

The export is rewritten in place after a byte copy is saved as <export>.T<old>. Three keys are
added: temperature_original, temperature_recal_objective and temperature_recal_source, so the
head itself says where its calibration came from. Nothing else in the file changes, and the
logit checksum is unaffected because it is taken before the temperature is applied."""
import sys, os, json, shutil
import numpy as np

tag = sys.argv[1]
path = f'output/models/{tag}_cond.npz'
rec = json.load(open(f'output/recal_T_{tag}.json'))
T_new = float(sys.argv[2]) if len(sys.argv) > 2 else float(rec['best_fine']['T'])
z = np.load(path, allow_pickle=True)
T_old = float(z['temperature'])
if abs(T_new - T_old) < 1e-9:
    print(f'{tag}: temperature already {T_old}, nothing to do'); sys.exit(0)
bak = f'{path}.T{T_old:.3f}'
if not os.path.exists(bak):
    shutil.copy2(path, bak)
    print(f'original saved as {bak}')
d = {k: z[k] for k in z.files}
d['temperature'] = np.array(T_new)
d['temperature_original'] = np.array(T_old)
d['temperature_recal_objective'] = np.array(rec['objective'])
d['temperature_recal_source'] = np.array(f'recal_T_{tag}.json, fine grid, stop width {rec["best_fine"]["width_stop"]:.3f}')
# np.savez appends .npz to a name that lacks it, so the temporary name must end in .npz
tmp = path[:-4] + '.tmp.npz'
np.savez(tmp, **d)
os.replace(tmp, path)
chk = np.load(path)
print(f'{tag}: temperature {T_old} -> {float(chk["temperature"])}  (original {float(chk["temperature_original"])})')
