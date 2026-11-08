#!/usr/bin/env python3
"""Write a recalibrated temperature into an export, with the original kept and the reason recorded.

    python apply_T.py C_1M            # takes best_fine.T from output/recal_T_C_1M.json
    python apply_T.py C_1M 1.125      # or an explicit value

The export is rewritten in place after a byte copy is saved as <export>.T<old>. Three keys are
added: temperature_original, temperature_recal_objective and temperature_recal_source, so the
head itself says where its calibration came from. Nothing else in the file changes, and the
logit checksum is unaffected because it is taken before the temperature is applied.

    T_DIR=output/temperature python apply_T.py --lmx MIXGEOV3S9 [MIXGEOV3S9s1 ...]

The rerun (D11): the refitted temperature of each model comes from T_DIR/<model>.json. When it
differs from the export's, output/models/<model>_cond.npz in the CURRENT folder becomes a new file
with it; a link there (to the export of the published run) is replaced by that file, never written
through, and no backup is made, since the published export stays where it is. The other links of the
same network in output/models (MIXP2v3fullfac0 -> MIXGEOV3S9, <tag>h400) are pointed at the new
file. The folder written must resolve under lmx_runs (the rerun) or lmx_phaseB (its tests), and
never into the published folder maxent_code (a prefix test on its real path, so RUN, itself named
lmx_runs/maxent_code, passes). When the temperature is unchanged nothing is written. temperature_recal_from
records the export the first refit started from: the target of a link, or, for a file in the folder (B0 copied
the exports into RUN), the published export of that name when the file is a byte copy of it; otherwise that the
file is a copy of its own. A second refit keeps temperature_original and temperature_recal_from."""
import sys, os, json, shutil, filecmp
import numpy as np

# --lmx writes only under these folders, never into the published run
PUBLISHED = '/mnt/beegfs/bassi/reweighting/maxent_code/'
LMX_ROOTS = ('/mnt/beegfs/bassi/reweighting/lmx_runs/', '/mnt/beegfs/bassi/reweighting/lmx_phaseB/')


def started_from(path, tag):
    """the export a first refit starts from: the target of a link; for a file, the published export of the same
    name when the file is a byte copy of it, else the file itself, said to be a copy of its own"""
    if os.path.islink(path):
        return os.path.realpath(path)
    pub = f'{PUBLISHED}output/models/{tag}_cond.npz'
    if os.path.isfile(pub) and filecmp.cmp(path, pub, shallow=False):
        return pub
    return f'{os.path.realpath(path)} (a file of the rerun folder, not a byte copy of {pub})'

if sys.argv[1] == '--lmx':
    TD = os.environ['T_DIR']
    for tag in sys.argv[2:]:
        path = f'output/models/{tag}_cond.npz'
        if not os.path.exists(path):
            raise FileNotFoundError(f'missing input {path}')
        rec = json.load(open(os.path.join(TD, f'{tag}.json')))
        T_new = float(rec['T'])
        z = np.load(path, allow_pickle=True)
        T_old = float(z['temperature'])
        if abs(T_new - T_old) < 1e-12:
            print(f'{tag}: temperature {T_old} unchanged by the refit, nothing written'); continue
        here = os.path.realpath(os.path.dirname(path)) + '/'
        if here.startswith(PUBLISHED) or not here.startswith(LMX_ROOTS):
            raise SystemExit(f'{tag}: {path} would be written in {here}, which is in the published {PUBLISHED} '
                             f'or not under {" or ".join(LMX_ROOTS)}: nothing written')
        d = {k: z[k] for k in z.files}
        d['temperature'] = np.array(T_new)
        d['temperature_original'] = np.array(float(z['temperature_original']) if 'temperature_original' in z.files else T_old)
        d['temperature_recal_objective'] = np.array(rec['objective'])
        d['temperature_recal_source'] = np.array(f'{TD}/{tag}.json ({rec.get("tool", "")}), data {rec.get("data", "")}, '
                                                 f'beam photons {rec.get("beam_photons", "")}, stop width {rec["stop_width"]:.4f}')
        # the export the first refit started from (the published one); a later refit keeps it
        d['temperature_recal_from'] = np.array(str(z['temperature_recal_from']) if 'temperature_recal_from' in z.files
                                               else started_from(path, tag))
        tmp = path[:-4] + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, **d)
        os.replace(tmp, path)            # replaces a link by the file, does not write through it
        chk = np.load(path)
        print(f'{tag}: temperature {T_old} -> {float(chk["temperature"])} in {path} (a new file; the export it started from: {d["temperature_recal_from"]})')
        # the other names of this network in output/models point at the new file
        for f in os.listdir('output/models'):
            p = os.path.join('output/models', f)
            if f.endswith('_cond.npz') and f != f'{tag}_cond.npz' and os.path.islink(p) \
                    and os.path.basename(os.path.realpath(p)) == f'{tag}_cond.npz':
                os.remove(p); os.symlink(f'{tag}_cond.npz', p)
                print(f'   {p} -> {tag}_cond.npz')
    sys.exit(0)

tag = sys.argv[1]
path = f'output/models/{tag}_cond.npz'
# an explicit value needs no recalibration file (the seed-study heads have none)
rec_path = f'output/recal_T_{tag}.json'
rec = json.load(open(rec_path)) if (len(sys.argv) <= 2 or os.path.exists(rec_path)) else None
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
d['temperature_recal_objective'] = np.array(rec['objective'] if rec else 'explicit value on the command line')
d['temperature_recal_source'] = np.array(f'recal_T_{tag}.json, fine grid, stop width {rec["best_fine"]["width_stop"]:.3f}'
                                         if rec else 'explicit value on the command line')
# np.savez appends .npz to a name that lacks it, so the temporary name must end in .npz
tmp = path[:-4] + '.tmp.npz'
np.savez(tmp, **d)
os.replace(tmp, path)
chk = np.load(path)
print(f'{tag}: temperature {T_old} -> {float(chk["temperature"])}  (original {float(chk["temperature_original"])})')
