"""The inputs of figs_r8.py widebox, from the stage-W results of the paper (DESIGN_TAG wb18). The paper
quotes them without a figure. They are the closure width per held-out coupling of the three references,
averaged over the observables, and the reference weights at alpha_s = 0.083 of the single run and of
the local union.
Writes output/widebox_wb18.json, output/wb18_weights_single_0.083.npy, output/wb18_weights_union_0.083.npy.
In the rerun (RUN holds the start mark output/.lmx_rerun_start, or LMX_CHECK=1) every result and weight dump
read must carry the fingerprint of the current mark and code (symmetry/rerun/lmx_fingerprint.py: a dump is
vouched for by the fingerprint of the scoring that wrote it); any other is refused."""
import json, os, shutil
import numpy as np
def load(p):
    d = json.load(open(p)); return d if "master" in d else list(d.values())[0]
def lmx_check(paths):
    """In the rerun, refuse a published copy or a result of another submission or code (F1)"""
    if os.environ.get('LMX_CHECK', '') != '1' and not os.path.exists('output/.lmx_rerun_start'):
        return
    import importlib.util
    s = importlib.util.spec_from_file_location('lmx_fingerprint', os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'symmetry', 'rerun', 'lmx_fingerprint.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    bad = m.check_files(paths)
    if bad:
        raise RuntimeError('inputs without the fingerprint of this rerun:\n' + '\n'.join(f'  {p}: {w}' for p, w in bad))
def per(p):
    return {"%g" % float(k.strip("()").split(",")[0]): float(np.mean(list(v.values()))) for k, v in load(p)["per_point"].items()}
RES = ["output/p2wb18single_W_fac_s0.json", "output/p2wb18pool_W_fac_s0.json"] + \
      ["output/p2wb18local%d_W_fac_s0.json" % r for r in range(9100, 9106)]
DUMPS = ["output/models/W_P2wb18singlefac0_w_9100.npy", "output/models/W_P2wb18local9100fac0_w_9100.npy"]
lmx_check(RES + DUMPS)
out = dict(single_centre=per(RES[0]), pool_all=per(RES[1]), local_union={})
for f in RES[2:]:
    out["local_union"].update(per(f))
out["summary"] = {k: float(np.mean(list(v.values()))) for k, v in out.items()}
json.dump(out, open("output/widebox_wb18.json", "w"), indent=1)
shutil.copy(DUMPS[0], "output/wb18_weights_single_0.083.npy")
shutil.copy(DUMPS[1], "output/wb18_weights_union_0.083.npy")
for name in ("single", "union"):
    w = np.load("output/wb18_weights_%s_0.083.npy" % name); u = w*len(w)
    print(name, "weights at 0.083: max %.1f (unit mean), N_eff %.0f, decades spanned %.1f" % (u.max(), 1/np.sum(w**2), np.log10(u.max()/max(u[u > 0].min(), 1e-30))))
print(json.dumps(out["summary"]))
