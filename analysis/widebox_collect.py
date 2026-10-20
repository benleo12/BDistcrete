"""The inputs of the wide-scan figure (figs_pro.py widebox) from the stage-W results of the paper
(DESIGN_TAG wb18): the closure width per held-out coupling of the three references, averaged over
the observables, and the reference weights at alpha_s = 0.083 of the single run and of the local union.
Writes output/widebox_wb18.json, output/wb18_weights_single_0.083.npy, output/wb18_weights_union_0.083.npy."""
import json, shutil
import numpy as np
def load(p):
    d = json.load(open(p)); return d if "master" in d else list(d.values())[0]
def per(p):
    return {"%g" % float(k.strip("()").split(",")[0]): float(np.mean(list(v.values()))) for k, v in load(p)["per_point"].items()}
out = dict(single_centre=per("output/p2wb18single_W_fac_s0.json"), pool_all=per("output/p2wb18pool_W_fac_s0.json"), local_union={})
for r in range(9100, 9106):
    out["local_union"].update(per("output/p2wb18local%d_W_fac_s0.json" % r))
out["summary"] = {k: float(np.mean(list(v.values()))) for k, v in out.items()}
json.dump(out, open("output/widebox_wb18.json", "w"), indent=1)
shutil.copy("output/models/W_P2wb18singlefac0_w_9100.npy", "output/wb18_weights_single_0.083.npy")
shutil.copy("output/models/W_P2wb18local9100fac0_w_9100.npy", "output/wb18_weights_union_0.083.npy")
for name in ("single", "union"):
    w = np.load("output/wb18_weights_%s_0.083.npy" % name); u = w*len(w)
    print(name, "weights at 0.083: max %.1f (unit mean), N_eff %.0f, decades spanned %.1f" % (u.max(), 1/np.sum(w**2), np.log10(u.max()/max(u[u > 0].min(), 1e-30))))
print(json.dumps(out["summary"]))
