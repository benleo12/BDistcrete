"""The K scan of App. C at the paper standard, in the form figs_pro.py head reads: for each stage, the
closure width at every number of terms K on the standard and the strict test, with the spread over
the held-out points (the standard deviation of the per-point widths) as in the first scan. K = 48 is
the first training of the Phase 2 cell. Writes output/rank_scan_final.json."""
import json, os
import numpy as np
def load(p):
    d = json.load(open(p)); return d if "master" in d else list(d.values())[0]
STAGES = {"A": (1, "p2augK{K}{t}_A_fac_s0.json", "p2augfull{t}_A_fac_s0.json"),
          "B": (2, "p2augK{K}{t}_B_fac_s0.json", "p2aug{t}_B_fac_s0.json"),
          "C": (3, "p2augK{K}{t}_C_fac_s0.json", "p2augfull{t}_C_fac_s0.json"),
          "E": (8, "p2v3K{K}{t}_E_fac_s0.json", "p2v3full{t}_E_fac_s0.json")}
KS = {"A": [1, 2, 3, 4, 6, 8, 16], "B": [1, 2, 3, 4, 6, 8, 16], "C": [1, 2, 3, 4, 6, 8, 16],
      "E": [1, 2, 4, 6, 8, 9, 10, 12, 16, 24, 32, 45]}
out = {}
for S, (d, pat, ref) in STAGES.items():
    row = dict(d=d, predicted_cliff=d + 1, K2=1 + d + d*(d + 1)//2, rank_ens=4)
    for t, key in (("", "scan"), ("strict", "scan_strict")):
        sc = {}
        for K in KS[S] + [48]:
            f = "output/" + (ref.format(t=t) if K == 48 else pat.format(K=K, t=t))
            if not os.path.exists(f):
                continue
            r = load(f); pp = [float(np.mean(list(v.values()))) for v in r["per_point"].values()]
            sc[str(K)] = dict(width=float(r["master"]), spread=float(np.std(pp)), n_points=len(pp))
        row[key] = sc
    out[S] = row
    print(S, " ".join("K%s:%.3f/%.3f" % (k, v["width"], row["scan_strict"].get(k, {}).get("width", float("nan"))) for k, v in row["scan"].items()))
json.dump(out, open("output/rank_scan_final.json", "w"), indent=1)
