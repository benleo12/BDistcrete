"""The K scan of Sec. 3.6 (Fig. 6) at the paper standard, in the form figs_r8.py kscan reads: for each
stage, the closure width at every number of terms K on the standard and the strict test, with the
spread over the held-out points (the standard deviation of the per-point widths) as in the first
scan. K = 48 is the first training of the Phase 2 cell. Writes output/rank_scan_final.json.
KSCAN_REQUIRE=1 (the rerun) makes a missing result file an error instead of a gap in the figure. In the
rerun (RUN holds the start mark output/.lmx_rerun_start, or LMX_CHECK=1) every result read must carry the
fingerprint of the current mark and code (symmetry/rerun/lmx_fingerprint.py); any other is refused."""
import json, os
import numpy as np
REQUIRE = os.environ.get('KSCAN_REQUIRE', '') == '1'
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
        raise RuntimeError('results without the fingerprint of this rerun:\n' + '\n'.join(f'  {p}: {w}' for p, w in bad))
STAGES = {"A": (1, "p2augK{K}{t}_A_fac_s0.json", "p2augfull{t}_A_fac_s0.json"),
          "B": (2, "p2augK{K}{t}_B_fac_s0.json", "p2aug{t}_B_fac_s0.json"),
          "C": (3, "p2augK{K}{t}_C_fac_s0.json", "p2augfull{t}_C_fac_s0.json"),
          "E": (8, "p2v3K{K}{t}_E_fac_s0.json", "p2v3full{t}_E_fac_s0.json")}
KS = {"A": [1, 2, 3, 4, 6, 8, 16], "B": [1, 2, 3, 4, 6, 8, 16], "C": [1, 2, 3, 4, 6, 8, 16],
      "E": [1, 2, 4, 6, 8, 9, 10, 12, 16, 24, 32, 45]}
cells = {(S, t, K): "output/" + (ref.format(t=t) if K == 48 else pat.format(K=K, t=t))
         for S, (d, pat, ref) in STAGES.items() for t in ("", "strict") for K in KS[S] + [48]}
missing = [f for f in cells.values() if not os.path.exists(f)]
if missing and REQUIRE:
    raise FileNotFoundError(f'missing input {missing[0]}' + (f' and {len(missing) - 1} more' if len(missing) > 1 else ''))
lmx_check([f for f in cells.values() if os.path.exists(f)])
out = {}
for S, (d, pat, ref) in STAGES.items():
    row = dict(d=d, predicted_cliff=d + 1, K2=1 + d + d*(d + 1)//2, rank_ens=4)
    for t, key in (("", "scan"), ("strict", "scan_strict")):
        sc = {}
        for K in KS[S] + [48]:
            f = cells[(S, t, K)]
            if not os.path.exists(f):
                continue
            r = load(f); pp = [float(np.mean(list(v.values()))) for v in r["per_point"].values()]
            sc[str(K)] = dict(width=float(r["master"]), spread=float(np.std(pp)), n_points=len(pp))
        row[key] = sc
    out[S] = row
    print(S, " ".join("K%s:%.3f/%.3f" % (k, v["width"], row["scan_strict"].get(k, {}).get("width", float("nan"))) for k, v in row["scan"].items()))
json.dump(out, open("output/rank_scan_final.json", "w"), indent=1)
