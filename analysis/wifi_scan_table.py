#!/usr/bin/env python3
"""The ensemble-size scan of the learning-uncertainty fit as LaTeX rows: for each ensemble size,
the number of basis weights, the loss on the comparison halves and the closure width with the mean
of members and with the WiFi weights, the baryon-count closure with the WiFi weights, and the
median learning uncertainty of the log weight over the fit set.

    python wifi_scan_table.py MIXGEO MIXGEO8 MIXGEO16 MIXGEO32
"""
import sys, json, os
rows = []
for tag in sys.argv[1:]:
    fit, diag = f'output/wifi_{tag}_fit.json', f'output/wifi_{tag}_diag.json'
    if not (os.path.exists(fit) and os.path.exists(diag)):
        print(f'% {tag}: missing {fit if not os.path.exists(fit) else diag}'); continue
    F, D = json.load(open(fit)), json.load(open(diag))
    N = int(round((F['n_weights'] - 1)**0.5))
    rows.append((N, F['n_weights'], D['published']['held_loss'], D['published']['closure'], D['wifi']['held_loss'], D['wifi']['closure'],
                 D['wifi']['per_obs']['nbaryon'], 100*F['logw_sigma_median']))
print(r'members & basis weights & \multicolumn{2}{c}{mean of members} & \multicolumn{3}{c}{WiFi weights} & learning uncertainty \\')
print(r' & & loss & closure & loss & closure & baryon count & per event (percent) \\')
for r in rows:
    print(f'${r[0]}$ & ${r[1]}$ & ${r[2]:.3f}$ & ${r[3]:.2f}$ & ${r[4]:.3f}$ & ${r[5]:.2f}$ & ${r[6]:.2f}$ & ${r[7]:.1f}$ \\\\')
