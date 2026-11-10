# Checks from the figure review of 2026-11-10

Small results of the checks that the paper's App. A and Secs. 3.2 and 3.5 cite. None of them
changes a published number; the paper states what each one found. The jobs ran on Symmetry in
`/mnt/beegfs/bassi/reweighting/lmx_runs/review_1110/<check>/`, which also holds their logs and
the larger files (event samples, npz).

| check | script (in `analysis/`) | files here |
|---|---|---|
| null of the closure width (Sec. 3.2) | `closure_null_sim.py` | `null_sim.json` |
| the choices of checkpoint and kappa with the true null as target (App. A.2) | `closure_model_check.py`, `recalibrate_T.py` (job 595192) | `model_check.json`, `recal_T_*.json` |
| learning covariance with each event's contributions joined (App. A.3, Sec. 3.5) | `wifi_joint_cov.py` (job 595180) | `wifi_*_cov.json`, `cmp_cov_*.json`, `cmp_bands_*.json`, `three_bands_*_joint.json`, `wifi_coverage_*_{joint,strat}.json` |
| beam-photon-free shapes, their fallback and the 100-particle cap (App. A) | `audit_obs_defs.py` (job 595185) | `audit_obs_defs.json` |
| fixed final-state ratios on events recorded in both conventions (App. A) | `conv_weighted_check.py` (job 595177 for the new runs) | `conv_default.json`, `conv_sherpa_points.json`, `conv_herwig_points.json` |

`r2_ladder.py` and `wifi_fit.py` are unchanged, since their checksums identify the code that
produced the stored results; `docs/REPRODUCE.md` notes the true null of the closure width.
