#!/usr/bin/env python3
"""Reconstruct the theory rates used for rate-and-shape matching.

Uses the existing NNLL+NLO cumulants and dispersive shift at the cached
calculation point. No fits are run and no published targets are overwritten.
The output retains the existing prescription of interpolating conditional
moments and window probabilities separately between coupling nodes.
--release-targets optionally writes a Targets archive with the paired rates.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ANALYSIS = Path(__file__).resolve().parent
POWERS = {
    "logt": (0, 1), "tau": (1, 0), "tlogt": (1, 1),
    "log2t": (0, 2), "tau2": (2, 0), "tlog2t": (1, 2),
    "t2logt": (2, 1), "log3t": (0, 3), "tau3": (3, 0),
    "tlog3t": (1, 3), "t3logt": (3, 1), "t2log2t": (2, 2),
    "log4t": (0, 4), "tau4": (4, 0),
}


@contextmanager
def analysis_directory():
    """The existing theory modules resolve their data relative to analysis/."""
    previous = Path.cwd()
    os.chdir(ANALYSIS)
    try:
        yield
    finally:
        os.chdir(previous)


def relative_error(value, reference):
    value, reference = np.asarray(value), np.asarray(reference)
    return float(np.max(np.abs(value - reference) /
                        np.maximum(np.abs(reference), 1e-15)))


def build_targets(tolerance=1e-7):
    """Return central/variation targets and checks without writing any files."""
    os.environ["THRUST_ORDER"] = "2"
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))

    with analysis_directory():
        import np_shift as N
        import thrust_chain as TC

        if TC.ORDER != 2:
            raise RuntimeError("thrust_chain was previously imported at another order")
        cache_path = ANALYSIS / "output/thrust_targets_nlo_point.npz"
        point_path = ANALYSIS / "output/calc_point.json"
        G = np.load(cache_path, allow_pickle=False)
        point = json.loads(point_path.read_text())
        keys = [str(k) for k in G["keys"]]
        window = np.asarray(G["window"], dtype=float)
        if not np.array_equal(window, [0.05, 1.0 / 3.0]):
            raise ValueError(f"Unexpected constraint window: {window}")
        asc, a0 = float(point["alpha_s"]), float(point["alpha_0"])
        if not (asc == float(G["alphas"][0]) and a0 == float(G["alpha0"][0])):
            raise ValueError("Calculation point and stored targets differ")
        sub_order = int(point.get("np_sub_order", 2))
        if sub_order != int(G["sub_order_central"]):
            raise ValueError("Calculation point and target subtraction orders differ")
        milan = float(G["milan"])
        ia = int(np.clip(np.searchsorted(TC.ASGRID, asc) - 1,
                         0, len(TC.ASGRID) - 2))
        a1, a2 = TC.ASGRID[ia:ia + 2]
        w = (asc - a1) / (a2 - a1)
        nodes = [(a1, 1.0 - w), (a2, w)]
        edges = np.linspace(*window, 200001)
        centers = 0.5 * (edges[1:] + edges[:-1])
        functions = np.array([centers ** POWERS[k][0] *
                              np.log(centers) ** POWERS[k][1] for k in keys])
        cumulants = {}

        def cumulant(mu, xv, scheme, alpha):
            key = (mu, xv, scheme, alpha)
            if key not in cumulants:
                tau, matched = TC.build(mu, xv, scheme, alpha)
                before = matched / matched[-1]
                clean, moved = N.sanitize(before)
                diagnostic = {
                    "endpoint_before_extra_division": float(matched[-1]),
                    "endpoint_after_sanitization": float(clean[-1]),
                    "minimum_before_sanitization": float(before.min()),
                    "minimum_after_sanitization": float(clean.min()),
                    "decreasing_intervals_before_sanitization":
                        int(np.count_nonzero(np.diff(before) < -1e-14)),
                    "decreasing_intervals_after_sanitization":
                        int(np.count_nonzero(np.diff(clean) < -1e-14)),
                    "max_sanitization_change": float(moved),
                }
                cumulants[key] = (tau, before, clean, diagnostic)
            return cumulants[key]

        def evaluate(mu, xv, scheme, *, milan_value=milan,
                     subtraction_order=sub_order):
            moments = np.zeros(len(keys))
            integrals = np.zeros(len(keys))
            pwin = pwin_before_sanitization = pwin_030 = 0.0
            node_diagnostics = []
            old_milan, old_sub = N.MILAN, N.SUB_ORDER
            N.MILAN, N.SUB_ORDER = milan_value, subtraction_order
            try:
                for alpha, weight in nodes:
                    tau, before, clean, diagnostic = cumulant(mu, xv, scheme, alpha)
                    shift = float(N.shift(mu, a0, asmz=alpha))
                    shifted_tau = tau + shift
                    mass = np.diff(N.cum_eval(edges, shifted_tau, clean))
                    probability = float(mass.sum())
                    before_mass = float(np.diff(N.cum_eval(window, shifted_tau, before))[0])
                    integral = functions @ mass
                    moments += weight * integral / probability
                    integrals += weight * integral
                    pwin += weight * probability
                    pwin_before_sanitization += weight * before_mass
                    pwin_030 += weight * float(np.diff(
                        N.cum_eval([window[0], 0.30], shifted_tau, clean))[0])
                    node_diagnostics.append({
                        **diagnostic, "alpha_s": float(alpha),
                        "interpolation_weight": float(weight), "shift": shift,
                        "pwin_before_sanitization": before_mass,
                        "pwin_after_sanitization": probability,
                        "sanitization_delta_pwin": probability - before_mass,
                        "minimum_window_cell_probability": float(mass.min()),
                        "monotone_window_cumulant": bool(np.all(mass >= -1e-14)),
                    })
            finally:
                N.MILAN, N.SUB_ORDER = old_milan, old_sub
            return {
                "pwin": pwin, "moments": moments.tolist(),
                "rate_weighted_moments": (pwin * moments).tolist(),
                "direct_cumulant_moment_integrals": integrals.tolist(),
                "separate_interpolation_relative_difference":
                    relative_error(pwin * moments, integrals),
                "pwin_005_030": pwin_030,
                "pwin_before_sanitization": pwin_before_sanitization,
                "sanitization_delta_pwin": pwin - pwin_before_sanitization,
                "nodes": node_diagnostics,
            }

        central = evaluate(1.0, 1.0, "logR")
        variations = {}
        errors = {
            "central_rate": relative_error(central["pwin"], G["pwin"][0, 0]),
            "central_moments": relative_error(central["moments"], G["central"][0, 0]),
        }
        for index, label in enumerate(G["scale_labels"]):
            label = str(label)
            mu, xv, scheme, cfac = ast.literal_eval(label)
            if cfac != 1.0:
                raise ValueError("Unexpected fixed-order variation in NLO targets")
            key = "scale:" + label
            row = evaluate(mu, xv, scheme)
            variations[key] = row
            errors[key + ":rate"] = relative_error(row["pwin"], G["pwin_scale"][0, 0, index])
            errors[key + ":moments"] = relative_error(row["moments"], G["scale"][0, 0, index])
        for index, label in enumerate(G["np_labels"]):
            label = str(label)
            if label == "milan_up":
                args = {"milan_value": 1.2 * milan}
            elif label == "milan_dn":
                args = {"milan_value": 0.8 * milan}
            elif label.startswith("sub"):
                args = {"subtraction_order": int(label[3:])}
            else:
                raise ValueError(f"Unknown NP variation: {label}")
            row = evaluate(1.0, 1.0, "logR", **args)
            variations["np:" + label] = row
            errors["np:" + label + ":moments"] = relative_error(row["moments"], G["np_vars"][0, 0, index])

        binned_path = ANALYSIS / "output/calc_bins_nlo_point.json"
        binned = json.loads(binned_path.read_text())["delphi"]
        lo, hi = np.asarray(binned["lo"]), np.asarray(binned["hi"])
        keep = (lo >= window[0] - 1e-12) & (hi <= 0.30 + 1e-12)
        for label, values in binned["calc"].items():
            row = central if label == "central" else variations[
                ("np:" if label in {str(x) for x in G["np_labels"]} else "scale:") + label]
            reference = float(np.sum(np.asarray(values)[keep] * (hi - lo)[keep]))
            errors["binned_005_030:" + label] = relative_error(row["pwin_005_030"], reference)
        maximum_error = max(errors.values())
        if maximum_error > tolerance:
            raise AssertionError(f"Cached targets disagree: relative error {maximum_error:.3g}")

        sources = [cache_path, point_path, binned_path,
                   ANALYSIS / TC.CSV, ANALYSIS / "output/eerad_LO.npy",
                   ANALYSIS / "output/eerad_NLO.npy", ANALYSIS / "output/eerad_NNLO_cum.npy",
                   ANALYSIS / "targets_nlo_point.py", ANALYSIS / "thrust_chain.py",
                   ANALYSIS / "np_shift.py", ANALYSIS / "match_v3.py", Path(__file__)]
        return {
            "purpose": "Isolated rate-and-shape matching trial; published target files unchanged.",
            "window": window.tolist(), "keys": keys,
            "alpha_s": asc, "alpha_0": a0, "milan": milan,
            "sub_order_central": sub_order, "fixed_order": 2,
            "normalization": {
                "definition": "Matched cumulant normalized to its endpoint before the dispersive shift.",
                "build_already_normalizes_endpoint": True,
                "additional_division": "The existing target builder divides by that endpoint again.",
                "sanitization": "Clip at zero, then take the running maximum before shifting.",
                "scope": "Full-window rates inherit the calculation's normalization over the full thrust range, including the peak and tail.",
                "interpolation": "Conditional moments and window rates are separately linearly interpolated in alpha_s, matching the published cache. Direct cumulant integrals are recorded separately.",
            },
            "central": central, "variations": variations,
            "checks": {"tolerance": tolerance, "max_relative_cache_error": maximum_error,
                       "relative_errors": errors,
                       "all_window_cumulants_monotone": all(
                           node["monotone_window_cumulant"]
                           for row in [central, *variations.values()] for node in row["nodes"])},
            "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Diagnostic JSON destination")
    parser.add_argument("--release-targets", type=Path,
                        help="Optional rate-and-shape Targets NPZ or JSON destination")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.parent == (ANALYSIS / "output").resolve():
        parser.error("Write trial targets outside the published output directory")
    result = build_targets()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.release_targets:
        sys.path.insert(0, str(ANALYSIS/'release'))
        from gentune.maxent import Targets, load_targets
        shape = load_targets(ANALYSIS/'release/targets_thrust_nlo.npz')
        rates = {label.split(':', 1)[1]: row['pwin'] for label, row in result['variations'].items()}
        calc = Targets(shape.names, shape.c, shape.sigma, shape.window,
                       variations=shape.variations, floor_rel=shape.floor_rel,
                       powers=shape.powers, rate=result['central']['pwin'], rate_variations=rates,
                       meta=dict(name='targets_thrust_rate_shape', version='2.0',
                                 alpha_s=result['alpha_s'], alpha_0=result['alpha_0'],
                                 pwin=result['central']['pwin'],
                                 prescription='Conditional-moment fit followed by exact window-rate matching.',
                                 rate_builder='analysis/rate_matching_targets.py',
                                 rate_builder_output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                 normalization='Window integral of the inclusive-normalized NNLL+NLO spectrum, with the same dispersive shift and variations as the conditional moments.'))
        destination=args.release_targets.expanduser().resolve()
        destination.parent.mkdir(parents=True,exist_ok=True)
        calc.save(destination)
        print(f'wrote {destination}')
    print(f"central pwin = {result['central']['pwin']:.16g}")
    for label, row in result["variations"].items():
        if label.startswith("np:"):
            print(f"{label} pwin = {row['pwin']:.16g}")
    print(f"max relative cache error = {result['checks']['max_relative_cache_error']:.3g}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
