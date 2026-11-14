#!/usr/bin/env python3
"""Physical before/after distributions with the same generator uncertainty prescription.

Uses only stored evaluations. The nominal predictions are the default-setting
predictions already evaluated in the local-validation archive. That archive's
parameter ranges are NOT used for the bands: both rows use Y_theta from the
full-range archive. No learning or calculated-moment variations enter these
generator bands.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import figs_r8 as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variations", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--published-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    full = np.load(args.variations, allow_pickle=True)
    nominal = np.load(args.nominal, allow_pickle=True)
    check = np.load(args.published_check, allow_pickle=True)
    labels = [str(x) for x in full["labels"]]
    assert labels == [str(x) for x in nominal["labels"]]
    assert str(full["mix"]) == str(nominal["mix"]) == "linear"
    n = len(labels)
    check_json = json.loads(args.published_check.with_suffix(".json").read_text())
    assert np.allclose(nominal["centre"], check_json["central"]["theta"], rtol=0, atol=1e-14)
    assert np.isclose(nominal["centre"][-1], .5)
    assert len(full["Y_theta"]) == 2000
    assert np.isclose(full["lo"][-1], 0) and np.isclose(full["hi"][-1], 1)
    indices = {
        "thrust": np.array([i for i, v in enumerate(labels) if v.startswith("thr_")]),
        "nch": np.array([i for i, v in enumerate(labels) if v.startswith("nch_")]),
    }
    edges = {k: full["thrust_edges" if k == "thrust" else "nch_edges"].copy()
             for k in indices}
    assert np.isinf(edges["nch"][-1])
    edges["nch"][-1] = 48  # display width only: the bin contains every n_ch >= 46
    result = {}
    for row, name in enumerate(("before", "after")):
        sl = slice(row*n, (row+1)*n)
        y = nominal["y_central"][sl]
        samples = full["Y_theta"][:, sl]
        sd = samples.std(axis=0, ddof=1)
        for key, ii in indices.items():
            result[name, key] = dict(nominal=y[ii], sd=sd[ii])
        suffix = "_before" if name == "before" else ""
        archived = np.r_[check[f"h_central{suffix}_low"], check[f"h_central{suffix}"]]
        archived_sd = np.r_[check[f"gen_hist{suffix}_low"], check[f"gen_hist{suffix}"]]
        assert np.allclose(result[name, "thrust"]["nominal"], archived, rtol=2e-8, atol=1e-12)
        assert np.allclose(result[name, "thrust"]["sd"], archived_sd, rtol=1e-12, atol=1e-14)
        assert np.isclose(y[indices["nch"]].sum(), 1, atol=2e-7)
        assert np.isclose(y[indices["thrust"]] @ np.diff(edges["thrust"]), 1, atol=2e-7)

    F.style()
    fig, axs = plt.subplots(2, 2, figsize=(F.TW, 4.55), sharex="col", sharey="col")
    fig.subplots_adjust(left=.105, right=.985, bottom=.11, top=.88, hspace=.38, wspace=.26)
    blue, gray, green = F.METHOD, F.BASE_DARK, F.THEORY
    clipped = []
    for row, name in enumerate(("before", "after")):
        color = gray if row == 0 else blue
        for col, key in enumerate(("thrust", "nch")):
            ax = axs[row, col]
            y, sd, e = result[name, key]["nominal"], result[name, key]["sd"], edges[key]
            floor = 1e-5 if key == "thrust" else 0
            lower, upper = y-sd, y+sd
            ax.stairs(upper, e, baseline=np.maximum(lower, floor), fill=True,
                      color=F.GEN, alpha=.22, linewidth=0)
            ax.stairs(y, e, baseline=None, color=color, linewidth=1.35)
            if row == 1:
                ax.stairs(result["before", key]["nominal"], e, baseline=None,
                          color=gray, linewidth=1, linestyle="--")
            ax.set_title(("(a)" if col == 0 else "(b)")+" Generators alone" if row == 0
                         else ("(c)" if col == 0 else "(d)")+" Thrust calculation imposed",
                         loc="left", pad=8)
            if key == "thrust":
                ax.set_yscale("log"); ax.set_ylim(1e-5, 35); ax.set_xlim(0, .5)
                ax.axvspan(.05, 1/3, color=green, alpha=.055, zorder=-1)
                for bound in (.05, 1/3):
                    ax.axvline(bound, color=green, linewidth=.8, linestyle=":")
                ax.set_ylabel(r"$(1/\sigma)\,d\sigma/d\tau$")
                if row == 0:
                    ax.text(.53, .94, "constraint region", transform=ax.transAxes,
                            ha="center", va="top", color=green, fontsize=7.5)
            else:
                ax.set_xlim(0, 48); ax.set_ylim(0, .175)
                ax.set_ylabel("Fraction of events")
                ax.set_yticks([0, .05, .10, .15])
                ax.set_xticks([0, 10, 20, 30, 40, 47])
                ax.set_xticklabels(["0", "10", "20", "30", "40", r"$\geq46$"])
            if row == 1:
                ax.set_xlabel(r"Thrust, $\tau=1-T$" if key == "thrust"
                              else r"Charged multiplicity, $n_{\rm ch}$")
            F.trim(ax)
            clipped.extend({"row": name, "observable": key, "bin": int(i),
                            "true_lower_edge": float(lower[i]), "plot_floor": floor}
                           for i in np.flatnonzero(lower < floor))
    fig.legend(handles=[
        Line2D([], [], color=gray, lw=1.35, label="Before"),
        Line2D([], [], color=blue, lw=1.35, label="After"),
        Patch(facecolor=F.GEN, alpha=.22, label="Generator uncertainty"),
    ], loc="upper center", bbox_to_anchor=(.55, 1.0), ncol=3, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"))
    plt.close(fig)

    record = {
        "purpose": "Before/after physical distributions, same generator uncertainty in all four panels.",
        "source_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.variations, args.nominal, args.published_check)},
        "nominal_parameters": nominal["centre"].tolist(),
        "prior_lo": full["lo"].tolist(), "prior_hi": full["hi"].tolist(),
        "number_of_generator_variations": len(full["Y_theta"]),
        "same_parameter_draws_before_and_after": True,
        "band": "Nominal prediction plus/minus generator SD. Learning, theory and sample statistics are separate.",
        "final_state": "The common event definition used to impose the calculation; no measurement conversion.",
        "thrust_nominal_matches_published_raw_prediction": True,
        "thrust_sd_matches_published_generator_band": True,
        "nominal_histograms_normalized_over_all_events": True,
        "multiplicity_overflow": "n_ch >= 46, displayed with a finite width.",
        "lower_edges_outside_plot_range": clipped,
        "values": {name: {key: {field: values.tolist() for field, values in result[name, key].items()}
                          for key in indices} for name in ("before", "after")},
        "output": str(args.output),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in (
        "output", "number_of_generator_variations", "same_parameter_draws_before_and_after",
        "thrust_nominal_matches_published_raw_prediction", "thrust_sd_matches_published_generator_band",
        "nominal_histograms_normalized_over_all_events", "lower_edges_outside_plot_range")}, indent=2))


if __name__ == "__main__":
    main()
