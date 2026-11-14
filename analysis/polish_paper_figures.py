#!/usr/bin/env python3
"""Clearer paper figures from the existing result files; no fits or event generation."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator, PercentFormatter
import numpy as np

import figs_r8 as F


def export(fig, stem, record):
    out = Path(F.OUT)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / ("r8_"+stem+".png"), dpi=240, bbox_inches=None)
    F.save(fig, stem)
    (out / ("r8_"+stem+".json")).write_text(json.dumps(record, indent=2)+"\n")


def draw_coverage():
    files = os.environ.get(
        "COV_IN", "output/wifi_coverage_MIXGEOV3S9.json "
        "output/wifi_coverage_MIXGEOV3S9s1.json "
        "output/wifi_coverage_MIXGEOV3S9s2.json").split()
    points = []
    for filename in files:
        local = json.loads(Path(filename).read_text())["points"]
        assert sum(int(p["rid"])//1000 == 7 for p in local) == 24
        assert sum(int(p["rid"])//1000 == 8 for p in local) == 24
        points.extend(local)
    obs = [("thrust", r"$\langle1-T\rangle$"), ("B_total", r"$\langle B_{\rm tot}\rangle$"),
           ("rho_heavy", r"$\langle\rho_H\rangle$"), ("nch", r"$\langle n_{\rm ch}\rangle$"),
           ("nbaryon", r"$\langle n_{\rm b}\rangle$")]
    groups = {"Sherpa": [p for p in points if int(p["rid"])//1000 == 7],
              "Herwig": [p for p in points if int(p["rid"])//1000 == 8]}
    for point in points:
        for key, _ in obs:
            v = point[key]
            variance = v["stat_pred"]**2+v["stat_truth"]**2
            assert np.isclose((v["pred"]-v["truth"])/np.sqrt(variance), v["z_without"], rtol=1e-10, atol=1e-10)
            assert np.isclose((v["pred"]-v["truth"])/np.sqrt(variance+v["learn"]**2), v["z_with"], rtol=1e-10, atol=1e-10)
    limits = [max(abs(p[k][z]) for p in points for k, _ in obs)
              for z in ("z_without", "z_with")]
    limits = [np.ceil(v+.65) for v in limits]
    fig, axes = plt.subplots(2, 2, figsize=(F.TW, 4.55), sharex=True, sharey="row")
    fig.subplots_adjust(left=.115, right=.985, bottom=.095, top=.93, hspace=.36, wspace=.16)
    rng = np.random.default_rng(5)
    record = {"purpose": "Same residuals, different uncertainty denominators, separated by generator.",
              "source_files": files, "panels": {}, "pooled_rms_by_observable": {}}
    for row, (z, ylabel, color) in enumerate((
            ("z_without", r"$\Delta/\sigma_{\rm stat}$", F.BASE_DARK),
            ("z_with", r"$\Delta/\sqrt{\sigma_{\rm stat}^2+\sigma_{\rm learn}^2}$", F.LEARN_TEXT))):
        for col, generator in enumerate(("Sherpa", "Herwig")):
            ax = axes[row, col]
            ax.axhspan(-1, 1, color=F.BASE, alpha=.10, lw=0)
            ax.axhline(0, color=F.REF, lw=.65)
            for y in (-2, 2):
                ax.axhline(y, color=F.REF, lw=.65, ls=(0, (3, 3)))
            all_z = []
            per_obs = {}
            for i, (key, label) in enumerate(obs):
                values = np.array([p[key][z] for p in groups[generator]])
                assert np.all(np.isfinite(values))
                all_z.extend(values)
                ax.scatter(i+rng.uniform(-.16, .16, len(values)), values,
                           s=6.5, color=color, alpha=.72, linewidths=0, zorder=3)
                per_obs[key] = float(np.sqrt(np.mean(values**2)))
            rms = float(np.sqrt(np.mean(np.asarray(all_z)**2)))
            ax.text(.98, .95, f"RMS {rms:.2f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1))
            ax.set_ylim(-limits[row], limits[row])
            ax.set_xlim(-.5, 4.5)
            ax.set_title(f"({chr(97+2*row+col)}) {generator}", loc="left", fontsize=10, pad=7)
            ax.set_xticks(range(5), [label for _, label in obs])
            ax.tick_params(axis="x", length=0, labelsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            F.trim(ax)
            record["panels"][f"{generator}/{z}"] = dict(
                number_of_points_per_observable=len(groups[generator]), rms_all=rms, rms_by_observable=per_obs)
    for key, _ in obs:
        record["pooled_rms_by_observable"][key] = {
            z: float(np.sqrt(np.mean([p[key][z]**2 for p in points])))
            for z in ("z_without", "z_with")}
    record["points_beyond_two_with_learning"] = {
        generator: sum(abs(p[key]["z_with"]) > 2 for p in rows for key, _ in obs)
        for generator, rows in groups.items()}
    assert record["points_beyond_two_with_learning"] == {"Sherpa": 0, "Herwig": 11}
    export(fig, "coverage", record)


def draw_window(data=None):
    d = F.window_split(return_data=True) if data is None else data
    assert d["data_error_model"] == "stat in quadrature, sys fully correlated"
    suffix = "_lin" if d["linear_mixture"] else ""
    e, cs = d["edges"], d["calculated_shape"]
    fig, axes = plt.subplots(1, 2, figsize=(F.TW, 2.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.96, bottom=.34, top=.91, wspace=.15)
    before = d["before"]/cs
    after = d["after"]/cs
    curves = [before+d["generator_before"]/cs, before-d["generator_before"]/cs,
              after+d["theory_up"]/cs, after-d["theory_down"]/cs]
    for generator in ("sherpa", "herwig"):
        curves.extend(d[generator][key]/cs for key in ("h_gen_only", "h_imposed"))
    ylim, ticks = F.nice_range(curves, pad=.09)
    for col, ax in enumerate(axes):
        color = F.BASE_DARK if col == 0 else F.METHOD
        center = before if col == 0 else after
        upper = center+(d["generator_before"] if col == 0 else d["theory_up"])/cs
        lower = center-(d["generator_before"] if col == 0 else d["theory_down"])/cs
        band_color = F.GEN if col == 0 else F.THEORY
        ax.stairs(upper, e, baseline=lower, fill=True, color=band_color, alpha=.22, lw=0)
        ax.stairs(center, e, baseline=None, color=color, lw=1.35)
        for generator, linestyle in (("sherpa", (0, (4, 2))), ("herwig", (0, (1, 1.3)))):
            key = "h_gen_only" if col == 0 else "h_imposed"
            ax.stairs(d[generator][key]/cs, e, baseline=None, color=color, lw=1, ls=linestyle)
        ax.axhline(1, color=F.REF, lw=.65, zorder=0)
        ax.set_xlim(.05, .30); ax.set_ylim(*ylim); ax.set_yticks(ticks)
        ax.set_xticks([.05, .10, .15, .20, .25, .30])
        ax.set_xlabel(r"$\tau=1-T$")
        ax.set_title("(a) Before" if col == 0 else "(b) After", loc="left", fontsize=10, pad=6)
        F.trim(ax)
    axes[0].set_ylabel("Shape / calculation", fontsize=10)
    residual = 100*np.max(d["generator_after"]/d["after"])
    fig.legend(handles=[
        Line2D([], [], color=F.DATA, lw=1.3, label="Mixture"),
        Patch(facecolor=F.GEN, alpha=.22, label="Generator"),
        Line2D([], [], color=F.DATA, ls=(0, (4, 2)), lw=1, label="Sherpa"),
        Patch(facecolor=F.THEORY, alpha=.22, label="Theory"),
        Line2D([], [], color=F.DATA, ls=(0, (1, 1.3)), lw=1, label="Herwig"),
    ], loc="lower center", bbox_to_anchor=(.55, .015), ncol=3, fontsize=10)
    export(fig, "window_shape"+suffix, {
        "normalization_interval": [.05, .30],
        "final_state": "The common generator event definition on which the moments are imposed.",
        "reference": "The calculated shape normalized over the same twelve bins.",
        "before_mixture": d["before"].tolist(), "after_mixture": d["after"].tolist(),
        "generator_sd_before": d["generator_before"].tolist(),
        "generator_sd_after": d["generator_after"].tolist(),
        "theory_up": d["theory_up"].tolist(), "theory_down": d["theory_down"].tolist(),
        "maximum_residual_generator_percent": float(residual),
        "old_relative_width_outline_removed": True})

    fractions = d["fractions"]
    rows = [("Sherpa", "SA", "SI"), ("Herwig", "HA", "HI"),
            ("Mixture", "mixA", "mixI")]
    fig, ax = plt.subplots(figsize=(F.TW, 2.65))
    fig.subplots_adjust(left=.19, right=.985, bottom=.30, top=.96)
    y = np.arange(5)[::-1]
    ax.axvspan(d["data_fraction"]-d["data_error"], d["data_fraction"]+d["data_error"],
               color=F.BASE, alpha=.12, lw=0, zorder=0)
    ax.axvline(d["data_fraction"], color=F.REF, lw=.7, ls=(0, (3, 3)), zorder=1)
    for (label, key0, key1), yi in zip(rows, y[:3]):
        ax.plot([fractions[key0], fractions[key1]], [yi+.11, yi-.11], color=F.BASE, lw=.65, zorder=2)
        ax.plot(fractions[key0], yi+.11, "o", ms=4, mfc="white", mec=F.BASE_DARK, mew=.9, zorder=5)
        ax.plot(fractions[key1], yi-.11, "o", ms=4, color=F.METHOD, zorder=6)
    fy = y[2]-.11
    ax.plot([fractions["mixI"]-d["fraction_generator"], fractions["mixI"]+d["fraction_generator"]],
            [fy, fy], color=F.GEN, alpha=.27, lw=7, solid_capstyle="butt", zorder=2)
    ax.plot([fractions["mixI"]-d["fraction_learning"], fractions["mixI"]+d["fraction_learning"]],
            [fy, fy], color=F.LEARN, lw=2.2, solid_capstyle="butt", zorder=4)
    ax.errorbar(fractions["calc"], y[3],
                xerr=[[d["fraction_theory_down"]], [d["fraction_theory_up"]]],
                fmt="s", ms=4, color=F.THEORY, lw=1.6, capsize=2, zorder=5)
    ax.errorbar(d["data_fraction"], y[4], xerr=d["data_error"],
                fmt="o", ms=4, color=F.DATA, lw=1.1, capsize=2, zorder=5)
    ax.set_yticks(y, [r[0] for r in rows]+["Calculation", "DELPHI"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-.5, 4.5)
    endpoints = [fractions[k] for k in fractions]
    endpoints.extend([fractions["mixI"]-d["fraction_generator"], fractions["mixI"]+d["fraction_generator"],
                      fractions["calc"]-d["fraction_theory_down"], fractions["calc"]+d["fraction_theory_up"],
                      d["data_fraction"]-d["data_error"], d["data_fraction"]+d["data_error"]])
    span = max(endpoints)-min(endpoints)
    ax.set_xlim(min(endpoints)-.055*span, max(endpoints)+.055*span)
    ax.set_xlabel(r"$P(0.05\leq\tau<0.30)$")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.spines["left"].set_visible(False); F.trim(ax)
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="white", mec=F.BASE_DARK, label="Before"),
        Line2D([], [], ls="", marker="o", color=F.METHOD, label="After"),
        Line2D([], [], color=F.GEN, alpha=.5, lw=4, label="Generator"),
        Line2D([], [], color=F.LEARN, lw=2, label="Learning"),
    ], loc="lower center", bbox_to_anchor=(.55, .015), ncol=4, fontsize=10, columnspacing=1.1)
    export(fig, "window_fraction"+suffix, {
        "interval": [.05, .30], "final_state": "Converted to the DELPHI definition.",
        "nominal_predictions": fractions, "generator_sd_of_mixture": d["fraction_generator"],
        "learning_sd_of_mixture": d["fraction_learning"],
        "theory_up_of_calculation": d["fraction_theory_up"], "theory_down_of_calculation": d["fraction_theory_down"],
        "data_fraction": d["data_fraction"], "data_uncertainty": d["data_error"],
        "data_error_model": d["data_error_model"],
        "individual_generator_markers": "Nominal predictions, without their parameter-variation bands."})


def draw_budget():
    filename = Path(os.environ.get("BANDS_JSON", "output/three_bands_MIXGEOV3S9.json"))
    d = json.loads(filename.read_text())
    lin = F.linear_inputs((str(filename), d))
    keys = ["tau_win", "pwin", "thrust", "B_total", "rho_heavy", "nch", "nbaryon"]
    labels = [r"$\langle\tau\rangle_{\rm win}$", r"$P_{\rm win}$", r"$\langle\tau\rangle$",
              r"$\langle B_{\rm tot}\rangle$", r"$\langle\rho_H\rangle$", r"$\langle n_{\rm ch}\rangle$",
              r"$\langle n_{\rm b}\rangle$"]
    c, g = d["central"], d["generator_band"]
    pred = np.array([c["imposed"][k] for k in keys])
    get = lambda q: 100*np.array([q[k] for k in keys])/np.abs(pred)
    series = [
        ("Generator before", 100*np.array([g["before"][k]["sd"] for k in keys])
         /np.abs(np.array([c["generator_only"][k] for k in keys])), F.BASE_DARK, "x", .18),
        ("Generator after", 100*np.array([g["after"][k]["sd"] for k in keys])/np.abs(pred), F.GEN, "s", .09),
        ("Theory +", get(c["theory_up"]), F.THEORY, "^", 0),
        (r"Theory $-$", get(c["theory_down"]), F.THEORY, "v", -.09),
        ("Learning", get(c["learning_band"]), F.LEARN, "D", -.18),
    ]
    fig, ax = plt.subplots(figsize=(F.TW, 3.9))
    fig.subplots_adjust(left=.16, right=.985, bottom=.15, top=.79)
    ys = np.arange(7)[::-1]
    all_values = []
    for name, values, color, marker, offset in series:
        assert np.all(values > 0)
        ax.scatter(values, ys+offset, color=color, marker=marker, s=13, linewidths=.8, label=name, zorder=3)
        all_values.extend(values)
    for boundary in (5.5, 4.5, 1.5):
        ax.axhline(boundary, color=F.BASE, lw=.55, alpha=.25)
    ax.set_yticks(ys, labels, fontsize=10)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(10**np.floor(np.log10(min(all_values)*.6)), max(all_values)*1.7)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="x", color=F.BASE, alpha=.16, linewidth=.6)
    ax.set_xlabel(r"Relative uncertainty [\%]", fontsize=10)
    ax.set_ylim(-.55, 6.55)
    F.trim(ax)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle="", markersize=4, label=name)
                        for name, values, color, marker, offset in [series[i] for i in (0, 2, 4, 1, 3)]],
               loc="upper center", bbox_to_anchor=(.55, .99), ncol=2, fontsize=10, columnspacing=1.3)
    export(fig, "obs"+("_lin" if lin else ""), {
        "source_file": str(filename), "quantity_order": keys,
        "percentages": {name: values.tolist() for name, values, color, marker, offset in series},
        "denominators": "Before-generator SD uses the before prediction; all after components use the after prediction.",
        "all_positive_values_displayed": True, "no_learning_component_hidden": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", choices=["coverage", "window", "budget"])
    args = parser.parse_args()
    F.style()
    for key in args.figures or ["coverage", "window", "budget"]:
        {"coverage": draw_coverage, "window": draw_window, "budget": draw_budget}[key]()


if __name__ == "__main__":
    main()
