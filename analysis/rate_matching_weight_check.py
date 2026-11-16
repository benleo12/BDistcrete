#!/usr/bin/env python3
"""Check sequential rate matching on the released reference-event weights.

This isolated experiment first reproduces the published central shape fit, then
rescales its weights inside and outside the full calculation window separately
for Sherpa and Herwig. It does not change the trained head or published results.
Only small prediction arrays are saved unless --save-weights is requested.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RELEASE = HERE / "release"
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / "examples"))
from gentune import axes, load_head, mixture
from gentune.maxent import load_targets
from reproduce_paper import OBS, Reference, factor_vectors, rel


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_statistics(w):
    return dict(normalization=float(w.sum()), finite=bool(np.all(np.isfinite(w))),
                positive=bool(np.all(w > 0)), minimum=float(w.min()), maximum=float(w.max()),
                maximum_over_mean=float(len(w) * w.max()), n_eff=float(1 / (w @ w)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=RELEASE / "reference_MIX17.npz")
    parser.add_argument("--head", type=Path, default=RELEASE / "models/MIX17_head.npz")
    parser.add_argument("--targets", type=Path, default=RELEASE / "targets_thrust_nlo.npz")
    parser.add_argument("--paper", type=Path, default=HERE / "output/three_bands_MIXGEOV3S9_lin.npz")
    parser.add_argument("--envelope-prefix", type=Path, default=HERE / "output/envelope_test_MIXGEOV3S9")
    parser.add_argument("--factors", type=Path, default=RELEASE / "examples/data/convention_factors.json")
    parser.add_argument("--target-fraction", type=float)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True, help="directory for the isolated JSON and NPZ")
    parser.add_argument("--save-weights", action="store_true", help="also save the full weights in a separate compressed archive")
    args = parser.parse_args()
    started = time.perf_counter()
    head, calc = load_head(args.head), load_targets(args.targets)
    with np.load(args.targets) as targets:
        target = float(np.asarray(targets["pwin"]).ravel()[0]) if args.target_fraction is None else args.target_fraction
    assert 0 < target < 1
    assert np.allclose(calc.window, [.05, 1 / 3], rtol=0, atol=1e-14)
    paper_json = json.loads(args.paper.with_suffix(".json").read_text())
    paper = np.load(args.paper)
    theta = axes.default("MIX17")
    assert np.allclose(theta, paper_json["central"]["theta"], rtol=0, atol=1e-12)
    print("Loading the released reference and event functions", flush=True)
    reference = Reference(args.reference, head, calc)
    loaded = time.perf_counter()
    print(f"Loaded {reference.N} events in {loaded-started:.2f} s", flush=True)
    before = mixture.weights(head, reference.A, theta)
    shape, info = mixture.impose(before, reference.inwin, reference.F, calc)
    fitted = time.perf_counter()
    print(f"Reproduced both central shape fits in {fitted-loaded:.2f} s", flush=True)
    factors = factor_vectors(args.factors)[0]
    nch_edges = np.r_[np.arange(0, 48, 2), np.inf]
    nch_bins = np.searchsorted(nch_edges, reference.X[:, 3], side="right") - 1
    assert np.all((nch_bins >= 0) & (nch_bins < len(nch_edges)-1))
    inside, outside = reference.inwin, ~reference.inwin
    complete = (reference.lo >= .05-1e-12) & (reference.hi <= .30+1e-12)
    assert complete.sum() == 12
    rate, rows, histograms, report, checks = {}, {}, {}, {}, {}
    arrays = dict(theta=theta, window=np.asarray(calc.window), target_fraction=target,
                  observable_names=np.asarray(OBS), moment_names=np.asarray(calc.names),
                  thrust_lo=reference.lo, thrust_hi=reference.hi, nch_edges=nch_edges)

    def predictions(w):
        return reference.predictions(w), np.bincount(nch_bins, weights=w, minlength=len(nch_edges)-1)

    for generator in mixture.GENERATORS:
        p = float(shape[generator][inside].sum())
        a, b = target / p, (1-target) / (1-p)
        rate[generator] = shape[generator] * np.where(inside, a, b)
        records = {}
        for stage, weights in (("before", before[generator]), ("shape", shape[generator]), ("rate", rate[generator])):
            row, nch = predictions(weights)
            rows[generator, stage], histograms[generator, stage] = row, nch
            win = float(weights[inside].sum())
            conditional = (weights[inside] @ reference.F) / win
            in_means = (weights[inside] @ reference.X[inside]) / win
            out_means = (weights[outside] @ reference.X[outside]) / (1-win)
            converted_histogram = row[len(OBS):] * factors[generator][len(OBS):]
            records[stage] = dict(
                means=dict(zip(OBS, row[:len(OBS)].tolist())), weights=weight_statistics(weights),
                conditional_moments=conditional.tolist(), conditional_inside_means=in_means.tolist(),
                conditional_outside_means=out_means.tolist(),
                converted_fraction_005_030=float(converted_histogram[complete] @ reference.width[complete]))
            arrays[f"{generator}_{stage}_means"] = row[:len(OBS)]
            arrays[f"{generator}_{stage}_thrust"] = row[len(OBS):]
            arrays[f"{generator}_{stage}_thrust_conv"] = converted_histogram
            arrays[f"{generator}_{stage}_nch"] = nch
            arrays[f"{generator}_{stage}_conditional_moments"] = conditional
        moment_error = rel(records["rate"]["conditional_moments"], records["shape"]["conditional_moments"])
        inside_error = rel(records["rate"]["conditional_inside_means"], records["shape"]["conditional_inside_means"])
        outside_error = rel(records["rate"]["conditional_outside_means"], records["shape"]["conditional_outside_means"])
        ratios = rate[generator] / shape[generator]
        checks[generator] = dict(
            rate_normalization_error=abs(rate[generator].sum()-1),
            target_fraction_error=abs(rate[generator][inside].sum()-target),
            conditional_moments_relative_error=moment_error,
            inside_conditional_means_relative_error=inside_error,
            outside_conditional_means_relative_error=outside_error,
            within_window_rescaling_error=float(np.max(np.abs(ratios[inside]-a))),
            outside_window_rescaling_error=float(np.max(np.abs(ratios[outside]-b))),
            shape_solver_converged=bool(info[generator]["converged"]),
            shape_solver_residual=float(info[generator]["residual"]))
        assert info[generator]["converged"]
        assert checks[generator]["rate_normalization_error"] < 1e-12
        assert checks[generator]["target_fraction_error"] < 1e-12
        assert max(moment_error, inside_error, outside_error) < 2e-12
        assert all(r["weights"]["finite"] and r["weights"]["positive"] for r in records.values())
        for stage, name, prefix in (("before", "generator_only", "h_central_before"), ("shape", "imposed", "h_central")):
            expected_means = [paper_json["central"]["generators"][generator][name][key] for key in OBS]
            expected_hist = np.r_[paper[f"{prefix}_{generator}_low"], paper[f"{prefix}_{generator}"]]
            checks[generator][f"{stage}_means_vs_paper"] = rel(rows[generator, stage][:len(OBS)], expected_means)
            checks[generator][f"{stage}_thrust_vs_paper"] = rel(rows[generator, stage][len(OBS):], expected_hist)
        envelope_path = Path(f"{args.envelope_prefix}_tune_{generator}_lin.npz")
        with np.load(envelope_path) as envelope:
            labels = [str(x) for x in envelope["labels"]]
            columns = np.array([i for i, label in enumerate(labels) if label.startswith("nch_")])
            assert np.array_equal(nch_edges, envelope["nch_edges"])
            for stage, offset in (("before", 0), ("shape", len(labels))):
                checks[generator][f"{stage}_nch_vs_paper"] = rel(histograms[generator, stage], envelope["y_central"][offset+columns])
        assert max(value for key, value in checks[generator].items() if key.endswith("vs_paper")) <= args.tol
        report[generator] = dict(inside_factor=a, outside_factor=b, predictions=records,
                                 shape_solver_iterations=int(info[generator]["iterations"]))

    mixed = {}
    for stage, weights in (("before", before), ("shape", shape), ("rate", rate)):
        w = .5 * (weights["sherpa"] + weights["herwig"])
        # Follow the paper's seven-mean convention: tau_win is the average of the
        # two generator-conditional means. Also record the actual mixture mean.
        row = .5 * (rows["sherpa", stage] + rows["herwig", stage])
        nch = .5 * (histograms["sherpa", stage] + histograms["herwig", stage])
        converted = .5 * (arrays[f"sherpa_{stage}_thrust_conv"] + arrays[f"herwig_{stage}_thrust_conv"])
        mixed[stage] = dict(means=dict(zip(OBS, row[:len(OBS)].tolist())), weights=weight_statistics(w),
                            actual_conditional_tau=float(w[inside] @ reference.tau[inside] / w[inside].sum()),
                            converted_fraction_005_030=float(converted[complete] @ reference.width[complete]))
        arrays[f"mixture_{stage}_means"] = row[:len(OBS)]
        arrays[f"mixture_{stage}_thrust"] = row[len(OBS):]
        arrays[f"mixture_{stage}_thrust_conv"] = converted
        arrays[f"mixture_{stage}_nch"] = nch
    report["mixture"] = dict(predictions=mixed)
    for stage, name in (("before", "generator_only"), ("shape", "imposed")):
        expected = [paper_json["central"][name][key] for key in OBS]
        checks[f"mixture_{stage}_means_vs_paper"] = rel(arrays[f"mixture_{stage}_means"], expected)
        assert checks[f"mixture_{stage}_means_vs_paper"] <= args.tol
    evaluated = time.perf_counter()
    sources = (args.reference, args.head, args.targets, args.paper, args.paper.with_suffix(".json"), args.factors)
    provenance = {str(path): dict(bytes=path.stat().st_size, sha256=sha256_file(path)) for path in sources}
    result = dict(
        experiment="Sequential hard rate matching after the existing soft conditional-moment fit",
        number_of_reference_events=reference.N, theta=theta.tolist(), window=list(calc.window), target_fraction=target,
        final_state="Common generator event definition; DELPHI factors applied only to reported converted histograms.",
        normalization="Weights sum to one over all reference events. No new events or training.",
        mixture_window_mean="The seven-mean vector follows the paper's average of generator-conditional tau means; actual mixture conditional tau is also recorded.",
        full_mean_order=["thrust", "B_total", "rho_heavy", "nch", "nbaryon"],
        predictions=report, checks=checks, source_files=provenance,
        timing_seconds=dict(load=loaded-started, central_shape_fit=fitted-loaded,
                            statistics_and_validation=evaluated-fitted, total=time.perf_counter()-started),
        baseline_relative_tolerance=args.tol, passed=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "weight_check.npz", **arrays)
    if args.save_weights:
        np.savez_compressed(args.output / "event_weights.npz", **{
            f"{generator}_{stage}": weights[generator]
            for stage, weights in (("before", before), ("shape", shape), ("rate", rate))
            for generator in mixture.GENERATORS})
    (args.output / "weight_check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=True, output=str(args.output), timing_seconds=result["timing_seconds"],
                         target_fraction=target, means={g: {s: r["means"] for s, r in entry["predictions"].items()}
                                                       for g, entry in report.items()}), indent=2), flush=True)


if __name__ == "__main__":
    main()
