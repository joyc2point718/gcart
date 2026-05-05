"""
Aggregate per-seed JSON results into mean +/- std tables and learning-curve
plots.

Reads:
    results/<model_name>_s<seed>.json

Writes:
    results/summary_clean.csv
    results/summary_corruption.csv
    results/table_main.md            (markdown table for the paper)
    results/learning_curves.png      (clean accuracy across epochs, all models)
    results/corruption_curves.png    (accuracy vs severity, per corruption)
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def load_results(results_dir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(results_dir, "*_s*.json")))
    rows = []
    for f in files:
        try:
            with open(f) as fp:
                rows.append(json.load(fp))
        except Exception as e:
            print(f"[skip] {f}: {e}")
    return rows


def _ms(values):
    """mean and sample std (or 0 if n<2)."""
    if not values:
        return float("nan"), 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def aggregate_clean(rows):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["config"]["model_name"]].append(r["final_clean_acc"])
    summary = {}
    for m, accs in by_model.items():
        mean, std = _ms(accs)
        summary[m] = {"n": len(accs), "mean": mean, "std": std}
    return summary


def aggregate_corruption(rows):
    """Returns dict: model -> corruption_key -> (mean, std, n)."""
    by_model = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = r["config"]["model_name"]
        for k, v in r["final_corruption_acc"].items():
            by_model[m][k].append(v)
    summary = {}
    for m, d in by_model.items():
        summary[m] = {}
        for k, vs in d.items():
            mean, std = _ms(vs)
            summary[m][k] = {"mean": mean, "std": std, "n": len(vs)}
    return summary


def _avg_over_severities(corr_summary, model, corruption_name):
    """Mean across severities 1..5 of the per-severity means."""
    means = []
    for sev in range(1, 6):
        key = f"{corruption_name}_s{sev}"
        if key in corr_summary[model]:
            means.append(corr_summary[model][key]["mean"])
    return statistics.fmean(means) if means else float("nan")


def write_main_table(clean_summary, corr_summary, out_md):
    """Markdown table: model | params | clean | brightness avg | contrast avg | darken avg"""
    # collect param counts from any one of the rows for each model
    # (we don't have param counts in summary, so we re-load)
    lines = []
    lines.append("| Model | Clean | Brightness (avg s1-5) | Contrast (avg s1-5) | Darken (avg s1-5) |")
    lines.append("|---|---|---|---|---|")
    # Order: baseline, classical_*, zerodce, zerodcepp, gcart, ablations
    preferred_order = [
        "baseline",
        "classical_identity",
        "classical_he",
        "classical_clahe",
        "classical_gamma_1.5",
        "classical_gamma_2.2",
        "classical_gamma_3.0",
        "zerodce",
        "zerodcepp",
        "gcart_no_mono",
        "gcart_hardhist",
        "gcart_poly",
        "gcart_lut",
        "gcart",
    ]
    seen = set()
    models_in_results = set(clean_summary.keys()) | set(corr_summary.keys())

    def _row(m):
        clean = clean_summary.get(m, {"mean": float("nan"), "std": 0.0, "n": 0})
        bright = _avg_over_severities(corr_summary, m, "brightness") \
                 if m in corr_summary else float("nan")
        contrast = _avg_over_severities(corr_summary, m, "contrast") \
                   if m in corr_summary else float("nan")
        darken = _avg_over_severities(corr_summary, m, "darken") \
                 if m in corr_summary else float("nan")
        return (
            f"| {m} "
            f"| {clean['mean']:.2f} +/- {clean['std']:.2f} (n={clean['n']}) "
            f"| {bright:.2f} | {contrast:.2f} | {darken:.2f} |"
        )

    for m in preferred_order:
        if m in models_in_results:
            lines.append(_row(m))
            seen.add(m)
    for m in sorted(models_in_results):
        if m not in seen:
            lines.append(_row(m))

    table = "\n".join(lines)
    with open(out_md, "w") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nSaved -> {out_md}")


def write_severity_table(corr_summary, out_md):
    """Detailed per-severity table (one corruption type at a time)."""
    out_lines = []
    for cname in ("brightness", "contrast", "darken"):
        out_lines.append(f"\n### Corruption: {cname}\n")
        header = "| Model | s=1 | s=2 | s=3 | s=4 | s=5 |"
        sep    = "|---|---|---|---|---|---|"
        out_lines.append(header)
        out_lines.append(sep)
        for m in sorted(corr_summary.keys()):
            row = [m]
            for sev in range(1, 6):
                key = f"{cname}_s{sev}"
                d = corr_summary[m].get(key)
                if d is None:
                    row.append("-")
                else:
                    row.append(f"{d['mean']:.2f}+/-{d['std']:.2f}")
            out_lines.append("| " + " | ".join(row) + " |")
    with open(out_md, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Saved -> {out_md}")


def plot_learning_curves(rows, out_png):
    """Plot clean-accuracy learning curves, averaged across seeds per model."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["config"]["model_name"]].append(r["history"]["clean_acc"])
    if not by_model:
        return
    plt.figure(figsize=(8, 5))
    for m, runs in by_model.items():
        # truncate to common length in case some runs were shorter
        L = min(len(r) for r in runs)
        runs = [r[:L] for r in runs]
        runs = list(zip(*runs))
        means = [statistics.fmean(s) for s in runs]
        stds = [statistics.pstdev(s) if len(s) > 1 else 0.0 for s in runs]
        epochs = list(range(1, L + 1))
        plt.plot(epochs, means, label=m, lw=1.5)
        if max(stds) > 0:
            plt.fill_between(
                epochs,
                [m_ - s_ for m_, s_ in zip(means, stds)],
                [m_ + s_ for m_, s_ in zip(means, stds)],
                alpha=0.15,
            )
    plt.xlabel("Epoch")
    plt.ylabel("Clean test accuracy (%)")
    plt.title("Learning curves (mean +/- std over seeds)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved -> {out_png}")


def plot_corruption_curves(corr_summary, out_png):
    """Accuracy vs severity for each corruption type, one subplot each."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, cname in zip(axes, ("brightness", "contrast", "darken")):
        for m in sorted(corr_summary.keys()):
            sevs = list(range(1, 6))
            vals = []
            errs = []
            for sev in sevs:
                key = f"{cname}_s{sev}"
                d = corr_summary[m].get(key)
                if d is None:
                    vals.append(float("nan"))
                    errs.append(0.0)
                else:
                    vals.append(d["mean"])
                    errs.append(d["std"])
            ax.errorbar(sevs, vals, yerr=errs, label=m, marker="o", lw=1.2,
                        capsize=2, markersize=4)
        ax.set_title(f"CIFAR-10-C-{cname}")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved -> {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No results in {args.results_dir}")
        return
    print(f"Loaded {len(rows)} run(s)")

    clean_summary = aggregate_clean(rows)
    corr_summary = aggregate_corruption(rows)

    write_main_table(
        clean_summary, corr_summary,
        os.path.join(args.results_dir, "table_main.md"),
    )
    write_severity_table(
        corr_summary,
        os.path.join(args.results_dir, "table_severities.md"),
    )
    plot_learning_curves(
        rows, os.path.join(args.results_dir, "learning_curves.png")
    )
    plot_corruption_curves(
        corr_summary, os.path.join(args.results_dir, "corruption_curves.png")
    )


if __name__ == "__main__":
    main()
