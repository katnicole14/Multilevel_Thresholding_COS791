"""
figures_experiment1.py

Report figures for Experiment 1, built from <results>/runs.jsonl:

  segmentation_<image>.png   original vs segmented image for each objective
                             and K (one algorithm, its median-fitness run)
  convergence_<image>.png    mean best-so-far fitness vs function evaluations,
                             one panel per objective x K, one line per algorithm
  time_vs_K.png              mean run time vs K per algorithm (scalability)

Usage:
    python experiments/figures_experiment1.py                      # all images
    python experiments/figures_experiment1.py --images img1 --algorithm JADE
    python experiments/figures_experiment1.py --results results/experiment1_quick
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Histogram import Process
from evaluation.segmentation import segment

ALGORITHM_ORDER = ["DE", "JADE", "SHADE", "L-SHADE", "LADE"]
OBJECTIVE_ORDER = ["Otsu", "Kapur", "Tsallis"]
# Fixed colour per algorithm in every figure (validated categorical palette,
# slots 1-5) plus a line style each, so lines stay distinguishable in
# greyscale print and for colour-blind readers.
COLOURS = {"DE": "#2a78d6", "JADE": "#eb6834", "SHADE": "#1baf7a", "L-SHADE": "#eda100", "LADE": "#e87ba4"}
LINESTYLES = {"DE": "-", "JADE": "--", "SHADE": "-.", "L-SHADE": ":", "LADE": (0, (5, 1, 1, 1, 1, 1))}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load_runs(results_dir):
    with (results_dir / "runs.jsonl").open() as f:
        return pd.DataFrame(json.loads(line) for line in f if line.strip())


def present(order, values):
    return [v for v in order if v in set(values)]


def segmentation_figure(runs, image, algorithm, images_dir, out):
    """Rows = K, columns = original + one per objective (median-fitness run)."""
    gray = Process.crop_white_border(Process.load_grayscale(images_dir / f"{image}.png"))
    subset = runs[(runs["image"] == image) & (runs["algorithm"] == algorithm)]
    ks = sorted(subset["K"].unique())
    objectives = present(OBJECTIVE_ORDER, subset["objective"])

    fig, axes = plt.subplots(len(ks), len(objectives) + 1,
                             figsize=(2.1 * (len(objectives) + 1), 1.9 * len(ks)), squeeze=False)
    for row, k in enumerate(ks):
        axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].set_ylabel(f"K = {k}", rotation=0, ha="right", va="center", fontsize=9)
        for col, objective in enumerate(objectives, start=1):
            group = subset[(subset["K"] == k) & (subset["objective"] == objective)]
            median_run = group.iloc[(group["best_fitness"] - group["best_fitness"].median()).abs().argmin()]
            ax = axes[row, col]
            ax.imshow(segment(gray, median_run["thresholds"]), cmap="gray", vmin=0, vmax=255)
            title = f"{objective}\n" if row == 0 else ""
            ax.set_title(f"{title}PSNR {median_run['psnr']:.2f} dB · SSIM {median_run['ssim']:.3f}",
                         fontsize=6.5, color=MUTED if row else INK)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{image}: original vs segmented ({algorithm}, median run)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def convergence_figure(runs, image, max_fes, out):
    """Panels = objective x K; mean best-so-far fitness over runs per algorithm."""
    subset = runs[runs["image"] == image]
    ks = sorted(subset["K"].unique())
    objectives = present(OBJECTIVE_ORDER, subset["objective"])
    algorithms = present(ALGORITHM_ORDER, subset["algorithm"])

    fig, axes = plt.subplots(len(objectives), len(ks), figsize=(2.3 * len(ks), 2.0 * len(objectives)),
                             squeeze=False)
    for row, objective in enumerate(objectives):
        for col, k in enumerate(ks):
            ax = axes[row, col]
            for algorithm in algorithms:
                group = subset[(subset["objective"] == objective) & (subset["K"] == k)
                               & (subset["algorithm"] == algorithm)]
                curves = np.array([[np.nan if v is None else v for v in c] for c in group["convergence"]],
                                  dtype=float)
                fes = np.linspace(max_fes / curves.shape[1], max_fes, curves.shape[1])
                # checkpoints before the initial population is evaluated are NaN
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_curve = np.nanmean(curves, axis=0)
                ax.plot(fes, mean_curve, color=COLOURS[algorithm],
                        linestyle=LINESTYLES[algorithm], label=algorithm)
            ax.set_title(f"{objective}, K = {k}")
            ax.ticklabel_format(axis="y", useOffset=False)
            if row == len(objectives) - 1:
                ax.set_xlabel("Function evaluations")
            if col == 0:
                ax.set_ylabel("Mean best fitness")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Convergence on {image} (mean over runs)", y=1.06, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def time_figure(runs, out):
    """Small multiples per objective: mean run time vs K, one line per algorithm."""
    objectives = present(OBJECTIVE_ORDER, runs["objective"])
    algorithms = present(ALGORITHM_ORDER, runs["algorithm"])
    mean_time = runs.groupby(["objective", "algorithm", "K"])["time_s"].mean()

    fig, axes = plt.subplots(1, len(objectives), figsize=(2.6 * len(objectives), 2.3),
                             sharey=True, squeeze=False)
    for ax, objective in zip(axes[0], objectives):
        for algorithm in algorithms:
            series = mean_time.loc[objective, algorithm]
            ax.plot(series.index, series.values, color=COLOURS[algorithm],
                    linestyle=LINESTYLES[algorithm], marker="o", markersize=3.5, label=algorithm)
        ax.set_title(objective)
        ax.set_xlabel("Number of thresholds K")
        ax.set_xticks(sorted(runs["K"].unique()))
    axes[0, 0].set_ylabel("Mean run time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment1")
    parser.add_argument("--images-dir", default="BDS500")
    parser.add_argument("--images", nargs="*", help="images to draw (default: all in the results)")
    parser.add_argument("--algorithm", default="L-SHADE", help="algorithm for the segmentation figures")
    args = parser.parse_args()

    results_dir = REPO_ROOT / args.results
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results_dir)
    max_fes = int(runs["max_fes"].iloc[0])

    for image in args.images or sorted(runs["image"].unique(), key=lambda s: (len(s), s)):
        segmentation_figure(runs, image, args.algorithm, REPO_ROOT / args.images_dir,
                            figures_dir / f"segmentation_{image}.png")
        convergence_figure(runs, image, max_fes, figures_dir / f"convergence_{image}.png")
        print(f"  {image}: segmentation + convergence")
    time_figure(runs, figures_dir / "time_vs_K.png")
    print(f"Figures written to {figures_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
