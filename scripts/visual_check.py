"""
visual_check.py

Look at the results by eye. For one image it runs one DE variant with each
objective (Otsu, Kapur, Tsallis) and saves a figure with:
  row 1: original image + segmented image per objective
         (every pixel replaced by the mean grey level of its class)
  row 2: histogram with the chosen thresholds as vertical lines
         + convergence curve per objective

What "correct" looks like:
  - thresholds sit in the valleys between histogram peaks (Otsu especially)
  - segmented images keep the main shapes/regions of the original
  - more thresholds (larger K) -> segmented image closer to the original
  - convergence curves only go up and flatten out before the end

Usage (from the repo root, images in ./BDS500):
    python scripts/visual_check.py
    python scripts/visual_check.py --image img4 --k 12 --algorithm JADE
    python scripts/visual_check.py --folder CHAOS --image <slice name>

Saves results/visual_<image>_K<k>_<algorithm>.png
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Histogram import Process
from Objective_functions.otsu import otsu_fitness, precompute_cumulative
from Objective_functions.Kapur import kapur_entropy
from Objective_functions.tsallis import tsallis
from algorithms.jade import run_jade
from algorithms.late_acceptance import run_lade
from algorithms.Shade import run_shade, run_lshade
from algorithms.SelectMainLP import run_de

ALGORITHMS = {
    "DE": run_de,
    "JADE": run_jade,
    "SHADE": run_shade,
    "LSHADE": run_lshade,
    "LADE": run_lade,
}


def segment(gray, thresholds):
    """Replace every pixel by the mean grey level of its class [0..t1], [t1+1..t2], ..."""
    labels = np.digitize(gray, np.asarray(thresholds) + 1)
    out = np.zeros_like(gray, dtype=float)
    for c in np.unique(labels):
        out[labels == c] = gray[labels == c].mean()
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="img1")
    parser.add_argument("--folder", default="BDS500")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--algorithm", default="LSHADE", choices=ALGORITHMS)
    parser.add_argument("--fes", type=int, default=3000, help="function evaluation budget")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image_path = REPO_ROOT / args.folder / f"{args.image}.png"
    gray = Process.load_grayscale(image_path)
    hist, pdf = Process.build_histogram(gray)
    P, S = precompute_cumulative(pdf)
    objectives = {
        "Otsu": partial(otsu_fitness, P=P, S=S),
        "Kapur": partial(kapur_entropy, probabilities=pdf),
        "Tsallis (q=0.8)": lambda t: tsallis(t, hist, 0.8),
    }

    run = ALGORITHMS[args.algorithm]
    population = np.random.default_rng(args.seed).uniform(1, 254, (30, args.k))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Original {args.image}")
    axes[1, 0].bar(range(256), hist, width=1, color="0.6")
    axes[1, 0].set_title("Histogram")

    print(f"{args.image}  K={args.k}  {args.algorithm}  {args.fes} FEs  seed {args.seed}\n")
    for column, (name, objective) in enumerate(objectives.items(), start=1):
        result = run(population, objective, args.seed, args.fes)
        t = result["best_thresholds"]
        print(f"{name:16s} thresholds {t}  fitness {result['best_fitness']:.4f}")

        axes[0, column].imshow(segment(gray, t), cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"{name}\n{t}", fontsize=9)

        for x in t:
            axes[1, 0].axvline(x, color=f"C{column}", linewidth=1, alpha=0.8)

        fes, best = zip(*result["convergence_history"])
        axes[1, column].plot(fes, best, color=f"C{column}")
        axes[1, column].set_title(f"{name} convergence")
        axes[1, column].set_xlabel("function evaluations")
        axes[1, column].set_ylabel("best fitness")

    for ax in axes[0]:
        ax.axis("off")
    axes[1, 0].legend(
        handles=[plt.Line2D([], [], color=f"C{i}") for i in range(1, 4)],
        labels=list(objectives), fontsize=8,
    )
    fig.suptitle(f"{args.image}, K={args.k}, {args.algorithm}")
    fig.tight_layout()

    out = REPO_ROOT / "results" / f"visual_{args.image}_K{args.k}_{args.algorithm}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nSaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
