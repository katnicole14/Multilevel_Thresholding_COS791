"""
verify_optimum.py

Correctness check against the TRUE optimum. With K=2 every pair of
thresholds (about 32,000) can be tried, so the best possible fitness is
known exactly. Every DE variant should reach it for every objective.

Usage (from the repo root, images in ./BDS500):
    python scripts/verify_optimum.py
    python scripts/verify_optimum.py --image img4 --folder BDS500
"""

import argparse
import itertools
import sys
from functools import partial
from pathlib import Path

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
    "Standard DE": run_de,
    "JADE": run_jade,
    "SHADE": run_shade,
    "L-SHADE": run_lshade,
    "LADE": run_lade,
}
K = 2
POP_SIZE = 20
MAX_FES = 2000
SEEDS = range(5)


def brute_force(objective):
    best_fitness, best_thresholds = -np.inf, None
    for t in itertools.combinations(range(1, 255), K):
        fitness = objective(np.array(t))
        if fitness > best_fitness:
            best_fitness, best_thresholds = fitness, list(t)
    return best_thresholds, best_fitness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="img1")
    parser.add_argument("--folder", default="BDS500")
    args = parser.parse_args()

    image_path = REPO_ROOT / args.folder / f"{args.image}.png"
    hist, pdf = Process.build_histogram(Process.load_grayscale(image_path))
    P, S = precompute_cumulative(pdf)
    objectives = {
        "Otsu": partial(otsu_fitness, P=P, S=S),
        "Kapur": partial(kapur_entropy, probabilities=pdf),
        "Tsallis": lambda t: tsallis(t, hist, 0.8),
    }

    all_ok = True
    for objective_name, objective in objectives.items():
        print(f"\n{objective_name}: brute force over all {K}-threshold combinations ...", flush=True)
        optimum_t, optimum_f = brute_force(objective)
        print(f"  TRUE optimum   {optimum_t}  fitness {optimum_f:.6f}")

        for algorithm_name, run in ALGORITHMS.items():
            hits = 0
            for seed in SEEDS:
                population = np.random.default_rng(seed).uniform(1, 254, (POP_SIZE, K))
                result = run(population, objective, seed, MAX_FES)
                hits += np.isclose(result["best_fitness"], optimum_f)
            ok = hits == len(SEEDS)
            all_ok &= ok
            print(f"  {algorithm_name:12s} reached optimum in {hits}/{len(SEEDS)} runs  "
                  f"{'PASS' if ok else 'FAIL'}  (last: {result['best_thresholds']})")

    print("\nALL PASS" if all_ok else "\nSOME FAILED")


if __name__ == "__main__":
    main()
