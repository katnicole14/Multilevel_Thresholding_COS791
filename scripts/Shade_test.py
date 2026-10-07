from functools import partial
from Histogram.Process import load_grayscale, build_histogram
from Objective_functions.otsu import precompute_cumulative, otsu_fitness
from algorithms.SelectMainLP import run_de  # your standard DE, for comparison
from algorithms.Shade import run_shade, run_lshade
from pathlib import Path

import numpy as np

if __name__ == "__main__":

    gray = load_grayscale(Path("BDS500/img1.png"))
    hist, pdf = build_histogram(gray)
    P, S = precompute_cumulative(pdf)
    objective = partial(otsu_fitness, P=P, S=S)

    K = 5
    POP = 50
    NFES_BUDGET = 60 * POP  # equalize total function evaluations across algorithms
    SEED = 0

    # every algorithm starts from the same initial population
    initial_population = np.random.default_rng(SEED).uniform(1, 254, size=(POP, K))

    de = run_de(initial_population, objective, SEED, NFES_BUDGET)
    sh = run_shade(initial_population, objective, SEED, NFES_BUDGET)
    # L-SHADE (shrinking population, same total NFES budget)
    ls = run_lshade(initial_population, objective, SEED, NFES_BUDGET)

    print(f"NFES budget = {NFES_BUDGET}")
    print(f"Standard DE : fitness={de['best_fitness']:.4f}  thresholds={de['best_thresholds']}")
    print(f"SHADE       : fitness={sh['best_fitness']:.4f}  thresholds={sh['best_thresholds']}")
    print(f"L-SHADE     : fitness={ls['best_fitness']:.4f}  thresholds={ls['best_thresholds']}")
    print()
    print("L-SHADE convergence (nfes, best_fitness) every ~10th checkpoint:")
    for nfes_pt, fit_pt in ls["convergence_history"][::10]:
        print(f"  nfes={nfes_pt:<6d} best_fitness={fit_pt:.2f}")
