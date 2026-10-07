from functools import partial
from Histogram.Process import load_grayscale, build_histogram
from Objective_functions.otsu import precompute_cumulative, otsu_fitness
from algorithms.threshold_repair import repair_thresholds
from algorithms.SelectMainLP import de_main_loop  # your standard DE, for comparison
from algorithms.Shade import shade,l_shade
from pathlib import Path 

import numpy as np

if __name__ == "__main__":

    gray = load_grayscale(Path("data/BDS500/img1.png"))
    hist, pdf = build_histogram(gray)
    P, S = precompute_cumulative(pdf)
    objective = partial(otsu_fitness, P=P, S=S)
 
    K = 5
    GENS = 60
    POP = 1000
    NFES_BUDGET = GENS * POP  # equalize total function evaluations across algorithms
 
    # standard DE
    np.random.seed(0)
    best_de, fit_de, hist_de = de_main_loop(
        dims=K, pop_size=POP, bounds=(1, 254),
        max_generations=GENS, objective_fn=objective
    )
 
    # SHADE (fixed population)
    best_sh, fit_sh, hist_sh = shade(
        dims=K, objective_fn=objective, pop_size=POP, bounds=(1, 254),
        max_generations=GENS, seed=0
    )
 
    # L-SHADE (shrinking population, same total NFES budget)
    best_ls, fit_ls, hist_ls = l_shade(
        dims=K, objective_fn=objective, max_nfes=NFES_BUDGET,
        pop_size_init=POP, pop_size_min=4, bounds=(1, 254), seed=0
    )
 
    print(f"NFES budget = {NFES_BUDGET}")
    print(f"Standard DE : fitness={fit_de:.4f}  thresholds={list(repair_thresholds(best_de))}")
    print(f"SHADE       : fitness={fit_sh:.4f}  thresholds={list(repair_thresholds(best_sh))}")
    print(f"L-SHADE     : fitness={fit_ls:.4f}  thresholds={list(repair_thresholds(best_ls))}")
    print()
    print("L-SHADE convergence (nfes, best_fitness) every ~10th checkpoint:")
    for nfes_pt, fit_pt in hist_ls[::10]:
        print(f"  nfes={nfes_pt:<6d} best_fitness={fit_pt:.2f}")
