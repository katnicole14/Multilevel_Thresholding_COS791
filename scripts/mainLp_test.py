from algorithms import SelectMainLP
from Objective_functions import otsu
from Histogram import Process
import numpy as np
from functools import partial

if __name__ == '__main__':
    K = 3

    # one image at a time (your generator, or a single path for now)
    for name, gray, hist, pdf in Process.iter_dataset_images("BDS500"):
        P, S = otsu.precompute_cumulative(pdf)          # once per image
        objective = partial(otsu.otsu_fitness, P=P, S=S)

        np.random.seed(0)
        best_sol, best_fit, history = SelectMainLP.de_main_loop(
            dims=K,
            pop_size=20,
            bounds=(0, 255),
            max_generations=50,
            objective_fn=objective,
            seed=0,
        )

        thresholds = otsu.repair_thresholds(best_sol)   # the real answer

        print(f"{name}: thresholds = {list(thresholds)}  fitness = {best_fit:.2f}")