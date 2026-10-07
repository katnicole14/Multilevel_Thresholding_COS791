from algorithms import SelectMainLP
from Objective_functions import otsu
from Histogram import Process
import numpy as np
from functools import partial

if __name__ == '__main__':
    K = 3
    POP = 20
    SEED = 0

    # one image at a time (your generator, or a single path for now)
    for name, gray, hist, pdf in Process.iter_dataset_images("BDS500"):
        P, S = otsu.precompute_cumulative(pdf)          # once per image
        objective = partial(otsu.otsu_fitness, P=P, S=S)

        initial_population = np.random.default_rng(SEED).uniform(1, 254, size=(POP, K))
        result = SelectMainLP.run_de(
            initial_population,
            objective,
            seed=SEED,
            maximum_function_evaluations=50 * POP,
        )

        print(f"{name}: thresholds = {result['best_thresholds']}  fitness = {result['best_fitness']:.2f}")
