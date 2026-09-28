from SelectionMainLoop import SelectMainLP
from Objective_functions import Otsu
from Histogram import Process
import numpy as np
from functools import partial

if __name__ == '__main__':
    K = 3

    # one image at a time (your generator, or a single path for now)
    for name, gray, hist, pdf in Process.iter_dataset_images("tests/BDS500"):
        P, S = Otsu.precompute_cumulative(pdf)          # once per image
        objective = partial(Otsu.otsu_fitness, P=P, S=S)

        np.random.seed(0)
        best_sol, best_fit, history = SelectMainLP.de_main_loop(
            dims=K,
            pop_size=20,
            bounds=(0, 255),
            max_generations=50,
            objective_fn=objective,
            seed=0,
        )

        thresholds = Otsu.repair_thresholds(best_sol)   # the real answer

        print(f"{name}: thresholds = {list(thresholds)}  fitness = {best_fit:.2f}")