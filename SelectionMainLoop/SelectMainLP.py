#Selecton +Main Loop
"""
DE's selection rule (standard "greedy" selection):
    Compare the trial vector's fitness against the ORIGINAL
    individual it was derived from (not the whole population).
    Keep whichever one is better; discard the other.
"""
from algorithms import crossover
from algorithms import mutation
import pandas as pd
import numpy  as np


def Selection(target: np.ndarray,
    target_fitness: float,
    trial: np.ndarray,
    trial_fitness: float,
    maximize: bool = True,
    ) -> tuple[np.ndarray, float]:
    """
    Greedy one-to-one DE selection.

    Parameters
    ----------
    target : the original individual (current population member)
    target_fitness : its already-computed fitness value
    trial : the candidate produced by mutation + crossover
    trial_fitness : the trial's fitness value
    maximize : True for Otsu/Kapur/Tsallis (we want to MAXIMIZE variance/entropy)

    Returns
    -------
    (winner_vector, winner_fitness) -- whichever of target/trial is better.
    Ties go to the trial (standard DE convention -- helps exploration).
    """
    if maximize:
        if trial_fitness >= target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness
    else:
        if trial_fitness <= target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

def fake_objective(t: np.ndarray) -> float:
    """
    Placeholder fitness function -- NOT real Otsu.
    Just something cheap so Person B can test that the LOOP mechanics
    (mutation -> crossover -> selection -> best-tracking) work correctly,
    before Person B's real otsu_fitness() is wired in.

    Here: reward vectors whose values are close to sorted & spread out
    (a stand-in "sanity" fitness so we can visually check convergence).
    """
    t_sorted = np.sort(t)
    spread = np.sum(np.diff(t_sorted))  # reward spacing between thresholds
    return spread


def de_main_loop(
    dims: int,
    pop_size: int = 20,
    bounds: tuple[float, float] = (0, 255),
    F: float = 0.5,
    CR: float = 0.9,
    max_generations: int = 50,
    objective_fn=fake_objective,
    maximize: bool = True,
    seed=None,
):
    """
    Minimal standard DE (DE/rand/1/bin) main loop.
    This is the skeleton Person B is responsible for in Week 2.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    population = np.random.uniform(lo, hi, size=(pop_size, dims))
    fitness = np.array([objective_fn(ind) for ind in population])

    best_idx = np.argmax(fitness) if maximize else np.argmin(fitness)
    best_solution, best_fitness = population[best_idx].copy(), fitness[best_idx]

    history = [best_fitness]  # track convergence for plotting later

    for gen in range(max_generations):
        for i in range(pop_size):
            mutant = mutation.mutate(population, i, F, bounds=(lo,hi),rng=rng) 
            mutant = np.clip(mutant, lo, hi)
            trial = crossover.crossover(population[i], mutant, CR,random_generator=rng)  
            trial = np.clip(trial, lo, hi)

            trial_fitness = objective_fn(trial)

           
            winner, winner_fitness = Selection(
                population[i], fitness[i], trial, trial_fitness, maximize
            )
            population[i] = winner
            fitness[i] = winner_fitness

        gen_best_idx = np.argmax(fitness) if maximize else np.argmin(fitness)
        if (maximize and fitness[gen_best_idx] > best_fitness) or \
           (not maximize and fitness[gen_best_idx] < best_fitness):
            best_solution = population[gen_best_idx].copy()
            best_fitness = fitness[gen_best_idx]

        history.append(best_fitness)

    return best_solution, best_fitness, history