#Selecton +Main Loop
"""
Standard DE (DE/rand/1/bin) for multilevel thresholding.

DE's selection rule (standard "greedy" selection):
    Compare the trial vector's fitness against the ORIGINAL
    individual it was derived from (not the whole population).
    Keep whichever one is better; discard the other.
"""
import numpy as np

# Support both ``import algorithms.SelectMainLP`` and running directly.
try:
    from .crossover import crossover
    from .mutation import mutate
    from .threshold_repair import repair_thresholds
except ImportError:
    from crossover import crossover
    from mutation import mutate
    from threshold_repair import repair_thresholds


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


def run_de(
    initial_population,
    objective_function,
    seed,
    maximum_function_evaluations,
    scale_factor=0.5,
    crossover_rate=0.9,
    levels=256,
):
    """Run one reproducible standard DE (DE/rand/1/bin) experiment.

    Same interface as run_jade / run_lade: objective_function takes one
    repaired integer threshold vector and returns a score to MAXIMISE, and
    the run stops after exactly maximum_function_evaluations evaluations.

    Returns a dict with seed, best_thresholds, best_fitness, generations,
    function_evaluations and convergence_history (list of
    (FEs used, best-so-far fitness), one entry per generation).
    """
    random_generator = np.random.default_rng(seed)
    population = np.asarray(initial_population, dtype=float)

    if population.ndim != 2:
        raise ValueError("initial_population must have shape (NP, K).")
    if len(population) < 4:
        raise ValueError("DE/rand/1 requires at least four population members.")
    if maximum_function_evaluations < len(population):
        raise ValueError("The FE budget must cover the initial population.")

    population_size = len(population)

    # Thresholds are stored as integers so every objective (Kapur indexes
    # the histogram with them) receives valid grey levels.
    population = np.array(
        [repair_thresholds(row, levels) for row in population], dtype=int
    )

    fitness_values = np.empty(population_size, dtype=float)
    function_evaluations = 0
    for index in range(population_size):
        fitness_values[index] = objective_function(population[index])
        function_evaluations += 1

    generation = 0
    best_index = int(np.argmax(fitness_values))
    best_solution = population[best_index].copy()
    best_fitness = float(fitness_values[best_index])
    convergence_history = [(function_evaluations, best_fitness)]

    while function_evaluations < maximum_function_evaluations:
        for i in range(population_size):
            if function_evaluations >= maximum_function_evaluations:
                break

            mutant = mutate(
                population, i, scale_factor,
                bounds=(1, levels - 2), rng=random_generator,
            )
            trial = crossover(
                current_solution=population[i],
                mutant=mutant,
                crossover_rate=crossover_rate,
                random_generator=random_generator,
                levels=levels,
            )
            trial = repair_thresholds(trial, levels)
            trial_fitness = float(objective_function(trial))
            function_evaluations += 1

            winner, winner_fitness = Selection(
                population[i], fitness_values[i], trial, trial_fitness
            )
            population[i] = winner
            fitness_values[i] = winner_fitness

        generation += 1
        generation_best_index = int(np.argmax(fitness_values))
        if fitness_values[generation_best_index] > best_fitness:
            best_solution = population[generation_best_index].copy()
            best_fitness = float(fitness_values[generation_best_index])
        convergence_history.append((function_evaluations, best_fitness))

    return {
        "seed": int(seed),
        "best_thresholds": best_solution.astype(int).tolist(),
        "best_fitness": best_fitness,
        "generations": generation,
        "function_evaluations": function_evaluations,
        "convergence_history": convergence_history,
    }
