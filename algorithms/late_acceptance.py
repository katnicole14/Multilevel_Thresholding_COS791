"""
late_acceptance.py

Late Acceptance Differential Evolution (LADE) for multilevel thresholding.

LADE is standard DE/rand/1/bin with DE's greedy one-to-one selection
replaced by the Late Acceptance (LA) rule of Burke & Bykov's Late
Acceptance Hill Climbing. A trial u_i is accepted if it is at least as
good as EITHER its target x_i OR the fitness value stored L generations
ago in the history buffer:

    x_i^{G+1} = u_i^G   if f(u_i^G) >= f(x_i^G) or f(u_i^G) >= H[G mod L]
                x_i^G   otherwise

(written with >= because Otsu, Kapur and Tsallis are all maximised).
After the decision, the target's current fitness is written back into
H[G mod L]. Because H[G mod L] holds an older, usually worse, fitness,
trials that are slightly worse than the current target can still be
accepted, which keeps diversity and helps escape local optima.

Each population member keeps its own circular history buffer, i.e. the
buffer has shape (L, NP) and member i reads/writes column i. This is the
direct translation of LAHC's single-solution history to a population.
With history_length=1 the history only ever holds f(x_i), so LADE
reduces exactly to standard greedy DE.

Since accepted solutions can be worse than their parents, the best-ever
solution is tracked separately from the population.

References
----------
Burke, E. K., & Bykov, Y. (2017). The late acceptance hill-climbing
heuristic. European Journal of Operational Research, 258(1), 70-78.
Storn, R., & Price, K. (1997). Differential Evolution - a simple and
efficient heuristic for global optimization over continuous spaces.
Journal of Global Optimization, 11(4), 341-359.
"""

import numpy as np

# Support both ``import algorithms.late_acceptance`` and running directly.
try:
    from .crossover import crossover
    from .mutation import mutate
    from .threshold_repair import repair_thresholds
except ImportError:
    from crossover import crossover
    from mutation import mutate
    from threshold_repair import repair_thresholds


def late_acceptance_select(trial_fitness, target_fitness, late_fitness):
    """
    Late Acceptance selection rule (Eq. 2 of the background, maximising).

    Returns True if the trial should replace the target, i.e. if it is
    at least as good as the target or as the fitness from L generations ago.
    """
    return trial_fitness >= target_fitness or trial_fitness >= late_fitness


def run_lade(
    initial_population,
    objective_function,
    seed,
    maximum_function_evaluations,
    scale_factor=0.5,
    crossover_rate=0.9,
    history_length=10,
    levels=256,
):
    """Run one reproducible Late Acceptance DE experiment.

    Parameters
    ----------
    initial_population : array-like, shape (NP, K)
        Starting threshold vectors. Repaired into valid thresholds first.
    objective_function : callable(thresholds) -> float
        Score to MAXIMISE (Otsu, Kapur or Tsallis bound to one image).
    seed : int
        Seed for numpy's default_rng, for reproducible runs.
    maximum_function_evaluations : int
        FE budget, shared with the other DE variants for a fair comparison.
    scale_factor : float, default 0.5
        DE mutation scale factor F.
    crossover_rate : float, default 0.9
        DE binomial crossover rate CR.
    history_length : int, default 10
        L, the number of generations kept in the late acceptance history.
    levels : int, default 256
        Number of grey levels in the image.

    Returns
    -------
    dict with the same keys as run_jade where they apply: seed,
    best_thresholds, best_fitness, generations, function_evaluations,
    convergence_history (list of (FEs used, best-so-far fitness)), plus
    late_acceptances (number of trials accepted ONLY because of the
    late acceptance rule, i.e. worse than their target).
    """
    random_generator = np.random.default_rng(seed)
    population = np.asarray(initial_population, dtype=float)

    if population.ndim != 2:
        raise ValueError("initial_population must have shape (NP, K).")
    if len(population) < 4:
        raise ValueError("DE/rand/1 requires at least four population members.")
    if history_length < 1:
        raise ValueError("history_length must be at least 1.")
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

    # H[slot, i] = fitness of member i recorded L generations ago.
    fitness_history = np.tile(fitness_values, (history_length, 1))
    late_acceptances = 0
    generation = 0

    best_index = int(np.argmax(fitness_values))
    best_solution = population[best_index].copy()
    best_fitness = float(fitness_values[best_index])
    convergence_history = [(function_evaluations, best_fitness)]

    while function_evaluations < maximum_function_evaluations:
        # Synchronous update: every mutant in this generation is built from
        # the same population, as in run_jade.
        next_population = population.copy()
        next_fitness = fitness_values.copy()
        history_slot = generation % history_length

        for current_index in range(population_size):
            if function_evaluations >= maximum_function_evaluations:
                break

            mutant = mutate(
                population,
                current_index,
                scale_factor,
                bounds=(1, levels - 2),
                rng=random_generator,
            )
            trial = crossover(
                current_solution=population[current_index],
                mutant=mutant,
                crossover_rate=crossover_rate,
                random_generator=random_generator,
                levels=levels,
            )
            trial = repair_thresholds(trial, levels)
            trial_fitness = float(objective_function(trial))
            function_evaluations += 1

            target_fitness = fitness_values[current_index]
            late_fitness = fitness_history[history_slot, current_index]

            if late_acceptance_select(trial_fitness, target_fitness, late_fitness):
                if trial_fitness < target_fitness:
                    late_acceptances += 1
                next_population[current_index] = trial
                next_fitness[current_index] = trial_fitness

                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()

            # Write the member's current fitness back into H[G mod L].
            fitness_history[history_slot, current_index] = next_fitness[current_index]

        population = next_population
        fitness_values = next_fitness
        generation += 1
        convergence_history.append((function_evaluations, best_fitness))

    return {
        "seed": int(seed),
        "best_thresholds": np.asarray(best_solution).astype(int).tolist(),
        "best_fitness": best_fitness,
        "generations": generation,
        "function_evaluations": function_evaluations,
        "history_length": history_length,
        "late_acceptances": late_acceptances,
        "convergence_history": convergence_history,
    }
