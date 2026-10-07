"""
late_acceptance.py

Late Acceptance Hill Climbing (LAHC) for multilevel thresholding.

LAHC is a single-solution local search metaheuristic. Instead of only
comparing a candidate against the CURRENT solution (as classic hill
climbing does), it also compares against the fitness recorded L
iterations ago. This makes the search tolerant of short sequences of
non-improving moves, which helps it escape local optima without the
extra machinery of simulated annealing's cooling schedule.

Reference: Burke, E. K., & Bykov, Y. (2017). The late acceptance
hill-climbing heuristic. European Journal of Operational Research,
258(1), 70-78.
"""

import numpy as np

from Objective_functions.otsu import repair_thresholds


def _fix_length(thresholds, target_length, levels):
    """
    Work around a known bug in Objective_functions.otsu.repair_thresholds:
    when perturbation collapses two thresholds onto the same value, it
    can return fewer than `target_length` values instead of nudging them
    apart. late_acceptance doesn't own that file, so instead of relying
    on it to preserve length, pad back up here by inserting a value into
    the widest remaining gap until the vector is the right length again.
    """
    values = list(np.atleast_1d(thresholds))

    while len(values) < target_length:
        bounds = [0] + values + [levels - 1]
        gaps = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        widest = int(np.argmax(gaps))
        new_value = (bounds[widest] + bounds[widest + 1]) // 2
        new_value = max(1, min(new_value, levels - 2))

        if new_value in values:
            break  # no room left to insert a distinct value; give up

        values.append(new_value)
        values.sort()

    return np.array(values[:target_length], dtype=int)


def _perturb(solution, levels, random_generator, step_size):
    """
    Produce a neighbour of `solution` by nudging ONE randomly chosen
    threshold by a random integer offset in [-step_size, step_size],
    then repairing it back into a valid, sorted, duplicate-free vector
    of the same length as `solution`.
    """
    neighbour = np.array(solution, dtype=int, copy=True)
    target_length = len(neighbour)

    position = random_generator.integers(0, target_length)
    offset = random_generator.integers(-step_size, step_size + 1)
    neighbour[position] += offset

    repaired = repair_thresholds(neighbour, levels)
    return _fix_length(repaired, target_length, levels)


def late_acceptance_hill_climbing(
    initial_solution,
    fitness_function,
    history_length,
    max_iterations,
    random_generator,
    levels=256,
    step_size=10,
    maximize=True,
):
    """
    Run Late Acceptance Hill Climbing starting from `initial_solution`.

    Parameters
    ----------
    initial_solution : array-like
        Starting threshold vector (length K).
    fitness_function : callable(thresholds) -> float
        Objective to optimise (e.g. otsu_fitness bound to an image's
        P/S via functools.partial, or kapur_entropy bound to its pdf).
    history_length : int
        L, the number of past iterations' fitness values kept for
        late acceptance comparisons. Must be >= 1.
    max_iterations : int
        Number of candidate moves to attempt. Must be >= 0.
    random_generator : numpy.random.Generator
        Source of randomness, for reproducible runs.
    levels : int, default 256
        Number of grey levels in the image (L in repair_thresholds).
    step_size : int, default 10
        Max absolute perturbation applied to one threshold per move.
    maximize : bool, default True
        True to maximise fitness (Otsu/Kapur/Tsallis all maximise),
        False to minimise.

    Returns
    -------
    best_solution : np.ndarray
        Best threshold vector found.
    best_fitness : float
        Fitness of best_solution.
    convergence : list[float]
        Best-so-far fitness value after every iteration, useful for
        convergence plots.

    Raises
    ------
    ValueError
        If initial_solution is empty, history_length < 1, or
        max_iterations < 0.
    """
    if len(initial_solution) == 0:
        raise ValueError("Initial solution array is empty.")

    if history_length < 1:
        raise ValueError("History length must be at least 1.")

    if max_iterations < 0:
        raise ValueError("Max iterations must be non-negative.")

    def is_better_or_equal(a, b):
        return a >= b if maximize else a <= b

    def is_better(a, b):
        return a > b if maximize else a < b

    target_length = len(initial_solution)
    current_solution = _fix_length(
        repair_thresholds(initial_solution, levels), target_length, levels
    )
    current_fitness = fitness_function(current_solution)

    best_solution = current_solution.copy()
    best_fitness = current_fitness

    # Late acceptance history: fitness values from L iterations ago.
    fitness_history = [current_fitness] * history_length
    convergence = []

    for iteration in range(max_iterations):
        candidate_solution = _perturb(
            current_solution, levels, random_generator, step_size
        )
        candidate_fitness = fitness_function(candidate_solution)

        history_slot = iteration % history_length

        # Accept if the candidate beats either the current solution
        # or the solution from `history_length` iterations ago.
        if is_better_or_equal(
            candidate_fitness, fitness_history[history_slot]
        ) or is_better_or_equal(candidate_fitness, current_fitness):
            current_solution = candidate_solution
            current_fitness = candidate_fitness

        fitness_history[history_slot] = current_fitness

        if is_better(current_fitness, best_fitness):
            best_solution = current_solution.copy()
            best_fitness = current_fitness

        convergence.append(best_fitness)

    return best_solution, best_fitness, convergence
