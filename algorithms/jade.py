#JADE for multilevel image thresholding.



import numpy as np

# Support both ``import algorithms.jade`` and running jade.py directly.
try:
    from .crossover import crossover
    from .threshold_repair import repair_thresholds
except ImportError:
    from crossover import crossover
    from threshold_repair import repair_thresholds


def generate_scale_factor(mean_scale_factor, random_generator):
    """Sample F from Cauchy(mean_F, 0.1), as required by JADE."""
    while True:
        scale_factor = mean_scale_factor + 0.1 * random_generator.standard_cauchy()
        if scale_factor > 0:
            return min(float(scale_factor), 1.0)


def generate_crossover_rate(mean_crossover_rate, random_generator):
    """Sample CR from Normal(mean_CR, 0.1) and restrict it to [0, 1]."""
    crossover_rate = random_generator.normal(mean_crossover_rate, 0.1)
    return float(np.clip(crossover_rate, 0.0, 1.0))


def select_pbest_index(fitness_values, p_best_rate, random_generator):
    """Randomly select one index from the best p proportion of solutions."""
    population_size = len(fitness_values)

    if population_size == 0:
        raise ValueError("Fitness values cannot be empty.")
    if not 0 < p_best_rate <= 1:
        raise ValueError("p_best_rate must be greater than 0 and at most 1.")

    # Otsu, Kapur and Tsallis are maximised, so sort from largest to smallest.
    ranked_indices = np.argsort(-np.asarray(fitness_values, dtype=float))
    pbest_count = max(2, int(np.ceil(p_best_rate * population_size)))
    pbest_count = min(pbest_count, population_size)
    eligible_indices = ranked_indices[:pbest_count]

    return int(random_generator.choice(eligible_indices))


def current_to_pbest_mutation(
    population,
    fitness_values,
    archive,
    current_index,
    scale_factor,
    p_best_rate,
    random_generator,
    bounds=(1, 254),
):
    """Create a mutant with JADE's current-to-pbest/1 strategy.

    v_i = x_i + F_i(x_pbest - x_i) + F_i(x_r1 - x_r2)

    r1 comes from the population. r2 comes from population union archive.
    """
    population = np.asarray(population, dtype=float)
    population_size = len(population)

    if population.ndim != 2:
        raise ValueError("Population must be a two-dimensional array.")
    if population_size < 4:
        raise ValueError("JADE requires at least four population members.")
    if len(fitness_values) != population_size:
        raise ValueError("Each population member needs one fitness value.")
    if not 0 <= current_index < population_size:
        raise ValueError("current_index is outside the population.")

    pbest_index = select_pbest_index(
        fitness_values, p_best_rate, random_generator
    )

    # r1 must be a population member different from the current solution.
    r1_candidates = [i for i in range(population_size) if i != current_index]
    r1_index = int(random_generator.choice(r1_candidates))

    # r2 may come from the population or archive. Population choices must be
    # distinct from both the current solution and r1.
    r2_candidates = []
    for index in range(population_size):
        if index != current_index and index != r1_index:
            r2_candidates.append(population[index])
    for archived_solution in archive:
        r2_candidates.append(np.asarray(archived_solution, dtype=float))

    r2_position = int(random_generator.integers(0, len(r2_candidates)))

    current_solution = population[current_index]
    pbest_solution = population[pbest_index]
    r1_solution = population[r1_index]
    r2_solution = r2_candidates[r2_position]

    mutant = (
        current_solution
        + scale_factor * (pbest_solution - current_solution)
        + scale_factor * (r1_solution - r2_solution)
    )

    # At this stage only repair the intensity bounds. Integer, ordering and
    # duplicate repair is performed after crossover by repair_thresholds().
    lower_bound, upper_bound = bounds
    return np.clip(mutant, lower_bound, upper_bound)


def update_parameter_means(
    mean_scale_factor,
    mean_crossover_rate,
    successful_scale_factors,
    successful_crossover_rates,
    learning_rate,
):
    """Update mean F with the Lehmer mean and mean CR with arithmetic mean."""
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be greater than 0 and at most 1.")

    if successful_scale_factors:
        successful_f = np.asarray(successful_scale_factors, dtype=float)
        lehmer_mean = np.sum(successful_f ** 2) / np.sum(successful_f)
        mean_scale_factor = (
            (1 - learning_rate) * mean_scale_factor
            + learning_rate * lehmer_mean
        )

    if successful_crossover_rates:
        arithmetic_mean = float(np.mean(successful_crossover_rates))
        mean_crossover_rate = (
            (1 - learning_rate) * mean_crossover_rate
            + learning_rate * arithmetic_mean
        )

    return float(mean_scale_factor), float(mean_crossover_rate)


def trim_archive(archive, maximum_size, random_generator):
    """Randomly remove archive entries until its size is at most NP."""
    while len(archive) > maximum_size:
        remove_index = int(random_generator.integers(0, len(archive)))
        archive.pop(remove_index)


def run_jade(
    initial_population,
    objective_function,
    seed,
    maximum_function_evaluations,
    p_best_rate=0.1,
    learning_rate=0.1,
    initial_mean_scale_factor=0.5,
    initial_mean_crossover_rate=0.5,
    levels=256,
):
    """Run one reproducible JADE experiment.

    objective_function must accept one repaired threshold vector and return a
    score to maximise. The result records the seed, best thresholds, best
    fitness, final adaptive parameters and convergence history.
    """
    random_generator = np.random.default_rng(seed)
    population = np.asarray(initial_population, dtype=float)

    if population.ndim != 2:
        raise ValueError("initial_population must have shape (NP, K).")
    if len(population) < 4:
        raise ValueError("JADE requires at least four population members.")
    if maximum_function_evaluations < len(population):
        raise ValueError("The FE budget must cover the initial population.")

    population_size = len(population)

    # Keep K unchanged while converting every initial candidate into a valid
    # ordered set of image thresholds. They are stored as integers so every
    # objective (Kapur indexes the histogram with them) gets valid grey levels.
    population = np.array(
        [repair_thresholds(row, levels) for row in population], dtype=int
    )

    fitness_values = np.empty(population_size, dtype=float)
    function_evaluations = 0
    for index in range(population_size):
        fitness_values[index] = objective_function(population[index])
        function_evaluations += 1

    mean_scale_factor = float(initial_mean_scale_factor)
    mean_crossover_rate = float(initial_mean_crossover_rate)
    archive = []
    generation = 0

    best_index = int(np.argmax(fitness_values))
    best_solution = population[best_index].copy()
    best_fitness = float(fitness_values[best_index])
    convergence_history = [(function_evaluations, best_fitness)]

    while function_evaluations < maximum_function_evaluations:
        # Synchronous update: all mutations in this generation use the same
        # original population rather than partially updated solutions.
        next_population = population.copy()
        next_fitness = fitness_values.copy()
        successful_f = []
        successful_cr = []

        for current_index in range(population_size):
            if function_evaluations >= maximum_function_evaluations:
                break

            scale_factor = generate_scale_factor(
                mean_scale_factor, random_generator
            )
            crossover_rate = generate_crossover_rate(
                mean_crossover_rate, random_generator
            )

            mutant = current_to_pbest_mutation(
                population=population,
                fitness_values=fitness_values,
                archive=archive,
                current_index=current_index,
                scale_factor=scale_factor,
                p_best_rate=p_best_rate,
                random_generator=random_generator,
                bounds=(1, levels - 2),
            )

            trial = crossover(
                current_solution=population[current_index],
                mutant=mutant,
                crossover_rate=crossover_rate,
                random_generator=random_generator,
                levels=levels,
            )

            # This second repair is harmless if crossover already repairs and
            # protects the integration if crossover later changes.
            trial = repair_thresholds(trial, levels)
            trial_fitness = float(objective_function(trial))
            function_evaluations += 1

            # All assignment objectives are maximised. Ties replace the parent,
            # but only a strict improvement archives the parent and counts as
            # a success for the F/CR adaptation.
            if trial_fitness >= fitness_values[current_index]:
                if trial_fitness > fitness_values[current_index]:
                    archive.append(population[current_index].copy())
                    successful_f.append(scale_factor)
                    successful_cr.append(crossover_rate)
                next_population[current_index] = trial
                next_fitness[current_index] = trial_fitness

        population = next_population
        fitness_values = next_fitness
        trim_archive(archive, population_size, random_generator)

        mean_scale_factor, mean_crossover_rate = update_parameter_means(
            mean_scale_factor,
            mean_crossover_rate,
            successful_f,
            successful_cr,
            learning_rate,
        )

        generation += 1
        generation_best_index = int(np.argmax(fitness_values))
        generation_best_fitness = float(fitness_values[generation_best_index])
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_solution = population[generation_best_index].copy()
        convergence_history.append((function_evaluations, best_fitness))

    return {
        "seed": int(seed),
        "best_thresholds": best_solution.astype(int).tolist(),
        "best_fitness": best_fitness,
        "generations": generation,
        "function_evaluations": function_evaluations,
        "final_mean_scale_factor": mean_scale_factor,
        "final_mean_crossover_rate": mean_crossover_rate,
        "archive_size": len(archive),
        "convergence_history": convergence_history,
    }