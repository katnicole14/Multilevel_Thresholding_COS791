"""
SHADE and L-SHADE for multilevel image thresholding.

Both use DE/current-to-pbest/1 with an external archive, binomial
crossover, and a historical memory of successful (F, CR) pairs. L-SHADE
also shrinks the population linearly over the FE budget.

Both share the run_jade / run_lade interface: they start from a given
initial population, MAXIMISE objective_function on repaired integer
thresholds, stop after exactly maximum_function_evaluations evaluations,
and return a result dict.
"""
import numpy as np

# Support both ``import algorithms.Shade`` and running directly.
try:
    from .threshold_repair import repair_thresholds
except ImportError:
    from threshold_repair import repair_thresholds


def sample_parameters(M_F, M_CR, rng):
    """Pick a random memory slot r, then F ~ Cauchy(M_F[r], 0.1), CR ~ N(M_CR[r], 0.1)."""
    r = rng.integers(0, len(M_F))
    F_i = -1
    while F_i <= 0:  # Cauchy can produce negative/zero -> resample
        F_i = rng.standard_cauchy() * 0.1 + M_F[r]
    F_i = min(F_i, 1.0)
    CR_i = float(np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0))
    return float(F_i), CR_i


def current_to_pbest_mutant(population, archive, i, pbest_pool, F_i, rng, lo, hi):
    """
    v_i = x_i + F_i (x_pbest - x_i) + F_i (x_r1 - x_r2)

    r1 is a population member != i; r2 comes from population UNION archive
    and, when taken from the population, is != i and != r1.
    """
    NP = len(population)
    pbest_idx = pbest_pool[rng.integers(0, len(pbest_pool))]

    r1 = rng.choice([idx for idx in range(NP) if idx != i])

    r2_candidates = [population[idx] for idx in range(NP) if idx != i and idx != r1]
    r2_candidates.extend(archive)
    x_r2 = r2_candidates[rng.integers(0, len(r2_candidates))]

    x = population.astype(float)
    mutant = x[i] + F_i * (x[pbest_idx] - x[i]) + F_i * (x[r1] - x_r2)
    return np.clip(mutant, lo, hi)


def binomial_crossover(target, mutant, CR_i, rng):
    """Take mutant[j] with probability CR_i (and always at j_rand), else target[j]."""
    trial = np.asarray(target, dtype=float).copy()  # float, so mutant values aren't truncated
    j_rand = rng.integers(0, len(trial))
    for j in range(len(trial)):
        if rng.random() < CR_i or j == j_rand:
            trial[j] = mutant[j]
    return trial


def improvement_weights(improvements):
    """
    Normalised weights |f(u) - f(x)| for the memory update.

    Kapur returns -inf when a class is empty, so leaving such a solution is
    an infinite improvement. Those successes then share the weight equally
    (instead of producing inf/inf = NaN and corrupting the memory).
    """
    w = np.asarray(improvements, dtype=float)
    if np.isinf(w).any():
        w = np.isinf(w).astype(float)
    return w / w.sum()


def weighted_lehmer_mean(values, weights):
    return np.sum(weights * values ** 2) / np.sum(weights * values)


def pbest_pool_of(fitness, p):
    """Indices of the top max(2, round(p * NP)) individuals (maximising)."""
    n_pbest = min(len(fitness), max(2, int(round(p * len(fitness)))))
    return np.argsort(fitness)[-n_pbest:]


def initialise(initial_population, objective_function, levels):
    population = np.asarray(initial_population, dtype=float)
    if population.ndim != 2:
        raise ValueError("initial_population must have shape (NP, K).")
    if len(population) < 4:
        raise ValueError("current-to-pbest/1 requires at least four population members.")

    # Thresholds are stored as integers so every objective (Kapur indexes
    # the histogram with them) receives valid grey levels.
    population = np.array(
        [repair_thresholds(row, levels) for row in population], dtype=int
    )
    fitness = np.array([float(objective_function(ind)) for ind in population])
    return population, fitness


def run_shade(
    initial_population,
    objective_function,
    seed,
    maximum_function_evaluations,
    memory_size=None,   # H; defaults to NP
    p=0.1,              # top p% used for pbest (e.g. 0.1 = top 10%)
    levels=256,
):
    """Run one reproducible SHADE experiment. Archive size is capped at NP."""
    rng = np.random.default_rng(seed)
    lo, hi = 1, levels - 2

    if maximum_function_evaluations < len(initial_population):
        raise ValueError("The FE budget must cover the initial population.")

    population, fitness = initialise(initial_population, objective_function, levels)
    pop_size = len(population)
    nfes = pop_size

    H = memory_size if memory_size is not None else pop_size
    # --- init memory: start at 0.5 for both F and CR (neutral guess) ---
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    memory_pos = 0  # circular pointer into the memory arrays
    archive = []

    best_idx = int(np.argmax(fitness))
    best_solution = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    history = [(nfes, best_fitness)]
    generation = 0

    while nfes < maximum_function_evaluations:
        S_F, S_CR, S_weights = [], [], []  # successful (F,CR) pairs this gen

        new_population = population.copy()
        new_fitness = fitness.copy()
        pbest_pool = pbest_pool_of(fitness, p)

        for i in range(pop_size):
            if nfes >= maximum_function_evaluations:
                break

            F_i, CR_i = sample_parameters(M_F, M_CR, rng)
            mutant = current_to_pbest_mutant(population, archive, i, pbest_pool, F_i, rng, lo, hi)
            trial = repair_thresholds(binomial_crossover(population[i], mutant, CR_i, rng), levels)

            trial_fitness = float(objective_function(trial))
            nfes += 1

            # --- selection: ties replace the parent, strict improvements
            # archive the parent and count as successes ---
            if trial_fitness >= fitness[i]:
                if trial_fitness > fitness[i]:
                    archive.append(population[i].copy())
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_weights.append(trial_fitness - fitness[i])  # improvement size
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        population, fitness = new_population, new_fitness
        while len(archive) > pop_size:
            archive.pop(rng.integers(0, len(archive)))

        # --- update memory (only if at least one success this generation) ---
        if S_F:
            w = improvement_weights(S_weights)  # normalize weights
            # weighted Lehmer mean for F, weighted arithmetic mean for CR
            M_F[memory_pos] = weighted_lehmer_mean(np.array(S_F), w)
            M_CR[memory_pos] = np.sum(w * np.array(S_CR))
            memory_pos = (memory_pos + 1) % H

        generation += 1
        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fitness:
            best_solution = population[gen_best_idx].copy()
            best_fitness = float(fitness[gen_best_idx])
        history.append((nfes, best_fitness))

    return {
        "seed": int(seed),
        "best_thresholds": best_solution.astype(int).tolist(),
        "best_fitness": best_fitness,
        "generations": generation,
        "function_evaluations": nfes,
        "archive_size": len(archive),
        "convergence_history": history,
    }


def run_lshade(
    initial_population,
    objective_function,
    seed,
    maximum_function_evaluations,
    pop_size_min=4,      # NP_min; DE/current-to-pbest/1 needs at least 4
    memory_size=6,       # L-SHADE uses a SMALLER memory than SHADE
    p=0.11,
    arch_rate=2.6,       # archive capacity = round(arch_rate * NP)
    levels=256,
):
    """
    Run one reproducible L-SHADE experiment: SHADE + Linear Population
    Size Reduction (LPSR) + external archive, driven by the FE budget.

    NP_init is the size of initial_population (L-SHADE normally uses
    18 * K). After every generation the population shrinks linearly
    towards pop_size_min by dropping the worst individuals.
    """
    rng = np.random.default_rng(seed)
    lo, hi = 1, levels - 2

    if maximum_function_evaluations < len(initial_population):
        raise ValueError("The FE budget must cover the initial population.")

    population, fitness = initialise(initial_population, objective_function, levels)
    NP_init = NP = len(population)
    nfes = NP  # initial population evaluation counts too

    H = memory_size
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    memory_pos = 0

    archive = []
    arch_size_max = int(round(arch_rate * NP_init))

    best_idx = int(np.argmax(fitness))
    best_solution = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    history = [(nfes, best_fitness)]
    generation = 0

    while nfes < maximum_function_evaluations:
        S_F, S_CR, S_weights = [], [], []
        new_population = population.copy()
        new_fitness = fitness.copy()
        pbest_pool = pbest_pool_of(fitness, p)

        for i in range(NP):
            if nfes >= maximum_function_evaluations:
                break

            F_i, CR_i = sample_parameters(M_F, M_CR, rng)
            mutant = current_to_pbest_mutant(population, archive, i, pbest_pool, F_i, rng, lo, hi)
            trial = repair_thresholds(binomial_crossover(population[i], mutant, CR_i, rng), levels)

            trial_fitness = float(objective_function(trial))
            nfes += 1

            if trial_fitness >= fitness[i]:
                if trial_fitness > fitness[i]:
                    # archive the replaced (losing) target before overwriting
                    archive.append(population[i].copy())
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    S_weights.append(trial_fitness - fitness[i])
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        population, fitness = new_population, new_fitness
        while len(archive) > arch_size_max:
            archive.pop(rng.integers(0, len(archive)))

        if S_F:
            w = improvement_weights(S_weights)
            # L-SHADE uses the weighted Lehmer mean for both F and CR
            M_F[memory_pos] = weighted_lehmer_mean(np.array(S_F), w)
            S_CR = np.array(S_CR)
            M_CR[memory_pos] = weighted_lehmer_mean(S_CR, w) if S_CR.sum() > 0 else 0.0
            memory_pos = (memory_pos + 1) % H

        # --- Linear Population Size Reduction (LPSR) ---
        NP_new = int(round(((pop_size_min - NP_init) / maximum_function_evaluations) * nfes + NP_init))
        NP_new = max(pop_size_min, NP_new)
        if NP_new < NP:
            # drop the worst (NP - NP_new) individuals
            keep = np.argsort(fitness)[-NP_new:]
            population = population[keep]
            fitness = fitness[keep]
            NP = NP_new
            arch_size_max = int(round(arch_rate * NP))
            while len(archive) > arch_size_max:
                archive.pop(rng.integers(0, len(archive)))

        generation += 1
        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fitness:
            best_solution = population[gen_best_idx].copy()
            best_fitness = float(fitness[gen_best_idx])
        history.append((nfes, best_fitness))

    return {
        "seed": int(seed),
        "best_thresholds": best_solution.astype(int).tolist(),
        "best_fitness": best_fitness,
        "generations": generation,
        "function_evaluations": nfes,
        "final_population_size": NP,
        "archive_size": len(archive),
        "convergence_history": history,
    }
